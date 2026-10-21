"""Decode the pinned 17-class YOLOv9 ReLU6 Edge TPU model's DFL outputs.

The compiled graph executes on the TPU. These operations decode its quantized
logits into scored rectangles and suppress overlapping detections on the host.
"""

import cv2
import numpy as np

from human_detect.schemas import Box


class YoloDflDecoder:
    def __init__(self, width, height, shapes):
        if width % 32 or height % 32:
            raise ValueError("YOLO dimensions must be multiples of 32")
        anchors, strides = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[: height // stride, : width // stride]
            anchors.append(np.column_stack((x.ravel(), y.ravel())) + 0.5)
            strides.append(np.full((x.size, 1), stride))
        self.anchors = np.concatenate(anchors).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)
        self.width, self.height = width, height
        expected = {(1, len(self.anchors), 17), (1, len(self.anchors), 64)}
        if len(shapes) != 2 or {tuple(s) for s in shapes} != expected:
            raise ValueError("expected pinned YOLOv9 outputs [1,N,17] and [1,N,64]")
        self.scores_index = next(i for i, shape in enumerate(shapes) if shape[-1] == 17)
        self.boxes_index = 1 - self.scores_index

    def decode(self, values, image_width, image_height, threshold, class_id):
        logits = values[self.scores_index][0]
        classes = logits.argmax(axis=1)
        best = logits[np.arange(len(logits)), classes]
        scores = 1 / (1 + np.exp(-np.clip(best, -80, 80)))
        mask = np.isfinite(best) & (scores >= threshold)
        if class_id is not None:
            mask &= classes == class_id
        if not mask.any():
            return []
        scores, classes = scores[mask], classes[mask]
        distributions = values[self.boxes_index][0][mask].reshape(-1, 4, 16)
        finite = np.isfinite(distributions).all(axis=(1, 2))
        distributions = distributions[finite]
        scores, classes = scores[finite], classes[finite]
        if not len(scores):
            return []
        probabilities = np.exp(distributions - distributions.max(axis=2, keepdims=True))
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = probabilities @ np.arange(16, dtype=np.float32)
        anchors, strides = self.anchors[mask][finite], self.strides[mask][finite]
        corners = np.concatenate(
            ((anchors - distances[:, :2]) * strides, (anchors + distances[:, 2:]) * strides),
            axis=1,
        )
        xywh = np.column_stack((corners[:, :2], corners[:, 2:] - corners[:, :2]))
        selected = []
        for category in np.unique(classes):
            candidates = np.flatnonzero(classes == category)
            kept = cv2.dnn.NMSBoxes(xywh[candidates].tolist(), scores[candidates].tolist(), 0, 0.4)
            selected.extend(candidates[np.asarray(kept, dtype=int).ravel()].tolist())
        result = []
        scale = np.array([image_width / self.width, image_height / self.height] * 2)
        for index in sorted(selected, key=lambda i: float(scores[i]), reverse=True):
            x1, y1, x2, y2 = corners[index] * scale
            box = Box(
                x1=max(0, min(image_width, int(np.floor(x1)))),
                y1=max(0, min(image_height, int(np.floor(y1)))),
                x2=max(0, min(image_width, int(np.ceil(x2)))),
                y2=max(0, min(image_height, int(np.ceil(y2)))),
            )
            if box.x2 > box.x1 and box.y2 > box.y1:
                result.append((box, float(scores[index])))
        return result
