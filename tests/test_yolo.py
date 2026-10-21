import numpy as np
import pytest

from human_detect.yolo import YoloDflDecoder


def outputs():
    scores = np.full((1, 2100, 17), -8, np.float32)
    boxes = np.full((1, 2100, 64), -20, np.float32)
    boxes.reshape(1, 2100, 4, 16)[:, :, :, 1] = 20
    return scores, boxes


def test_yolo_dfl_decodes_logits_class_filter_and_clips_to_image():
    scores, boxes = outputs()
    scores[0, 0, 0] = np.log(4)  # sigmoid(log(4)) = 0.8
    scores[0, 20, 1] = 4  # another class must not become a person
    decoder = YoloDflDecoder(320, 320, [scores.shape, boxes.shape])
    detections = decoder.decode([scores, boxes], 640, 320, 0.5, 0)
    assert len(detections) == 1
    box, score = detections[0]
    assert score == pytest.approx(0.8)
    assert box.model_dump() == dict(x1=0, y1=0, x2=24, y2=12)
    assert decoder.decode([scores, boxes], 640, 320, 0.9, 0) == []


def test_yolo_nms_suppresses_duplicates_and_accepts_reversed_outputs():
    scores, boxes = outputs()
    scores[0, 0, 0], scores[0, 1, 0] = 3, 2
    # Large overlapping boxes on neighboring anchors should yield one person.
    boxes[:] = -20
    boxes.reshape(1, 2100, 4, 16)[:, :, :, 4] = 20
    decoder = YoloDflDecoder(320, 320, [boxes.shape, scores.shape])
    detections = decoder.decode([boxes, scores], 320, 320, 0.5, 0)
    assert len(detections) == 1
    boxes[0, 0, 0] = np.nan
    assert len(decoder.decode([boxes, scores], 320, 320, 0.5, 0)) == 1


def test_yolo_rejects_unknown_model_contract():
    with pytest.raises(ValueError, match="pinned YOLOv9"):
        YoloDflDecoder(320, 320, [(1, 2100, 80), (1, 2100, 64)])
