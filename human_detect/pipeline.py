import threading
import time
from datetime import datetime, timezone

from human_detect.runtime import EngineUnavailable
from human_detect.schemas import Box, DetectionResult, Face, Identity, Person


def iou(a, b):
    overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - overlap
    return overlap / union if union else 0


def face_crop(rgb, box):
    h, w = rgb.shape[:2]
    dx, dy = int((box.x2 - box.x1) * 0.1), int((box.y2 - box.y1) * 0.1)
    return rgb[max(0, box.y1 - dy) : min(h, box.y2 + dy), max(0, box.x1 - dx) : min(w, box.x2 + dx)]


class Pipeline:
    def __init__(self, settings, models, store):
        self.settings, self.models, self.store = settings, models, store
        self.lock = threading.Lock()
        self.failure = None

    def _check(self):
        if self.failure:
            raise EngineUnavailable(self.failure)

    def _invoke(self, call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except Exception as exc:
            self.failure = "TPU inference failed; restart service after checking hardware/runtime"
            raise EngineUnavailable(self.failure) from exc

    def analyze(self, rgb, captured_at=None):
        start = time.perf_counter()
        captured_at = captured_at or datetime.now(timezone.utc)
        with self.lock:
            self._check()
            cfg = self.settings
            bodies = self._invoke(self.models.person.detect, rgb, cfg.person_threshold, class_id=0)
            persons = [Person(box=b, confidence=s) for b, s in bodies[: cfg.max_people]]
            candidates = []
            for index, person in enumerate(persons):
                body = person.box
                crop = rgb[body.y1 : body.y2, body.x1 : body.x2]
                for local, score in self._invoke(self.models.face.detect, crop, cfg.face_threshold):
                    box = Box(
                        x1=body.x1 + local.x1,
                        y1=body.y1 + local.y1,
                        x2=body.x1 + local.x2,
                        y2=body.y1 + local.y2,
                    )
                    candidates.append((index, box, score))
            # Overlapping body crops can detect the same face more than once.
            selected = []
            for index, box, score in sorted(candidates, key=lambda item: item[2], reverse=True):
                if any(iou(box, other) > 0.4 for other in selected):
                    continue
                selected.append(box)
                identity = Identity(status="face_too_small")
                if min(box.x2 - box.x1, box.y2 - box.y1) >= cfg.min_face_pixels:
                    vector = self._invoke(self.models.embedding.embed, face_crop(rgb, box))
                    identity = self.store.match(
                        vector, cfg.similarity_threshold, cfg.identity_margin
                    )
                persons[index].faces.append(Face(box=box, confidence=score, identity=identity))
        return DetectionResult(
            captured_at=captured_at,
            width=rgb.shape[1],
            height=rgb.shape[0],
            persons=persons,
            processing_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    def enroll(self, rgb, person_id, name):
        # Registration photos may contain just a face, without a detectable body.
        with self.lock:
            self._check()
            faces = self._invoke(self.models.face.detect, rgb, self.settings.face_threshold)
            if len(faces) != 1:
                raise ValueError("enrollment requires exactly one detectable face")
            box, _ = faces[0]
            if min(box.x2 - box.x1, box.y2 - box.y1) < self.settings.min_face_pixels:
                raise ValueError("face is too small for enrollment")
            vector = self._invoke(self.models.embedding.embed, face_crop(rgb, box))
            return self.store.enroll(person_id, name, vector)
