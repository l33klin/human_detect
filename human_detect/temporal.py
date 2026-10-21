"""Per-stream tracking and bounded reuse of TPU face embeddings.

Each stream owns a session. Cached results carry age metadata and conservative tracking
failures refresh the complete pipeline on TPU.
"""

import time
from datetime import datetime, timezone

import cv2
import numpy as np

from human_detect.pipeline import face_crop, iou
from human_detect.schemas import Box, DetectionResult


def patch(rgb, box):
    return cv2.resize(face_crop(rgb, box), (24, 24)).astype(np.float32) / 255


def appearance_distance(left, right):
    return float(np.mean(np.abs(left - right)))


class EmbeddingCache:
    def __init__(self, ttl):
        self.ttl, self.entries = ttl, []
        self.now, self.hits = 0.0, 0
        self.used = set()

    def begin(self, now):
        self.now, self.hits = now, 0
        self.entries = [e for e in self.entries if 0 <= now - e["at"] < self.ttl]
        self.used.clear()

    def get(self, rgb, box, compute):
        current = patch(rgb, box)
        choices = [
            e
            for e in self.entries
            if id(e) not in self.used
            and iou(box, e["box"]) >= 0.5
            and appearance_distance(current, e["patch"]) < 0.1
        ]
        if len(choices) == 1:
            entry = choices[0]
            entry["box"] = box
            self.used.add(id(entry))
            self.hits += 1
            return entry["vector"]
        vector = compute()
        entry = dict(box=box, patch=current, vector=vector, at=self.now)
        self.entries.append(entry)
        self.used.add(id(entry))
        return vector

    def age(self, now):
        return max((now - e["at"] for e in self.entries), default=0)


class TemporalResult(DetectionResult):
    temporal: dict


def transform_box(box, matrix, width, height):
    points = np.float32([[box.x1, box.y1], [box.x2, box.y1], [box.x1, box.y2], [box.x2, box.y2]])
    transformed = cv2.transform(points[None], matrix)[0]
    x1, y1 = np.floor(transformed.min(axis=0) + 1e-5).astype(int)
    x2, y2 = np.ceil(transformed.max(axis=0) - 1e-5).astype(int)
    result = Box(
        x1=max(0, min(width, x1)),
        y1=max(0, min(height, y1)),
        x2=max(0, min(width, x2)),
        y2=max(0, min(height, y2)),
    )
    return result if result.x2 > result.x1 and result.y2 > result.y1 else None


class TemporalSession:
    def __init__(self, pipeline):
        self.pipeline, self.cfg = pipeline, pipeline.settings
        self.cache = EmbeddingCache(self.cfg.temporal_identity_seconds)
        self.previous = None
        self.last_result = None
        self.points, self.patches = [], []
        self.frame_index, self.detected_index = 0, -100
        self.detected_at, self.previous_at = 0.0, 0.0
        self.store_revision = pipeline.store.revision

    def _gray(self, rgb):
        height, width = rgb.shape[:2]
        scale = min(1, 640 / width, 480 / height)
        small = cv2.resize(rgb, (round(width * scale), round(height * scale)))
        return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    def _seed(self, gray, result):
        gh, gw = gray.shape
        sx, sy = gw / result.width, gh / result.height
        points = []
        for person in result.persons:
            b = person.box
            mask = np.zeros_like(gray)
            mask[int(b.y1 * sy) : int(b.y2 * sy), int(b.x1 * sx) : int(b.x2 * sx)] = 255
            points.append(cv2.goodFeaturesToTrack(gray, 60, 0.01, 5, mask=mask))
        return points

    def _track(self, rgb, gray):
        height, width = rgb.shape[:2]
        gh, gw = gray.shape
        persons = [p.model_copy(deep=True) for p in self.last_result.persons]
        updated = []
        for index, person in enumerate(persons):
            previous_points = self.points[index]
            if previous_points is None or len(previous_points) < 8:
                return None
            new, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous, gray, previous_points, None, winSize=(21, 21), maxLevel=3
            )
            if new is None or status is None:
                return None
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.previous, new, None, winSize=(21, 21), maxLevel=3
            )
            if back is None or back_status is None:
                return None
            valid = (status.ravel() == 1) & (back_status.ravel() == 1)
            valid &= np.linalg.norm(back - previous_points, axis=2).ravel() < 1.5
            if np.count_nonzero(valid) < 8:
                return None
            factor = np.float32([width / gw, height / gh])
            old, current = previous_points[valid, 0] * factor, new[valid, 0] * factor
            matrix, inliers = cv2.estimateAffinePartial2D(old, current, method=cv2.RANSAC)
            if matrix is None or inliers is None or float(np.mean(inliers)) < 0.6:
                return None
            if not 0.85 <= float(np.hypot(matrix[0, 0], matrix[1, 0])) <= 1.15:
                return None
            moved = transform_box(person.box, matrix, width, height)
            if moved is None:
                return None
            person.box = moved
            for face_index, face in enumerate(person.faces):
                box = transform_box(face.box, matrix, width, height)
                if box is None:
                    return None
                if appearance_distance(patch(rgb, box), self.patches[index][face_index]) >= 0.1:
                    return None
                face.box = box
            updated.append(new[valid].copy())
        self.points = updated
        return persons

    def analyze(self, rgb, captured_at=None):
        tick = time.perf_counter()
        self.pipeline._check()
        captured_at = captured_at or datetime.now(timezone.utc)
        now = captured_at.timestamp()
        gray = self._gray(rgb)
        self.frame_index += 1
        revision = self.pipeline.store.revision
        invalid = revision != self.store_revision or (
            self.previous_at
            and not 0 < now - self.previous_at <= self.cfg.temporal_detection_seconds
        )
        if invalid:
            self.cache.entries.clear()
        refresh = (
            invalid
            or self.previous is None
            or self.previous.shape != gray.shape
            or self.last_result is None
            or not self.last_result.persons
            or (self.last_result.width, self.last_result.height) != (rgb.shape[1], rgb.shape[0])
            or self.frame_index - self.detected_index >= self.cfg.temporal_detect_every
            or now - self.detected_at >= self.cfg.temporal_detection_seconds
            or self.cache.age(now) >= self.cfg.temporal_identity_seconds
        )
        if not refresh:
            difference = (
                np.mean(np.abs(gray.astype(np.float32) - self.previous.astype(np.float32))) / 255
            )
            refresh = difference > 0.12
            if refresh:
                self.cache.entries.clear()
        tracked = None if refresh else self._track(rgb, gray)
        if tracked is None:
            self.cache.begin(now)
            result = self.pipeline.analyze(rgb, captured_at, embedding_cache=self.cache)
            self.points = self._seed(gray, result)
            self.patches = [[patch(rgb, f.box) for f in p.faces] for p in result.persons]
            self.detected_at, self.detected_index = now, self.frame_index
            mode = "detected"
        else:
            result = DetectionResult(
                captured_at=captured_at,
                width=rgb.shape[1],
                height=rgb.shape[0],
                persons=tracked,
                processing_ms=0,
            )
            mode = "tracked"
        self.previous, self.previous_at = gray, now
        self.store_revision = revision
        self.last_result = result
        value = result.model_dump()
        value["processing_ms"] = round((time.perf_counter() - tick) * 1000, 2)
        return TemporalResult(
            **value,
            temporal=dict(
                mode=mode,
                detection_age_ms=round((now - self.detected_at) * 1000, 2),
                identity_age_ms=round(self.cache.age(now) * 1000, 2),
                identity_source="cached_embedding"
                if mode == "tracked" or self.cache.hits
                else "current",
            ),
        )
