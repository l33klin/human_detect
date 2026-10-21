"""Only reuse verified empty results; occupied scenes retain the temporal policy."""

import time
from datetime import datetime, timezone

import cv2
import numpy as np

from human_detect.schemas import DetectionResult
from human_detect.temporal import TemporalResult, TemporalSession


class MotionResult(TemporalResult):
    motion: dict


class EmptySceneSession:
    def __init__(self, pipeline):
        self.pipeline, self.cfg = pipeline, pipeline.settings
        self.temporal = TemporalSession(pipeline)
        self.empty_gray = None
        self.empty_at = self.previous_at = None
        self.empty_size = None
        self.store_revision = pipeline.store.revision

    @staticmethod
    def gray(rgb):
        h, w = rgb.shape[:2]
        scale = min(1, 320 / w, 240 / h)
        small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (3, 3), 0)

    def analyze(self, rgb, captured_at=None):
        started = time.perf_counter()
        self.pipeline._check()
        captured_at = captured_at or datetime.now(timezone.utc)
        now = captured_at.timestamp()
        size = tuple(rgb.shape[:2])
        revision = self.pipeline.store.revision
        invalid = (
            self.previous_at is None
            or not 0 < now - self.previous_at <= self.cfg.temporal_detection_seconds
            or size != self.empty_size
            or revision != self.store_revision
        )
        gray = self.gray(rgb)
        changed = None
        if self.empty_gray is not None and not invalid:
            # Compare with the last TPU-verified empty frame, not the previous frame:
            # small incremental changes must accumulate instead of being absorbed.
            changed = float(
                np.mean(cv2.absdiff(gray, self.empty_gray) >= self.cfg.motion_pixel_threshold)
            )
        moved = changed is None or changed >= self.cfg.motion_changed_fraction
        interval = self.cfg.empty_scene_seconds
        idle = (
            not invalid
            and self.empty_at is not None
            and now - self.empty_at < interval
            and not moved
        )
        if idle:
            result = TemporalResult(
                **DetectionResult(
                    captured_at=captured_at,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    persons=[],
                    processing_ms=0,
                ).model_dump(),
                temporal=dict(
                    mode="idle",
                    detection_age_ms=round((now - self.empty_at) * 1000, 2),
                    identity_age_ms=0,
                    identity_source="none",
                ),
            )
            reason = "verified_empty"
        else:
            if invalid:
                # Reconnects, timestamp gaps, size and identity-store changes reset
                # both the motion anchor and all temporal identity assumptions.
                self.temporal = TemporalSession(self.pipeline)
            result = self.temporal.analyze(rgb, captured_at)
            reason = "reset" if invalid else "motion" if moved else "periodic"
            if result.persons:
                self.empty_at, self.empty_gray = None, None
                reason = "occupied"
            else:
                self.empty_at = now
                self.empty_gray = gray
        self.previous_at, self.empty_size, self.store_revision = now, size, revision
        values = result.model_dump()
        values["processing_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return MotionResult(
            **values,
            motion=dict(
                reason=reason,
                changed_fraction=changed,
                motion_enabled=True,
                empty_scene_seconds=interval,
            ),
        )
