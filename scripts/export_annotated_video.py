#!/usr/bin/env python3
"""Export every frame of a silent clip for visual model/cadence comparisons.

Uses the service's real TPU pipeline, tracking, annotator, VAAPI decoder/encoder.
Video timestamps drive temporal policy; export speed is NOT live-stream capacity.
"""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from human_detect.annotation import Annotator
from human_detect.config import Settings
from human_detect.identity import IdentityStore
from human_detect.media import MediaConnection
from human_detect.motion import EmptySceneSession
from human_detect.pipeline import Pipeline
from human_detect.runtime import ModelBundle
from human_detect.temporal import TemporalSession


def export(args):
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("choose a new output name")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = Settings(
        _env_file=None,
        models_dir=args.models,
        database=args.database,
        capture_backend="vaapi",
        media_encoder="vaapi",
        media_enabled=True,
        media_inference_fps=args.inference_fps,
        person_threshold=args.person_threshold,
        empty_scene_enabled=not args.disable_empty_scene,
        media_font=args.font,
    )
    models = ModelBundle(cfg)
    store = IdentityStore(cfg.database, models.fingerprint, models.dimension)
    media, frames = None, []
    started = time.monotonic()
    try:
        media = MediaConnection(str(args.clip), str(args.output), cfg, output_format="mp4")
        if media.audio:
            raise ValueError("this comparison exporter supports silent clips only")
        cfg.temporal_detect_every = max(1, math.ceil(float(media.rate) / cfg.media_inference_fps))
        pipeline = Pipeline(cfg, models, store)
        session = (
            EmptySceneSession(pipeline) if cfg.empty_scene_enabled else TemporalSession(pipeline)
        )
        annotator = Annotator(cfg.media_font, max(14, min(24, round(media.video.width / 80))))
        origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for packet in media.input.demux(video=0):
            for frame in packet.decode():
                if frame.pts is None:
                    raise ValueError("source timestamp missing")
                seconds = float(frame.pts * frame.time_base)
                rgb = frame.to_ndarray(format="rgb24")
                result = session.analyze(rgb, origin + timedelta(seconds=seconds))
                value = result.model_dump(mode="json")
                value["source_frame"] = dict(
                    index=len(frames),
                    pts=frame.pts,
                    time_base=str(frame.time_base),
                    seconds=seconds,
                )
                frames.append(value)
                annotator.draw(rgb, value["persons"])
                media.video_frame(rgb, frame.pts, frame.time_base)
        if media.decoder.wait(timeout=10) != 0:
            raise RuntimeError("hardware decoder failed")
        deadline = time.monotonic() + 30
        while media.written < len(frames):
            if media.writer_error or time.monotonic() >= deadline:
                raise RuntimeError("failed to drain annotated output")
            time.sleep(0.02)
        media.writer_stop.set()
        media.writer.join(timeout=3)
        if media.writer.is_alive():
            raise RuntimeError("output writer did not stop")
        media.output.close()
        media.encoder.stdin.close()
        if media.encoder.wait(timeout=30) != 0:
            raise RuntimeError("hardware encoder failed")
        report = dict(
            source_filename=args.clip.name,
            source_sha256=hashlib.sha256(args.clip.read_bytes()).hexdigest(),
            output_filename=args.output.name,
            models=models.report,
            frames=len(frames),
            fps=str(media.rate),
            width=media.video.width,
            height=media.video.height,
            audio=False,
            cpu_nn_fallback=False,
            export="all frames; source timestamps drive temporal cadence; not a live benchmark",
            target_detection_hz=args.inference_fps,
            empty_scene_enabled=cfg.empty_scene_enabled,
            temporal_detect_every=cfg.temporal_detect_every,
            person_threshold=cfg.person_threshold,
            face_threshold=cfg.face_threshold,
            modes=dict(Counter(f["temporal"]["mode"] for f in frames)),
            person_counts=dict(Counter(len(f["persons"]) for f in frames)),
            face_counts=dict(Counter(sum(len(p["faces"]) for p in f["persons"]) for f in frames)),
            elapsed_seconds=time.monotonic() - started,
            results=frames,
        )
        args.output.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        print(
            json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False),
            flush=True,
        )
    finally:
        if media is not None:
            media.close()
        store.close()
        models.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--database", type=Path, default=Path("/tmp/export-identities.sqlite3"))
    parser.add_argument("--inference-fps", type=float, default=3)
    parser.add_argument("--person-threshold", type=float, default=0.5)
    parser.add_argument("--disable-empty-scene", action="store_true")
    parser.add_argument("--font", default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    export(parser.parse_args())
