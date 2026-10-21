"""Controlled empty-scene and actual H.264 decode benchmarks, using public assets only."""

import argparse
import hashlib
import json
import platform
import random
import resource
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from experiments.benchmark import Meter, load_assets, summarize, trace_result
from human_detect.config import Settings
from human_detect.identity import IdentityStore
from human_detect.motion import EmptySceneSession
from human_detect.pipeline import Pipeline
from human_detect.runtime import ModelBundle
from human_detect.temporal import TemporalSession
from human_detect.video import FFmpegCapture

SCENES = ("empty", "entry_exit", "occupied", "slow_entry", "lighting")
POLICIES = ("combined", "motion", "idle", "motion_idle")
BASE = datetime(2026, 10, 21, tzinfo=timezone.utc)


def frame(assets, scene, size, index, count):
    w, h = size
    empty = np.full((h, w, 3), 45, np.uint8)
    if scene == "empty":
        return empty
    if scene == "lighting":
        return np.full_like(empty, 45 + int(25 * (1 + np.sin(index / 8))))
    person = cv2.resize(assets["reference"], size)
    if scene == "occupied":
        return person
    start, end = int(count * 0.22), int(count * 0.72)
    if not start <= index < end:
        return empty
    if scene == "slow_entry":
        return cv2.addWeighted(
            person, min(1, (index - start + 1) / 10), empty, 1 - min(1, (index - start + 1) / 10), 0
        )
    return person


def processor(pipeline, policy):
    pipeline.settings.motion_enabled = policy in ("motion", "motion_idle")
    pipeline.settings.empty_scene_enabled = policy in ("idle", "motion_idle")
    return TemporalSession(pipeline) if policy == "combined" else EmptySceneSession(pipeline)


def analyze_frames(pipeline, meter, assets, scene, size, count, policy, fps):
    for _ in range(5):
        pipeline.analyze(frame(assets, scene, size, 0, count))
    session = processor(pipeline, policy)
    meter.clear()
    timings, trace, reasons, digest = [], [], Counter(), hashlib.sha256()
    for i in range(count):
        rgb = frame(assets, scene, size, i, count)
        digest.update(memoryview(rgb))
        start = time.perf_counter()
        result = session.analyze(rgb, BASE + timedelta(seconds=i / fps))
        timings.append((time.perf_counter() - start) * 1000)
        trace.append(trace_result(result))
        reasons[getattr(result, "motion", {}).get("reason", "temporal")] += 1
    return dict(
        scene=scene,
        size=list(size),
        policy=policy,
        frames=count,
        input_fps=fps,
        corpus_sha256=digest.hexdigest(),
        processing=summarize(timings),
        frame_ms=timings,
        trace=trace,
        reasons=dict(reasons),
        stages=stage_summary(meter),
    )


def stage_summary(meter):
    return {k: dict(calls=len(v), total_ms=sum(v), **summarize(v)) for k, v in meter.calls.items()}


def encode(assets, scene, size, count, path):
    w, h = size
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        "25",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for i in range(count):
            encoder.stdin.write(memoryview(frame(assets, scene, size, i, count)))
        encoder.stdin.close()
        if encoder.wait(timeout=120):
            raise RuntimeError("public benchmark clip encoding failed")


class CVReader:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            raise RuntimeError("OpenCV benchmark decoder failed")

    def read(self):
        ok, bgr = self.capture.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def close(self):
        self.capture.release()


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def decode_run(path, backend, count, pipeline=None, meter=None, policy="combined", scale=None):
    if pipeline:
        meter.clear()
        session = processor(pipeline, policy)
    tick, cpu, children = time.perf_counter(), time.process_time(), child_cpu()
    reader = (
        CVReader(path)
        if backend == "opencv"
        else FFmpegCapture(
            str(path), backend, "/dev/dri/renderD128", 5000, 20_000_000, output_size=scale
        )
    )
    reads, processing, trace, digest = [], [], [], hashlib.sha256()
    first_ms = None
    try:
        for i in range(count):
            before = time.perf_counter()
            rgb = reader.read()
            if rgb is None:
                raise RuntimeError(f"decoder stopped after {i} of {count} frames")
            reads.append((time.perf_counter() - before) * 1000)
            if first_ms is None:
                first_ms = (time.perf_counter() - tick) * 1000
            # Sparse pixel digest is negligible in the timed decode loop. Full
            # output/identity agreement is evaluated separately from throughput.
            digest.update(rgb[::64, ::64].tobytes())
            if pipeline and i % 5 == 0:
                before = time.perf_counter()
                result = session.analyze(rgb, BASE + timedelta(seconds=i / 25))
                processing.append((time.perf_counter() - before) * 1000)
                trace.append(trace_result(result))
        assert reader.read() is None, "unexpected extra frames"
    finally:
        reader.close()
    elapsed = time.perf_counter() - tick
    cpu_seconds = time.process_time() - cpu + child_cpu() - children
    return dict(
        backend=backend,
        scale=list(scale) if scale else None,
        policy=policy,
        decoded_frames=count,
        analyzed_frames=len(trace),
        elapsed_ms=elapsed * 1000,
        cpu_ms=cpu_seconds * 1000,
        decoded_fps=count / elapsed,
        first_frame_ms=first_ms,
        reads=summarize(reads[1:]),
        processing=summarize(processing) if processing else None,
        frame_ms=processing,
        trace=trace,
        rgb_sample_sha256=digest.hexdigest(),
        output_size=[rgb.shape[1], rgb.shape[0]],
        stages=stage_summary(meter) if meter else {},
    )


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--phase", choices=["motion", "decode", "integrated"], required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--revision", default="unspecified")
    parser.add_argument("--backends", default="opencv,ffmpeg,vaapi")
    parser.add_argument("--skip-scale", action="store_true")
    parser.add_argument("--reuse-clips", action="store_true")
    args = parser.parse_args()
    backends = args.backends.split(",")
    if not set(backends) <= {"opencv", "ffmpeg", "vaapi", "vaapi-i420"}:
        parser.error("unknown decoder backend")
    args.work.mkdir(parents=True, exist_ok=True)
    assets, hashes = load_assets()
    cfg = Settings(_env_file=None, models_dir=args.models, database=args.work / "identity.sqlite3")
    bundle = ModelBundle(cfg) if args.phase != "decode" else None
    store = None
    meter = Meter()
    pipeline = None
    results = []
    rng = random.Random(20261021)
    try:
        if bundle:
            for role in ("person", "face", "embedding"):
                setattr(bundle, role, meter.wrap(getattr(bundle, role), role))
            store = IdentityStore(cfg.database, bundle.fingerprint, bundle.dimension)
            pipeline = Pipeline(cfg, bundle, store)
            pipeline.enroll(assets["reference"], "reference", "Reference")
        if args.phase == "motion":
            for repeat in range(args.repeats):
                for size in ((1280, 720), (1920, 1080)):
                    for scene in SCENES:
                        policies = list(POLICIES)
                        rng.shuffle(policies)
                        for policy in policies:
                            row = analyze_frames(
                                pipeline, meter, assets, scene, size, args.frames, policy, 5
                            )
                            row["repeat"] = repeat
                            results.append(row)
                            print(
                                json.dumps(
                                    {
                                        k: row[k]
                                        for k in ("scene", "size", "policy", "repeat", "processing")
                                    }
                                ),
                                flush=True,
                            )
        else:
            clips = []
            if args.phase == "decode":
                for size in ((1920, 1080), (3840, 2160)):
                    path = args.work / f"testsrc-{size[0]}.mp4"
                    if args.reuse_clips and path.exists():
                        clips.append(("testsrc", size, path))
                        continue
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-y",
                            "-f",
                            "lavfi",
                            "-i",
                            f"testsrc2=size={size[0]}x{size[1]}:rate=25",
                            "-frames:v",
                            str(args.frames),
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-crf",
                            "23",
                            "-threads",
                            "1",
                            "-pix_fmt",
                            "yuv420p",
                            str(path),
                        ],
                        check=True,
                        timeout=180,
                    )
                    clips.append(("testsrc", size, path))
            else:
                for scene in ("empty", "entry_exit", "occupied"):
                    path = args.work / f"{scene}-1080.mp4"
                    if not (args.reuse_clips and path.exists()):
                        encode(assets, scene, (1920, 1080), args.frames, path)
                    clips.append((scene, (1920, 1080), path))
            for repeat in range(args.repeats):
                for scene, size, path in clips:
                    variants = [
                        (b, p, None)
                        for b in backends
                        for p in (("combined", "motion_idle") if pipeline else ("none",))
                    ]
                    if not pipeline and not args.skip_scale:
                        variants += [(b, "none", (1280, 720)) for b in ("ffmpeg", "vaapi")]
                    rng.shuffle(variants)
                    for backend, policy, scale in variants:
                        if pipeline:
                            for _ in range(5):
                                pipeline.analyze(frame(assets, scene, size, 0, args.frames))
                        try:
                            row = decode_run(
                                path, backend, args.frames, pipeline, meter, policy, scale
                            )
                        except Exception as exc:
                            row = dict(backend=backend, policy=policy, scale=scale, error=str(exc))
                        row.update(
                            scene=scene,
                            size=list(size),
                            repeat=repeat,
                            encoded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        )
                        results.append(row)
                        print(
                            json.dumps(
                                {k: v for k, v in row.items() if k not in ("trace", "frame_ms")}
                            ),
                            flush=True,
                        )
        report = dict(
            revision=args.revision,
            seed=20261021,
            phase=args.phase,
            assets_sha256=hashes,
            python=platform.python_version(),
            platform=platform.platform(),
            models=bundle.report if bundle else None,
            note="Public synthetic replay; excludes network. Timing includes RGB IPC "
            "for FFmpeg. No camera frames retained; GPU mode has no CPU decode fallback.",
            results=results,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if store:
            store.close()
        if bundle:
            for role in ("person", "face", "embedding"):
                getattr(bundle, role).close()


if __name__ == "__main__":
    main()
