"""Deterministic synthetic-video benchmark; never reports replay as real camera accuracy."""

import argparse
import hashlib
import io
import json
import platform
import random
import resource
import statistics
import tempfile
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from human_detect.config import Settings
from human_detect.identity import IdentityStore
from human_detect.pipeline import Pipeline
from human_detect.runtime import ModelBundle

DATA = "https://raw.githubusercontent.com/google-coral/test_data/104342d2d3480b3e66203073dac24f4e2dbb4c41/"
SCENARIOS = ("static", "translation", "occlusion", "switch", "two_people")


def load_assets():
    assets, hashes = {}, {}
    for key, filename in (("reference", "grace_hopper.bmp"), ("alternate", "face.jpg")):
        with urllib.request.urlopen(DATA + filename, timeout=30) as response:
            data = response.read()
        hashes[key] = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as image:
            assets[key] = np.asarray(image.convert("RGB")).copy()
    return assets, hashes


def render_frame(assets, scenario, width, height, index, count):
    image = cv2.resize(assets["reference"], (width, height))
    phase = index / max(1, count)
    if scenario == "translation":
        dx = int(np.sin(phase * np.pi * 2) * width * 0.04)
        dy = int(np.sin(phase * np.pi * 4) * height * 0.015)
        image = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height))
    elif scenario == "occlusion" and 0.35 <= phase < 0.65:
        image[int(height * 0.05) : int(height * 0.60), int(width * 0.20) : int(width * 0.80)] = 0
    elif scenario == "switch":
        if 0.3 <= phase < 0.6:
            image = cv2.resize(assets["alternate"], (width, height))
        elif 0.6 <= phase < 0.75:
            image[:] = 0
    elif scenario == "two_people":
        image[:, : width // 2] = cv2.resize(assets["reference"], (width // 2, height))
        image[:, width // 2 :] = cv2.resize(assets["alternate"], (width - width // 2, height))
    return image


class Meter:
    def __init__(self):
        self.calls = defaultdict(list)

    def clear(self):
        self.calls.clear()

    def wrap(self, model, stage):
        meter = self

        class Measured:
            def close(self):
                model.interpreter = None
                model.delegate = None

            def __getattr__(self, name):
                function = getattr(model, name)
                if name not in {"detect", "embed"}:
                    return function

                def measured(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        return function(*args, **kwargs)
                    finally:
                        meter.calls[stage].append((time.perf_counter() - start) * 1000)

                return measured

        return Measured()


def summarize(values):
    ordered = sorted(values)
    return dict(
        mean_ms=round(statistics.mean(ordered), 3),
        p50_ms=round(statistics.median(ordered), 3),
        p95_ms=round(float(np.percentile(ordered, 95)), 3),
    )


def trace_result(result):
    value = result.model_dump(mode="json")
    return {"persons": value["persons"], "temporal": value.get("temporal")}


def run_unit(pipeline, meter, assets, scenario, size, frames, fps, mode):
    first = render_frame(assets, scenario, *size, 0, frames)
    for _ in range(5):
        pipeline.analyze(first)
    if mode == "temporal":
        from human_detect.temporal import TemporalSession

        processor = TemporalSession(pipeline)
    else:
        processor = pipeline
    meter.clear()
    durations, trace, corpus_hash = [], [], hashlib.sha256()
    started = datetime(2026, 10, 20, tzinfo=timezone.utc)
    for index in range(frames):
        rgb = render_frame(assets, scenario, *size, index, frames)
        corpus_hash.update(rgb.tobytes())
        tick = time.perf_counter()
        result = processor.analyze(rgb, started + timedelta(seconds=index / fps))
        durations.append((time.perf_counter() - tick) * 1000)
        trace.append(trace_result(result))
    return dict(
        scenario=scenario,
        size=list(size),
        mode=mode,
        frames=frames,
        corpus_sha256=corpus_hash.hexdigest(),
        processing=summarize(durations),
        processing_fps=round(1000 / statistics.mean(durations), 3),
        stages={
            name: dict(calls=len(values), total_ms=round(sum(values), 3), **summarize(values))
            for name, values in meter.calls.items()
        },
        frame_ms=[round(v, 3) for v in durations],
        trace=trace,
    )


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", help="label=models_dir; repeat per variant")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--input-fps", type=float, default=25)
    parser.add_argument("--sizes", default="512x600,1280x720,1920x1080")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="unspecified")
    args = parser.parse_args()
    if args.frames < 10 or args.repeats < 1 or args.input_fps <= 0:
        parser.error("frames >= 10, repeats >= 1 and input-fps > 0 are required")
    sizes = [tuple(map(int, size.split("x"))) for size in args.sizes.split(",")]
    scenarios = args.scenarios.split(",")
    if any(s not in SCENARIOS for s in scenarios):
        parser.error("unknown scenario")
    assets, hashes = load_assets()
    rng = random.Random(20261020)
    results, variants, resources = [], [], []
    try:
        with tempfile.TemporaryDirectory(prefix="hd-benchmark-") as temporary:
            for index, spec in enumerate(args.variant or ["baseline=models"]):
                label, directory = spec.split("=", 1)
                cfg = Settings(
                    _env_file=None,
                    models_dir=Path(directory),
                    database=Path(temporary) / f"{index}.sqlite3",
                )
                bundle, meter = ModelBundle(cfg), Meter()
                for role in ("person", "face", "embedding"):
                    setattr(bundle, role, meter.wrap(getattr(bundle, role), role))
                store = IdentityStore(cfg.database, bundle.fingerprint, bundle.dimension)
                resources.append(store)
                pipeline = Pipeline(cfg, bundle, store)
                pipeline.enroll(assets["reference"], "reference", "Reference")
                variants.append((label, "baseline", pipeline, meter))
                if args.temporal:
                    variants.append((label + "+temporal", "temporal", pipeline, meter))
            for repeat in range(args.repeats):
                for size in sizes:
                    for scenario in scenarios:
                        order = list(variants)
                        rng.shuffle(order)
                        for label, mode, pipeline, meter in order:
                            result = run_unit(
                                pipeline,
                                meter,
                                assets,
                                scenario,
                                size,
                                args.frames,
                                args.input_fps,
                                mode,
                            )
                            result.update(variant=label, repeat=repeat)
                            results.append(result)
                            print(
                                json.dumps(
                                    {
                                        k: result[k]
                                        for k in (
                                            "variant",
                                            "repeat",
                                            "scenario",
                                            "size",
                                            "processing",
                                        )
                                    }
                                ),
                                flush=True,
                            )
            report = dict(
                revision=args.revision,
                seed=20261020,
                python=platform.python_version(),
                platform=platform.platform(),
                input_fps=args.input_fps,
                assets_sha256=hashes,
                models={
                    label: pipeline.models.report
                    for label, mode, pipeline, _ in variants
                    if mode == "baseline"
                },
                peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                note="Synthetic replay; processing only, excludes decode/network/pacing. "
                "Temporal frames use source timestamps, not wall-clock time. "
                "Detection agreement with baseline is not ground-truth accuracy.",
                results=results,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    finally:
        for store in resources:
            store.close()
        for _, mode, pipeline, _ in variants:
            if mode == "baseline":
                for role in ("person", "face", "embedding"):
                    getattr(pipeline.models, role).close()


if __name__ == "__main__":
    main()
