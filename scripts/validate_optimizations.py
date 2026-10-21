"""Compare supported video profiles on real TPU; synthetic replay is not an accuracy test."""

import argparse
import hashlib
import io
import json
import platform
import random
import tempfile
import time
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from human_detect.config import Settings
from human_detect.identity import IdentityStore
from human_detect.motion import EmptySceneSession
from human_detect.pipeline import Pipeline
from human_detect.runtime import ModelBundle

DATA = (
    "https://raw.githubusercontent.com/google-coral/test_data/"
    "104342d2d3480b3e66203073dac24f4e2dbb4c41/"
)


class CountedModel:
    def __init__(self, model):
        self.model, self.calls = model, 0

    def detect(self, *args, **kwargs):
        self.calls += 1
        return self.model.detect(*args, **kwargs)

    def embed(self, *args):
        self.calls += 1
        return self.model.embed(*args)

    def close(self):
        self.model.close()


def load_assets():
    assets, hashes = {}, {}
    for key, name in (("reference", "grace_hopper.bmp"), ("alternate", "face.jpg")):
        with urllib.request.urlopen(DATA + name, timeout=30) as response:
            raw = response.read()
        hashes[key] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as image:
            assets[key] = np.asarray(image.convert("RGB")).copy()
    return assets, hashes


def frame(assets, scenario, index, count):
    image = cv2.resize(assets["reference"], (1920, 1080))
    if scenario == "empty" or (
        scenario == "entry_exit" and not count // 3 <= index < count * 2 // 3
    ):
        image[:] = 0
    elif scenario == "translation":
        offset = int(np.sin(index / count * np.pi * 2) * 1920 * 0.04)
        image = cv2.warpAffine(image, np.float32([[1, 0, offset], [0, 1, 0]]), (1920, 1080))
    elif scenario == "crowded":
        image[:, :960] = cv2.resize(assets["reference"], (960, 1080))
        image[:, 960:] = cv2.resize(assets["alternate"], (960, 1080))
    return image


def identities(result):
    return Counter(
        f["identity"]["person_id"]
        for p in result["persons"]
        for f in p["faces"]
        if f["identity"]["person_id"] is not None
    )


def replay(pipeline, assets, scenario, count, optimized):
    first = frame(assets, scenario, 0, count)
    for _ in range(5):
        pipeline.analyze(first)
    for role in ("person", "face", "embedding"):
        getattr(pipeline.models, role).calls = 0
    session = EmptySceneSession(pipeline) if optimized else pipeline
    start = datetime(2026, 10, 21, tzinfo=timezone.utc)
    rows, durations, checksum = [], [], hashlib.sha256()
    for index in range(count):
        rgb = frame(assets, scenario, index, count)
        checksum.update(rgb.tobytes())
        tick = time.perf_counter()
        result = session.analyze(rgb, start + timedelta(seconds=index / 5))
        durations.append((time.perf_counter() - tick) * 1000)
        rows.append(result.model_dump(mode="json"))
    return dict(
        mean_ms=round(float(np.mean(durations)), 3),
        p95_ms=round(float(np.percentile(durations, 95)), 3),
        calls={
            role: getattr(pipeline.models, role).calls for role in ("person", "face", "embedding")
        },
        corpus_sha256=checksum.hexdigest(),
        trace=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--joint-models", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 15 or args.repeats < 1:
        parser.error("frames >= 15 and repeats >= 1 required")
    cv2.setNumThreads(1)
    assets, hashes = load_assets()
    results, bundles, stores = [], [], []
    rng = random.Random(20261021)
    try:
        with tempfile.TemporaryDirectory(prefix="hd-final-replay-") as temporary:
            variants = []
            for name, directory, optimized in (
                ("baseline", args.models, False),
                ("optimized_individual", args.models, True),
                ("optimized_joint", args.joint_models, True),
            ):
                cfg = Settings(
                    _env_file=None, models_dir=directory, database=Path(temporary) / f"{name}.db"
                )
                bundle = ModelBundle(cfg)
                bundles.append(bundle)
                for role in ("person", "face", "embedding"):
                    setattr(bundle, role, CountedModel(getattr(bundle, role)))
                store = IdentityStore(cfg.database, bundle.fingerprint, bundle.dimension)
                stores.append(store)
                pipeline = Pipeline(cfg, bundle, store)
                pipeline.enroll(assets["reference"], "reference", "Reference")
                variants.append((name, pipeline, optimized))
            for repeat in range(args.repeats):
                for scenario in ("empty", "static", "entry_exit", "translation", "crowded"):
                    order = variants.copy()
                    rng.shuffle(order)
                    group = {}
                    for name, pipeline, optimized in order:
                        row = replay(pipeline, assets, scenario, args.frames, optimized)
                        row.update(profile=name, scenario=scenario, repeat=repeat)
                        group[name] = row
                        results.append(row)
                        print(
                            json.dumps(
                                {
                                    k: row[k]
                                    for k in ("profile", "scenario", "repeat", "mean_ms", "calls")
                                }
                            ),
                            flush=True,
                        )
                    baseline = group["baseline"]
                    for row in group.values():
                        assert row["corpus_sha256"] == baseline["corpus_sha256"]
                        differences = Counter()
                        for left, right in zip(baseline["trace"], row["trace"]):
                            differences["extra_positive_identities"] += sum(
                                (identities(right) - identities(left)).values()
                            )
                            differences["missing_positive_identities"] += sum(
                                (identities(left) - identities(right)).values()
                            )
                            differences["presence_disagreements"] += bool(left["persons"]) != bool(
                                right["persons"]
                            )
                        row["agreement"] = dict(differences)
            report = dict(
                python=platform.python_version(),
                platform=platform.platform(),
                assets_sha256=hashes,
                frames=args.frames,
                repeats=args.repeats,
                input_fps=5,
                image_size=[1920, 1080],
                results=results,
                note="Synthetic replay, pipeline only; excludes decode, network and pacing. "
                "Agreement is relative to baseline, not ground-truth recognition accuracy.",
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        for store in stores:
            store.close()
        for bundle in bundles:
            bundle.close()


if __name__ == "__main__":
    main()
