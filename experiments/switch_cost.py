"""Fixed-call microbenchmark to distinguish detector-pair from full-chain cache effects.

These are tensor invocations, not end-to-end face recognition. All stages get the same
source image, independently resized, so no detector threshold can change the call count.
"""

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.benchmark import load_assets, summarize
from human_detect.config import Settings
from human_detect.runtime import ModelBundle


def invoke(models, rgb, stages):
    return [getattr(models, stage).run(rgb) for stage in stages]


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets, hashes = load_assets()
    rgb = assets["reference"]
    variants, rows = {}, []
    with tempfile.TemporaryDirectory(prefix="hd-switch-") as temporary:
        for index, spec in enumerate(args.variant):
            label, directory = spec.split("=", 1)
            cfg = Settings(
                _env_file=None,
                models_dir=Path(directory),
                database=Path(temporary) / f"{index}.sqlite3",
            )
            variants[label] = ModelBundle(cfg)
        reference = invoke(variants["individual"], rgb, ("person", "face", "embedding"))
        equivalence = {}
        for label, models in variants.items():
            candidate = invoke(models, rgb, ("person", "face", "embedding"))
            differences = [
                float(np.max(np.abs(left - right)))
                for a, b in zip(reference, candidate)
                for left, right in zip(a, b)
            ]
            equivalence[label] = dict(
                max_abs_output_difference=max(differences),
                exact=all(value == 0 for value in differences),
            )
        rng = random.Random(20261020)
        for repeat in range(args.repeats):
            for stages in (("person", "face"), ("person", "face", "embedding")):
                labels = list(variants)
                rng.shuffle(labels)
                for label in labels:
                    models = variants[label]
                    for _ in range(10):
                        invoke(models, rgb, stages)
                    durations = []
                    for _ in range(args.runs):
                        start = time.perf_counter()
                        invoke(models, rgb, stages)
                        durations.append((time.perf_counter() - start) * 1000)
                    row = dict(
                        variant=label,
                        repeat=repeat,
                        stages=stages,
                        runs=args.runs,
                        processing=summarize(durations),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        args.output.write_text(
            json.dumps(
                dict(
                    assets_sha256=hashes,
                    equivalence_vs_individual=equivalence,
                    results=rows,
                    note="Fixed-call model invocation timing, not pipeline/camera FPS. "
                    "Embedding model is independently compiled and unchanged.",
                ),
                indent=2,
            )
            + "\n"
        )
        for models in variants.values():
            for role in ("person", "face", "embedding"):
                model = getattr(models, role)
                model.interpreter = None
                model.delegate = None


if __name__ == "__main__":
    main()
