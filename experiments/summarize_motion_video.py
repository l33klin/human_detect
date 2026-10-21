"""Summarize reproducible motion/decoder measurements and same-frame agreement."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.compare import paired_frames


def pooled(rows):
    values = [v for r in rows for v in r.get("frame_ms", [])]
    output = dict(
        runs=len(rows),
        stages={
            stage: sum(r.get("stages", {}).get(stage, {}).get("calls", 0) for r in rows)
            for stage in ("person", "face", "embedding")
        },
    )
    if values:
        output.update(
            mean_ms=statistics.mean(values),
            p95_ms=float(np.percentile(values, 95)),
            repeat_mean_ms=[statistics.mean(r["frame_ms"]) for r in rows],
        )
    if "elapsed_ms" in rows[0]:
        output.update(
            mean_elapsed_ms=statistics.mean(r["elapsed_ms"] for r in rows),
            mean_cpu_ms=statistics.mean(r["cpu_ms"] for r in rows),
            mean_first_frame_ms=statistics.mean(r["first_frame_ms"] for r in rows),
            total_decoded_frames=sum(r["decoded_frames"] for r in rows),
            decoded_fps=sum(r["decoded_frames"] for r in rows)
            / (sum(r["elapsed_ms"] for r in rows) / 1000),
            cpu_ms_per_decoded_frame=sum(r["cpu_ms"] for r in rows)
            / sum(r["decoded_frames"] for r in rows),
        )
    return output


def agreement(left, right, fps=5):
    assert len(left) == len(right)
    result = paired_frames(left, right)
    result["extra_person_frames"] = sum(
        bool(candidate["persons"]) and not reference["persons"]
        for reference, candidate in zip(left, right)
    )
    result["missing_person_frames"] = sum(
        bool(reference["persons"]) and not candidate["persons"]
        for reference, candidate in zip(left, right)
    )
    left_on = [bool(r["persons"]) for r in left]
    right_on = [bool(r["persons"]) for r in right]
    delays = []
    missed = 0
    for i, present in enumerate(left_on):
        if present and (i == 0 or not left_on[i - 1]):
            end = next((j for j in range(i, len(left_on)) if not left_on[j]), len(left_on))
            found = next((j for j in range(i, end) if right_on[j]), None)
            if found is None:
                missed += 1
            else:
                delays.append((found - i) * 1000 / fps)
    result.update(entry_delay_vs_baseline_ms=delays, missed_baseline_segments=missed)
    return result


def summarize(report):
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    errors = []
    for r in report["results"]:
        size = tuple(r["size"])
        backend = r.get("backend", "rgb")
        scale = tuple(r.get("scale") or [])
        label = (backend, r["policy"], scale)
        if "error" in r:
            errors.append(r)
            continue
        groups[(r["scene"], size, label)].append(r)
        if r.get("trace"):
            pairs[(r["scene"], size, r["repeat"])][label] = r
    output = []
    for (scene, size, (backend, policy, scale)), rows in groups.items():
        rows.sort(key=lambda r: r["repeat"])
        output.append(
            dict(
                scene=scene,
                size=list(size),
                backend=backend,
                policy=policy,
                scale=list(scale),
                **pooled(rows),
            )
        )
    comparisons = []
    for (scene, size, repeat), variants in pairs.items():
        for (backend, policy, scale), r in variants.items():
            reference = variants[(backend, "combined", scale)]
            comparisons.append(
                dict(
                    scene=scene,
                    size=list(size),
                    repeat=repeat,
                    backend=backend,
                    policy=policy,
                    reference="same_backend_combined",
                    agreement=agreement(reference["trace"], r["trace"]),
                )
            )
            if backend.startswith("vaapi") and ("ffmpeg", policy, scale) in variants:
                reference = variants[("ffmpeg", policy, scale)]
                comparisons.append(
                    dict(
                        scene=scene,
                        size=list(size),
                        repeat=repeat,
                        backend=backend,
                        policy=policy,
                        reference="ffmpeg_same_policy",
                        rgb_sample_equal=reference["rgb_sample_sha256"] == r["rgb_sample_sha256"],
                        agreement=agreement(reference["trace"], r["trace"]),
                    )
                )
    return dict(
        phase=report["phase"],
        revision=report["revision"],
        revisions=report.get("revisions", [report["revision"]]),
        groups=output,
        errors=errors,
        comparisons=comparisons,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    revisions = [report["revision"]]
    for path in args.extra:
        extra = json.loads(path.read_text())
        assert report["phase"] == extra["phase"]
        assert report["assets_sha256"] == extra["assets_sha256"]
        assert report["models"] == extra["models"]
        report["results"].extend(extra["results"])
        revisions.append(extra["revision"])
    report["revisions"] = revisions
    result = summarize(report)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["groups"]:
        print(json.dumps(row))
    print("ERRORS", len(result["errors"]))


if __name__ == "__main__":
    main()
