"""Pool replay timings and compare the combination with temporal optimization alone."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.compare import paired_frames, summarize_report


def summarize(path):
    report = json.loads(path.read_text())
    groups = defaultdict(dict)
    for row in report["results"]:
        groups[(row["scenario"], tuple(row["size"]))].setdefault(row["variant"], []).append(row)
    output = []
    for (scenario, size), variants in groups.items():
        stats = {}
        for label, rows in variants.items():
            rows.sort(key=lambda row: row["repeat"])
            values = [value for row in rows for value in row["frame_ms"]]
            stats[label] = dict(
                mean_ms=statistics.mean(values),
                p95_ms=float(np.percentile(values, 95)),
                repeat_mean_ms=[statistics.mean(row["frame_ms"]) for row in rows],
                calls={
                    stage: sum(row["stages"].get(stage, {}).get("calls", 0) for row in rows)
                    for stage in ("person", "face", "embedding")
                },
            )
        temporal = stats["baseline+temporal"]
        combined = stats["joint+temporal"]
        pairs = zip(variants["baseline+temporal"], variants["joint+temporal"])
        agreements = []
        for left, right in pairs:
            assert left["corpus_sha256"] == right["corpus_sha256"]
            agreement = paired_frames(left["trace"], right["trace"])
            agreement["identical_trace_frames"] = sum(
                a == b for a, b in zip(left["trace"], right["trace"])
            )
            agreement["stage_calls_equal"] = {
                stage: value["calls"] for stage, value in left["stages"].items()
            } == {stage: value["calls"] for stage, value in right["stages"].items()}
            agreements.append(agreement)
        output.append(
            dict(
                scenario=scenario,
                size=list(size),
                variants=stats,
                combined_vs_temporal_reduction_percent=100
                * (1 - combined["mean_ms"] / temporal["mean_ms"]),
                combined_vs_baseline_speedup=stats["baseline"]["mean_ms"] / combined["mean_ms"],
                repeat_reduction_percent=[
                    100 * (1 - c / t)
                    for c, t in zip(combined["repeat_mean_ms"], temporal["repeat_mean_ms"])
                ],
                combined_vs_temporal_agreement=agreements,
            )
        )
    return dict(
        input_fps=report["input_fps"],
        revision=report["revision"],
        total_frames=sum(row["frames"] for row in report["results"]),
        peak_rss_mib=report["peak_rss_mib"],
        cases=output,
        baseline_comparison=summarize_report(report),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    for source in args.input:
        value = summarize(source)
        (destination / (source.stem + "-summary.json")).write_text(
            json.dumps(value, indent=2) + "\n"
        )
        with gzip.GzipFile(destination / (source.stem + "-raw.json.gz"), "wb", mtime=0) as target:
            target.write(source.read_bytes())
        for case in value["cases"]:
            print(
                source.stem,
                case["scenario"],
                case["size"],
                {
                    k: (round(v["mean_ms"], 2), round(v["p95_ms"], 2))
                    for k, v in case["variants"].items()
                },
                "increment",
                round(case["combined_vs_temporal_reduction_percent"], 2),
                "rounds",
                [round(v, 2) for v in case["repeat_reduction_percent"]],
            )


if __name__ == "__main__":
    main()
