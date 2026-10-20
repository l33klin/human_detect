"""Compare deterministic replay outputs against the same-frame baseline."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from human_detect.pipeline import iou
from human_detect.schemas import Box


def paired_frames(reference, candidate):
    counts = defaultdict(int)
    body_ious, face_ious = [], []
    for left, right in zip(reference, candidate):
        available = set(range(len(right["persons"])))
        counts["frames"] += 1
        counts["reference_persons"] += len(left["persons"])
        counts["candidate_persons"] += len(right["persons"])
        counts["same_person_count_frames"] += len(left["persons"]) == len(right["persons"])
        left_faces = [f for p in left["persons"] for f in p["faces"]]
        right_faces = [f for p in right["persons"] for f in p["faces"]]
        counts["reference_faces"] += len(left_faces)
        counts["candidate_faces"] += len(right_faces)
        left_ids = Counter(
            f["identity"]["person_id"] for f in left_faces if f["identity"]["person_id"] is not None
        )
        right_ids = Counter(
            f["identity"]["person_id"]
            for f in right_faces
            if f["identity"]["person_id"] is not None
        )
        counts["extra_positive_identities"] += sum((right_ids - left_ids).values())
        counts["missing_positive_identities"] += sum((left_ids - right_ids).values())
        for person in left["persons"]:
            if not available:
                continue
            scores = [
                (iou(Box(**person["box"]), Box(**right["persons"][index]["box"])), index)
                for index in available
            ]
            score, index = max(scores)
            if score < 0.5:
                continue
            available.remove(index)
            counts["matched_bodies_iou50"] += 1
            body_ious.append(score)
            other = right["persons"][index]
            free_faces = set(range(len(other["faces"])))
            for face in person["faces"]:
                counts["reference_faces_in_matched_bodies"] += 1
                if not free_faces:
                    continue
                face_score, face_index = max(
                    (iou(Box(**face["box"]), Box(**other["faces"][j]["box"])), j)
                    for j in free_faces
                )
                if face_score < 0.5:
                    continue
                free_faces.remove(face_index)
                counts["matched_faces_iou50"] += 1
                face_ious.append(face_score)
                left_id = face["identity"]["person_id"]
                right_id = other["faces"][face_index]["identity"]["person_id"]
                counts["same_identity_faces"] += left_id == right_id
                counts["cached_identity_disagreements"] += left_id != right_id
        temporal = right.get("temporal") or {}
        counts["tracked_frames"] += temporal.get("mode") == "tracked"
        counts["full_detection_frames"] += temporal.get("mode") == "detected"
    return dict(
        **counts,
        mean_body_iou=round(statistics.mean(body_ious), 5) if body_ious else None,
        mean_face_iou=round(statistics.mean(face_ious), 5) if face_ious else None,
    )


def summarize_report(report, reference="baseline"):
    groups = defaultdict(dict)
    for row in report["results"]:
        groups[(row["scenario"], tuple(row["size"]), row["repeat"])][row["variant"]] = row
    output = []
    for (scenario, size, repeat), variants in groups.items():
        baseline = variants[reference]
        for name, row in variants.items():
            assert row["corpus_sha256"] == baseline["corpus_sha256"], "different input frames"
            calls = {stage: value["calls"] for stage, value in row["stages"].items()}
            output.append(
                dict(
                    variant=name,
                    scenario=scenario,
                    size=list(size),
                    repeat=repeat,
                    processing=row["processing"],
                    processing_fps=row["processing_fps"],
                    calls=calls,
                    speedup=round(
                        baseline["processing"]["mean_ms"] / row["processing"]["mean_ms"], 3
                    ),
                    agreement=paired_frames(baseline["trace"], row["trace"]),
                )
            )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    result = summarize_report(report, args.reference)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in result:
        print(json.dumps({k: row[k] for k in ("variant", "scenario", "size", "repeat", "speedup")}))


if __name__ == "__main__":
    main()
