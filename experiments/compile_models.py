"""Controlled compilation: same two quantized source models, separate versus shared cache.

The existing compiled embedder is copied unchanged and explicitly excluded from co-compilation.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from human_detect.runtime import audit_model

BASE = "https://raw.githubusercontent.com/google-coral/test_data/104342d2d3480b3e66203073dac24f4e2dbb4c41/"
SOURCES = {
    "person": "ssdlite_mobiledet_coco_qat_postprocess.tflite",
    "face": "ssd_mobilenet_v2_face_quant_postprocess.tflite",
}
SOURCE_SHA256 = {
    "person": "32c486140391eb4dc43fca7113ad392be632dc5366687f2731f73d740678693f",
    "face": "4c33a81508aed53a2b55b91693957d777f051e24cd62f69270f512957d422aa2",
}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", type=Path, help="system compiler executable")
    parser.add_argument("--compiler-bundle", type=Path, help="portable edgetpu_compiler_bin folder")
    parser.add_argument("--original-models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("experiments/artifacts"))
    args = parser.parse_args()
    if args.compiler_bundle:
        folder = args.compiler_bundle.resolve()
        command = [
            str(folder / "ld-linux-x86-64.so.2"),
            "--library-path",
            str(folder),
            str(folder / "edgetpu_compiler"),
        ]
    elif args.compiler:
        command = [str(args.compiler.resolve())]
    else:
        parser.error("provide --compiler or --compiler-bundle")
    version = subprocess.check_output(command + ["--version"], text=True).strip()
    source_dir = args.output / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_report = {}
    for role, name in SOURCES.items():
        path = source_dir / name
        if not path.exists():
            with urllib.request.urlopen(BASE + name, timeout=60) as response:
                path.write_bytes(response.read())
        if checksum(path) != SOURCE_SHA256[role]:
            raise RuntimeError(f"source model checksum mismatch: {role}")
        source_report[role] = dict(file=name, url=BASE + name, sha256=checksum(path))
    original = json.loads((args.original_models / "manifest.json").read_text())
    embedding = args.original_models / original["embedding"]["file"]
    if checksum(embedding) != original["embedding"]["sha256"]:
        raise RuntimeError("original embedder checksum mismatch")
    report = dict(
        compiler_version=version,
        source_models=source_report,
        limitation="Only person/face detectors co-compiled; original embedder unchanged.",
        original_embedding_sha256=checksum(embedding),
        variants={},
    )
    for label, order in (
        ("individual", ["person", "face"]),
        ("co-person-first", ["person", "face"]),
        ("co-face-first", ["face", "person"]),
    ):
        output = args.output / label
        output.mkdir(parents=True, exist_ok=True)
        groups = [[role] for role in order] if label == "individual" else [order]
        commands = []
        for index, roles in enumerate(groups):
            argv = command + ["--out_dir", str(output), "--show_operations"]
            argv += [str(source_dir / SOURCES[role]) for role in roles]
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=180)
            log = completed.stdout + completed.stderr
            (output / f"compile-{index}.txt").write_text(log)
            print(log, flush=True)
            completed.check_returncode()
            commands.append(argv)
        manifest = {}
        for role in ("person", "face"):
            filename = Path(SOURCES[role]).stem + "_edgetpu.tflite"
            path = output / filename
            audit_model(path)
            manifest[role] = dict(
                file=filename, sha256=checksum(path), url=source_report[role]["url"], generated=True
            )
        shutil.copyfile(embedding, output / embedding.name)
        manifest["embedding"] = original["embedding"]
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report["variants"][label] = dict(commands=commands, models=manifest)
    (args.output / "compilation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
