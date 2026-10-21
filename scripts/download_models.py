#!/usr/bin/env python3
"""Fetch pinned model artifacts; no runtime downloads or unverified overwrites."""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    directory = parser.parse_args().models
    directory.mkdir(exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    for item in manifest.values():
        target = directory / item["file"]
        expected = item["sha256"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise SystemExit(
                    f"Checksum mismatch: {target}; remove it explicitly to re-download"
                )
            print(f"OK {target.name}")
            continue
        temporary = target.with_suffix(".part")
        try:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                with temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch: {target.name}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded {target.name}")


if __name__ == "__main__":
    main()
