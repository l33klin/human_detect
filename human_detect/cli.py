import argparse
import ctypes.util
import json
import platform
from pathlib import Path

from human_detect.config import Settings
from human_detect.runtime import ModelBundle, audit_model


def doctor(invoke=False):
    settings = Settings()
    report = {
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "libedgetpu": ctypes.util.find_library("edgetpu"),
        "pci_devices": [str(p) for p in Path("/dev").glob("apex_*")],
        "models": {},
        "inference_tested": False,
    }
    failed = False
    try:
        manifest = json.loads((settings.models_dir / "manifest.json").read_text())
        for role, item in manifest.items():
            audit = audit_model(settings.models_dir / item["file"])
            report["models"][role] = audit
            if audit["sha256"] != item["sha256"]:
                raise RuntimeError(f"{role} checksum mismatch")
        if invoke:
            ModelBundle(settings)
            report["inference_tested"] = True
    except Exception as exc:
        report["error"] = str(exc)
        failed = True
    if not report["libedgetpu"] or not report["pci_devices"]:
        failed = True
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(prog="human-detect")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    check = commands.add_parser("doctor")
    check.add_argument("--invoke", action="store_true", help="invoke all three models on TPU")
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor(args.invoke))
    import uvicorn

    # One process owns the TPU; camera capture children never load interpreters.
    uvicorn.run("human_detect.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
