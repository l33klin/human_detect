"""Real loopback HTTP video -> capture process -> PCIe TPU -> JSON smoke test."""

import argparse
import io
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from human_detect.app import create_app
from human_detect.config import Settings
from human_detect.runtime import ModelBundle
from human_detect.video import FFmpegCapture


def run(models, clip, work, backend, kill_parent=False):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}/live"
    server = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-re",
            "-stream_loop",
            "-1",
            "-i",
            str(clip),
            "-c",
            "copy",
            "-f",
            "mpegts",
            "-listen",
            "1",
            url,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cfg = Settings(
        _env_file=None,
        models_dir=models,
        database=work / f"smoke-{backend}-{kill_parent}.sqlite3",
        temporal_enabled=True,
        empty_scene_enabled=True,
        capture_backend=backend,
        api_key="local-test-token",
    )
    resources = []

    def factory(settings):
        bundle = ModelBundle(settings)
        resources.append(bundle)
        return bundle

    app = create_app(cfg, factory)
    try:
        with TestClient(app, headers={"Authorization": "Bearer local-test-token"}) as client:
            reference_url = (
                "https://raw.githubusercontent.com/google-coral/test_data/"
                "104342d2d3480b3e66203073dac24f4e2dbb4c41/grace_hopper.bmp"
            )
            with urllib.request.urlopen(reference_url, timeout=30) as response:
                raw = response.read()
            body = io.BytesIO()
            with Image.open(io.BytesIO(raw)) as image:
                image.convert("RGB").save(body, format="PNG")
            assert (
                client.post(
                    "/v1/identities/reference/samples?name=Reference", content=body.getvalue()
                ).status_code
                == 201
            )
            assert client.get("/health/ready").json()["cpu_fallback"] is False
            response = client.post("/v1/streams", json=dict(source=url, fps=5))
            assert response.status_code == 201
            stream_id = response.json()["stream_id"]
            stream = app.state.streams.get(stream_id)
            deadline = time.monotonic() + 30
            snapshot = stream.snapshot()
            while snapshot["processed"] < 6 and time.monotonic() < deadline:
                time.sleep(0.1)
                snapshot = client.get(f"/v1/streams/{stream_id}").json()
            assert snapshot["processed"] >= 6, snapshot
            assert snapshot["state"] == "running", snapshot
            assert snapshot["capture_backend"] == backend
            latest = snapshot["latest"]
            assert latest["width"] == 1920 and latest["height"] == 1080
            assert latest["persons"]
            faces = [f for p in latest["persons"] for f in p["faces"]]
            assert any(f["identity"]["person_id"] == "reference" for f in faces)
            children_path = Path(f"/proc/{stream.process.pid}/task/{stream.process.pid}/children")
            children = children_path.read_text().split()
            assert children, "expected an FFmpeg capture child"
            assert client.delete("/v1/identities/reference").status_code == 200
            before = snapshot["sequence"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                snapshot = client.get(f"/v1/streams/{stream_id}").json()
                if snapshot["sequence"] > before:
                    break
                time.sleep(0.05)
            assert snapshot["sequence"] > before
            assert all(
                face["identity"]["person_id"] is None
                for person in snapshot["latest"]["persons"]
                for face in person["faces"]
            ), snapshot
            if kill_parent:
                stream.process.kill()
                stream.process.join(timeout=2)
            assert client.delete(f"/v1/streams/{stream_id}").status_code == 200
            assert not stream.process.is_alive()
            assert not stream.thread.is_alive()
            for child in children:
                deadline = time.monotonic() + 5
                stat = Path(f"/proc/{child}/stat")
                while stat.exists() and time.monotonic() < deadline:
                    if stat.read_text().split()[2] == "Z":
                        break
                    time.sleep(0.05)
                assert not stat.exists() or stat.read_text().split()[2] == "Z"
            return dict(
                backend=backend,
                processed=snapshot["processed"],
                size=[latest["width"], latest["height"]],
                known_identity=True,
                motion_metadata=latest["motion"],
                kill_parent=kill_parent,
                capture_and_decoder_released=True,
                cpu_dnn_fallback=False,
                live_identity_deletion_invalidates_cache=True,
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=3)
        for bundle in resources:
            bundle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    results = [
        run(args.models, args.clip, args.work, "ffmpeg"),
        run(args.models, args.clip, args.work, "vaapi"),
        run(args.models, args.clip, args.work, "vaapi", kill_parent=True),
    ]
    failed = FFmpegCapture(
        str(args.clip), "vaapi", "/dev/dri/nonexistent-hd-test", 5000, 20_000_000, 5
    )
    try:
        assert failed.read() is None
        assert failed.process.wait(timeout=5) != 0
    finally:
        failed.close()
    args.output.write_text(
        json.dumps(dict(results=results, invalid_device_no_fallback=True), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
