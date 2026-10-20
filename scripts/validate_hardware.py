#!/usr/bin/env python3
"""Opt-in hardware smoke test using an image and an in-memory MJPEG stream.

Requires the dev extra (TestClient/httpx). No images or videos are persisted.
"""

import argparse
import io
import json
import statistics
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from human_detect.app import create_app
from human_detect.config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="one-person image with one visible face")
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--fps", type=float, default=5)
    args = parser.parse_args()
    if args.image:
        raw = args.image.read_bytes()
    else:
        url = (
            "https://raw.githubusercontent.com/google-coral/test_data/"
            "104342d2d3480b3e66203073dac24f4e2dbb4c41/grace_hopper.bmp"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
    jpeg = io.BytesIO()
    with Image.open(io.BytesIO(raw)) as image:
        image.convert("RGB").save(jpeg, format="JPEG")
    frame = jpeg.getvalue()
    stop = threading.Event()

    class Camera(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not stop.is_set():
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.flush()
                    stop.wait(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Camera)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = {}
    try:
        with tempfile.TemporaryDirectory(prefix="human-detect-smoke-") as directory:
            cfg = Settings(_env_file=None, database=Path(directory) / "identities.sqlite3")
            with TestClient(create_app(cfg)) as client:
                ready = client.get("/health/ready")
                assert ready.status_code == 200, ready.text
                response = client.post(
                    "/v1/identities/smoke-test/samples?name=SmokeTest", content=raw
                )
                assert response.status_code == 201, response.text
                results = []
                for _ in range(20):
                    response = client.post("/v1/images/analyze", content=raw)
                    assert response.status_code == 200, response.text
                    results.append(response.json())
                latest = results[-1]
                faces = [face for person in latest["persons"] for face in person["faces"]]
                assert faces, "test image must have a detectable body and face"
                assert any(f["identity"]["person_id"] == "smoke-test" for f in faces), latest
                timings = sorted(result["processing_ms"] for result in results)
                report["images"] = {
                    "runs": len(results),
                    "persons": len(latest["persons"]),
                    "faces": len(faces),
                    "processing_p50_ms": round(statistics.median(timings), 2),
                    "processing_p95_ms": timings[int(len(timings) * 0.95) - 1],
                    "same_image_enrollment_match": True,
                }
                source = f"http://127.0.0.1:{server.server_port}/camera"
                response = client.post("/v1/streams", json={"source": source, "fps": args.fps})
                assert response.status_code == 201, response.text
                stream_id = response.json()["stream_id"]
                started = time.monotonic()
                while time.monotonic() - started < args.seconds:
                    stop.wait(0.2)
                    snapshot = client.get(f"/v1/streams/{stream_id}").json()
                    assert snapshot["state"] != "failed", snapshot
                elapsed = time.monotonic() - started
                assert snapshot["processed"] > 0, snapshot
                video_faces = [f for p in snapshot["latest"]["persons"] for f in p["faces"]]
                assert any(f["identity"]["person_id"] == "smoke-test" for f in video_faces)
                report["video"] = {
                    "protocol": "HTTP MJPEG",
                    "target_fps": args.fps,
                    "elapsed_seconds": round(elapsed, 2),
                    "processed": snapshot["processed"],
                    "observed_fps_including_startup": round(snapshot["processed"] / elapsed, 2),
                    "skipped": snapshot["skipped"],
                    "state": snapshot["state"],
                }
                assert client.delete(f"/v1/streams/{stream_id}").status_code == 200
                assert client.delete("/v1/identities/smoke-test").status_code == 200
                unknown = client.post("/v1/images/analyze", content=raw).json()
                assert all(
                    f["identity"]["status"] == "unknown"
                    for p in unknown["persons"]
                    for f in p["faces"]
                )
                report["deleted_identity_returns_unknown"] = True
                report["ready_after_test"] = client.get("/health/ready").status_code == 200
                report["note"] = (
                    "Same-image matching verifies integration, not recognition accuracy."
                )
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
