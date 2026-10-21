#!/usr/bin/env python3
"""Exercise real TPU + timestamped RTSP output using public or supplied test media.

Run in the service's Linux environment with a private MediaMTX binary and an
enrollable single-face sample. No production cameras or existing databases are used.
"""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from human_detect.app import create_app
from human_detect.config import Settings


def wait_for(client, ids, predicate, seconds=35):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        values = [client.get(f"/v1/streams/{key}").json() for key in ids]
        if all(predicate(value) for value in values):
            return values
        time.sleep(0.2)
    raise RuntimeError(json.dumps(values))


def run(args):
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    children, logs = [], []
    config = work / "mediamtx.yml"
    config.write_text(
        "logLevel: warn\nrtspAddress: 127.0.0.1:18554\nrtspTransports: [tcp]\n"
        "webrtcAddress: 127.0.0.1:18889\nwebrtcLocalUDPAddress: 127.0.0.1:18189\n"
        "webrtcAdditionalHosts: [127.0.0.1]\nrtmp: false\nhls: false\nsrt: false\n"
        "moq: false\npaths:\n  all_others: {}\n"
    )

    def launch(command, name):
        log = (work / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )
        children.append(process)
        return process

    report = {
        "streams": args.streams,
        "external_clip": bool(args.clip),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.clip:
        digest = hashlib.sha256()
        with args.clip.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        report["source_clip_sha256"] = digest.hexdigest()
    try:
        server = launch([str(args.mediamtx), str(config)], "mediamtx")
        time.sleep(0.5)
        assert server.poll() is None, "private MediaMTX failed to start; check unused test ports"
        # Encode once before timing. Re-encoding two synthetic cameras in real
        # time would compete for CPU with the service we want to measure.
        clip = args.clip or work / "reference.mp4"
        if not args.clip:
            prepared = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-filter_threads",
                    "1",
                    "-loop",
                    "1",
                    "-framerate",
                    "25",
                    "-i",
                    str(args.sample),
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000",
                    "-t",
                    "5",
                    "-vf",
                    "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-threads:v",
                    "1",
                    "-g",
                    "25",
                    "-c:a",
                    "aac",
                    "-y",
                    str(clip),
                ],
                capture_output=True,
                timeout=60,
            )
            assert prepared.returncode == 0, prepared.stderr.decode()
        source_info = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(clip)]
            )
        )
        source_video = next(s for s in source_info["streams"] if s["codec_type"] == "video")
        width, height = source_video["width"], source_video["height"]
        has_audio = any(s["codec_type"] == "audio" for s in source_info["streams"])
        report["input"] = dict(
            width=width, height=height, fps=source_video["avg_frame_rate"], audio=has_audio
        )
        for index in range(args.streams):
            launch(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-re",
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(clip),
                    "-c",
                    "copy",
                    "-f",
                    "rtsp",
                    "-rtsp_transport",
                    "tcp",
                    f"rtsp://127.0.0.1:18554/input{index}",
                ],
                f"source-{index}",
            )
        time.sleep(1)
        cfg = Settings(
            _env_file=None,
            database=work / "identities.sqlite3",
            models_dir=args.models,
            media_enabled=True,
            capture_backend=args.backend,
            media_encoder=args.encoder,
            media_publish_url="rtsp://127.0.0.1:18554",
            media_rtsp_port=18554,
            media_public_host="127.0.0.1",
            media_webrtc_url="http://127.0.0.1:18889",
            media_font=args.font,
        )
        app = create_app(cfg)
        with TestClient(app) as client:
            assert client.get("/health/ready").status_code == 200
            enrolled = client.post(
                "/v1/identities/reference/samples?name=参考人物", content=args.sample.read_bytes()
            )
            assert enrolled.status_code == 201, enrolled.text
            ids = []
            for index in range(args.streams):
                response = client.post(
                    "/v1/streams",
                    json=dict(
                        source=f"rtsp://127.0.0.1:18554/input{index}",
                        name=f"验证视频 {index + 1}",
                        publish=True,
                    ),
                )
                assert response.status_code == 201, response.text
                ids.append(response.json()["stream_id"])
            first = wait_for(client, ids, lambda value: value["processed"] >= 50)
            # Drain startup probing buffers before measuring a real-time rate.
            time.sleep(5)
            first = [client.get(f"/v1/streams/{key}").json() for key in ids]
            started = time.monotonic()
            observations = []
            while time.monotonic() - started < args.seconds:
                for key in ids:
                    current = client.get(f"/v1/streams/{key}").json()
                    result = current.get("latest") or {}
                    faces = [f for p in result.get("persons", []) for f in p["faces"]]
                    observations.append(
                        dict(
                            stream_id=key,
                            frame=result.get("frame"),
                            state=current["state"],
                            persons=len(result.get("persons", [])),
                            faces=len(faces),
                            matched=sum(f["identity"]["status"] == "matched" for f in faces),
                            mode=result.get("temporal", {}).get("mode"),
                            processing_ms=result.get("processing_ms"),
                        )
                    )
                time.sleep(0.2)
            last = [client.get(f"/v1/streams/{key}").json() for key in ids]
            elapsed = time.monotonic() - started
            report["measurement_seconds"] = elapsed
            report["processed_fps"] = [
                (b["processed"] - a["processed"]) / elapsed for a, b in zip(first, last)
            ]
            report["snapshots"] = last
            report["observations"] = observations
            assert args.clip or all(
                any(
                    f["identity"]["status"] == "matched"
                    for p in value["latest"]["persons"]
                    for f in p["faces"]
                )
                for value in last
            ), "reference identity did not match in both streams"
            report["output_fps"] = [
                (b["media"]["output_frames"] - a["media"]["output_frames"]) / elapsed
                for a, b in zip(first, last)
            ]
            if args.benchmark_only:
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(
                    json.dumps(
                        {k: v for k, v in report.items() if k != "observations"},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    flush=True,
                )
                return
            report["codecs"] = []
            report["packet_timestamps"] = []
            for value in last:
                command = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-i",
                    value["output"]["rtsp_url"],
                    "-show_streams",
                    "-show_packets",
                    "-read_intervals",
                    "%+3",
                    "-of",
                    "json",
                ]
                probe = subprocess.run(command, capture_output=True, text=True, timeout=18)
                assert probe.returncode == 0, probe.stderr
                data = json.loads(probe.stdout)
                codecs = [
                    {
                        k: stream.get(k)
                        for k in (
                            "codec_name",
                            "codec_type",
                            "profile",
                            "width",
                            "height",
                            "r_frame_rate",
                            "sample_rate",
                            "channels",
                            "has_b_frames",
                        )
                    }
                    for stream in data["streams"]
                ]
                assert {s["codec_name"] for s in codecs} == (
                    {"h264", "opus"} if has_audio else {"h264"}
                )
                video = next(s for s in codecs if s["codec_type"] == "video")
                assert (video["width"], video["height"]) == (width, height)
                assert video["has_b_frames"] == 0
                report["codecs"].append(codecs)
                times = {}
                for packet in data["packets"]:
                    if "pts_time" in packet:
                        times.setdefault(packet["stream_index"], []).append(
                            float(packet["pts_time"])
                        )
                assert all(all(b > a for a, b in zip(t, t[1:])) for t in times.values())
                if has_audio:
                    audio_index = next(
                        s["index"] for s in data["streams"] if s["codec_type"] == "audio"
                    )
                    audio_times = times[audio_index]
                    assert len(audio_times) >= 125, "audio was lost while dropping video frames"
                    assert max(b - a for a, b in zip(audio_times, audio_times[1:])) < 0.06
                report["packet_timestamps"].append(
                    {str(k): dict(first=v[0], last=v[-1], count=len(v)) for k, v in times.items()}
                )
                # Decode a small sample to prove labels/boxes are actually in the RTSP pixels.
                pixels = subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-rtsp_transport",
                        "tcp",
                        "-i",
                        value["output"]["rtsp_url"],
                        "-frames:v",
                        "1",
                        "-an",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "pipe:1",
                    ],
                    capture_output=True,
                    timeout=15,
                )
                assert pixels.returncode == 0 and len(pixels.stdout) == width * height * 3
                import numpy as np

                rgb = np.frombuffer(pixels.stdout, np.uint8).reshape(height, width, 3)
                if args.preview:
                    from PIL import Image

                    Image.fromarray(rgb).save(args.preview)
                green = (rgb[:, :, 1].astype(int) - rgb[:, :, 0] > 60) & (rgb[:, :, 1] > 150)
                assert int(green.sum()) > 100
            # A single authoritative identity store invalidates cached matches immediately.
            assert client.delete("/v1/identities/reference").status_code == 200

            def no_match(value):
                return not any(
                    f["identity"]["status"] == "matched"
                    for p in (value.get("latest") or {}).get("persons", [])
                    for f in p["faces"]
                )

            wait_for(client, ids, no_match)
            report["identity_delete"] = True
            # Stop one stream explicitly; only the other should auto-resume.
            assert client.post(f"/v1/streams/{ids[0]}/stop").json()["state"] == "stopped"
        with TestClient(create_app(cfg)) as client:
            assert client.get(f"/v1/streams/{ids[0]}").json()["state"] == "stopped"
            if len(ids) > 1:
                wait_for(client, [ids[1]], lambda value: value["processed"] >= 10)
            assert client.post(f"/v1/streams/{ids[0]}/start").status_code == 200
            before = wait_for(client, ids, lambda value: value["processed"] >= 10)
            # Kill only our capture worker. The process-group watchdog must recover.
            stream = client.app.state.streams.get(ids[0])
            stream.process.kill()
            old_epoch = before[0]["latest"]["frame"]["source_epoch"]
            wait_for(
                client,
                [ids[0]],
                lambda v: (
                    v["state"] == "running" and v["latest"]["frame"]["source_epoch"] != old_epoch
                ),
                seconds=30,
            )
            report["worker_restart"] = True
            for key in ids:
                assert client.delete(f"/v1/streams/{key}").status_code == 200
            report["stop_start_and_service_restart"] = True
        report["cpu_nn_fallback"] = False
        report["uid"] = __import__("os").getuid()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in {"snapshots", "codecs", "observations"}
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    finally:
        for process in children:
            process.terminate()
        for process in children:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediamtx", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--streams", type=int, choices=[1, 2], default=1)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--backend", choices=["ffmpeg", "vaapi"], default="vaapi")
    parser.add_argument("--encoder", choices=["libx264", "vaapi"], default="vaapi")
    parser.add_argument("--font", default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    run(parser.parse_args())
