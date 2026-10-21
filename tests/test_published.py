import io
import json
import multiprocessing
import time
from fractions import Fraction
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace

import av
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from human_detect.app import create_app
from human_detect.media import MediaConnection, encoder_command, media_decoder_command
from human_detect.published import PublishedStream
from tests.fakes import fake_models, synthetic_media


def wait_result(client, key):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = client.get(f"/v1/streams/{key}").json()
        if value["processed"] >= 2:
            return value
        time.sleep(0.03)
    pytest.fail(str(value))


def test_frame_pairing_persistence_stop_restore_and_shared_memory_cleanup(settings, monkeypatch):
    settings.media_enabled = True
    monkeypatch.setattr("human_detect.published.media_worker", synthetic_media)
    application = create_app(settings, fake_models)
    with TestClient(application) as client:
        response = client.post(
            "/v1/streams",
            json=dict(
                source="rtsp://user:secret@camera/live?token=private", name="入口", publish=True
            ),
        )
        assert response.status_code == 201
        key = response.json()["stream_id"]
        value = wait_result(client, key)
        assert value["latest"]["frame"] == value["media"]["frame"]
        assert value["latest"]["frame"]["time_base"] == "1/90000"
        assert value["output"]["rtsp_url"] == f"rtsp://testserver:8554/annotated/{key}"
        assert "secret" not in json.dumps(value)
        assert "private" not in json.dumps(value)
        assert settings.database.with_name("streams.sqlite3").stat().st_mode & 0o777 == 0o600
        shared_name = application.state.streams.get(key).shared_name
        assert client.post(f"/v1/streams/{key}/stop").json()["state"] == "stopped"
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=shared_name)
    # Explicitly stopped streams remain stopped after a service restart.
    with TestClient(create_app(settings, fake_models)) as client:
        value = client.get("/v1/streams").json()[0]
        assert value["stream_id"] == key and value["state"] == "stopped"
        assert client.post(f"/v1/streams/{key}/start").status_code == 200
        wait_result(client, key)
    # A running stream resumes automatically, keeping its public output URL.
    with TestClient(create_app(settings, fake_models)) as client:
        wait_result(client, key)
        assert client.delete(f"/v1/streams/{key}").status_code == 200
    with TestClient(create_app(settings, fake_models)) as client:
        assert client.get("/v1/streams").json() == []


def test_whep_proxy_rewrites_location_and_hides_upstream(settings, monkeypatch):
    settings.media_enabled = True
    monkeypatch.setattr("human_detect.published.media_worker", synthetic_media)
    seen = []

    def upstream(request):
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(
                201,
                text="answer",
                headers={
                    "content-type": "application/sdp",
                    "etag": '"revision"',
                    "location": "http://mediamtx:8889/annotated/key/whep/session-123",
                },
            )
        return httpx.Response(204)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "human_detect.app.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(upstream), **kwargs),
    )
    with TestClient(create_app(settings, fake_models)) as client:
        key = client.post("/v1/streams", json={"source": "rtsp://camera/live"}).json()["stream_id"]
        path = f"/v1/streams/{key}/whep"
        response = client.post(path, content="offer", headers={"Content-Type": "application/sdp"})
        assert response.status_code == 201 and response.text == "answer"
        assert response.headers["location"] == path + "/session-123"
        assert response.headers["etag"] == '"revision"'
        assert client.delete(response.headers["location"]).status_code == 204
        assert seen[0].url.path == f"/annotated/{key}/whep"
        assert client.post(path, content=b"x" * 65537).status_code == 413


def test_media_commands_preserve_timestamps_and_do_not_drop_video_rate(settings):
    settings.capture_backend = "vaapi"
    source = "rtsp://user:secret@camera/live"
    command = media_decoder_command(source, settings)
    assert command[command.index("-i") + 1] == source
    assert "fps=" not in " ".join(command)
    assert "-rw_timeout" not in command
    assert "-copyts" in command and "0:a:0?" in command
    assert command[command.index("-hwaccel_output_format") + 1] == "vaapi"
    output = encoder_command("rtsp://localhost/output", 25, settings)
    assert output[output.index("-c:v") + 1] == "h264_vaapi"
    assert output[output.index("-rc_mode") + 1] == "CQP"
    assert "libx264" not in output and "libopus" in output
    assert output[output.index("-bf") + 1] == "0"


@pytest.mark.parametrize("rate", [Fraction(60000, 1001), Fraction(25)])
def test_nut_without_average_rate_preserves_source_cadence(settings, monkeypatch, rate):
    # An initial rawvideo NUT frame carries the nominal rate even when there
    # are not enough frames to estimate average_rate during the live probe.
    source = io.BytesIO()
    with av.open(source, mode="w", format="nut") as container:
        video = container.add_stream("rawvideo", rate=rate)
        video.width, video.height, video.pix_fmt = 32, 24, "rgb24"
        frame = av.VideoFrame.from_ndarray(np.zeros((24, 32, 3), np.uint8), format="rgb24")
        frame.pts, frame.time_base = 0, 1 / rate
        for packet in video.encode(frame):
            container.mux(packet)
    source.seek(0)
    with av.open(io.BytesIO(source.getvalue()), format="nut") as container:
        assert container.streams.video[0].average_rate is None
        assert container.streams.video[0].guessed_rate == rate

    def process(**pipes):
        fields = dict(stdin=None, stdout=None, terminate=lambda: None, wait=lambda **_: 0)
        fields.update(pipes)
        return SimpleNamespace(**fields)

    decoder = process(stdout=source)
    encoder = process(stdin=io.BytesIO())
    processes = iter((decoder, encoder))
    monkeypatch.setattr("human_detect.media.subprocess.Popen", lambda *a, **k: next(processes))
    connection = MediaConnection("rtsp://camera/live", "rtsp://localhost/output", settings)
    try:
        assert connection.rate == rate
        assert connection.video.average_rate == rate
    finally:
        connection.close()


def test_ui_is_available_when_api_requires_key(settings):
    settings.api_key = "test-key"
    with TestClient(create_app(settings, fake_models)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/ui/app.js").status_code == 200
        assert client.get("/v1/config").status_code == 401
        assert (
            client.post(
                "/v1/streams",
                json={"source": "rtsp://camera/live"},
                headers={"Authorization": "Bearer test-key"},
            ).status_code
            == 201
        )


def test_killed_worker_shared_memory_is_reclaimed(settings, monkeypatch):
    settings.media_enabled = True
    monkeypatch.setattr("human_detect.published.media_worker", synthetic_media)
    app = create_app(settings, fake_models)
    with TestClient(app) as client:
        key = client.post("/v1/streams", json={"source": "rtsp://camera/live"}).json()["stream_id"]
        wait_result(client, key)
        stream = app.state.streams.get(key)
        stream.process.kill()
        stream.process.join(timeout=2)
        assert client.delete(f"/v1/streams/{key}").status_code == 200
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=stream.shared_name)


def test_full_response_queue_restarts_instead_of_killing_supervisor(settings, monkeypatch):
    with TestClient(create_app(settings, fake_models)) as client:
        stream = PublishedStream(
            "rtsp://camera/live",
            3,
            settings,
            client.app.state.pipeline,
            client.app.state.executor,
            multiprocessing.get_context("spawn"),
        )
        restarted = []
        monkeypatch.setattr(stream, "_restart_capture", lambda: restarted.append(True))
        try:
            stream.responses.put({"stale": True})
            assert not stream._reply({"frame_id": 1}, {"persons": []})
            assert restarted == [True]
            assert stream.snapshot()["state"] == "reconnecting"
        finally:
            stream.close()


def test_restart_storage_failure_does_not_launch_unmanaged_worker(settings, monkeypatch):
    settings.media_enabled = True
    monkeypatch.setattr("human_detect.published.media_worker", synthetic_media)
    with TestClient(create_app(settings, fake_models)) as client:
        key = client.post("/v1/streams", json={"source": "rtsp://camera/live"}).json()["stream_id"]
        wait_result(client, key)
        client.post(f"/v1/streams/{key}/stop")

        def fail_save(*args):
            raise OSError("storage unavailable")

        with monkeypatch.context() as changed:
            changed.setattr(client.app.state.streams.store, "save", fail_save)
            changed.setattr(
                PublishedStream, "start", lambda self: pytest.fail("must persist first")
            )
            with pytest.raises(OSError, match="storage unavailable"):
                client.post(f"/v1/streams/{key}/start")
