import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from human_detect.app import create_app, decode_image
from human_detect.schemas import Box
from tests.fakes import fake_models


def test_enrollment_detection_delete(client, image_bytes):
    assert client.get("/health/ready").json()["cpu_fallback"] is False
    result = client.post("/v1/images/analyze", content=image_bytes).json()
    person = result["persons"][0]
    assert person["box"] == {"x1": 10, "y1": 20, "x2": 200, "y2": 230}
    assert person["faces"][0]["box"] == {"x1": 30, "y1": 50, "x2": 90, "y2": 110}
    assert person["faces"][0]["identity"]["status"] == "unknown"
    response = client.post("/v1/identities/alice/samples?name=Alice", content=image_bytes)
    assert response.status_code == 201
    assert response.json()["samples"] == 1
    assert (
        client.post("/v1/identities/alice/samples?name=Alice", content=image_bytes).json()[
            "samples"
        ]
        == 2
    )
    result = client.post("/v1/images/analyze", content=image_bytes).json()
    identity = result["persons"][0]["faces"][0]["identity"]
    assert identity["person_id"] == "alice"
    assert identity["status"] == "matched"
    assert client.delete("/v1/identities/alice").status_code == 200
    assert client.get("/v1/identities").json() == []
    assert client.delete("/v1/identities/alice").status_code == 404


def test_enrollment_rejects_multiple_or_small_faces(client, image_bytes):
    detector = client.app.state.pipeline.models.face
    detector.boxes = detector.boxes * 2
    assert client.post("/v1/identities/a/samples?name=A", content=image_bytes).status_code == 422
    detector.boxes = [(Box(x1=0, y1=0, x2=20, y2=20), 0.9)]
    assert client.post("/v1/identities/a/samples?name=A", content=image_bytes).status_code == 422
    result = client.post("/v1/images/analyze", content=image_bytes).json()
    assert result["persons"][0]["faces"][0]["identity"]["status"] == "face_too_small"
    assert client.get("/v1/identities").json() == []


def test_no_person_skips_face_inference(client, image_bytes):
    models = client.app.state.pipeline.models
    models.person.boxes = []
    result = client.post("/v1/images/analyze", content=image_bytes).json()
    assert result["persons"] == []
    assert models.face.calls == 0


def test_failed_tpu_latches_unready_without_fallback(client, image_bytes):
    client.app.state.pipeline.models.person.fail = True
    assert client.post("/v1/images/analyze", content=image_bytes).status_code == 503
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health/live").status_code == 200
    client.app.state.pipeline.models.person.fail = False
    assert client.post("/v1/images/analyze", content=image_bytes).status_code == 503


def test_running_inference_timeout_marks_service_unready(client, settings, image_bytes):
    settings.inference_timeout_seconds = 0.02
    detector = client.app.state.pipeline.models.person
    original = detector.detect

    def slow(*args, **kwargs):
        time.sleep(0.08)
        return original(*args, **kwargs)

    detector.detect = slow
    assert client.post("/v1/images/analyze", content=image_bytes).status_code == 504
    assert client.get("/health/ready").status_code == 503


def test_queued_timeout_is_cancelled_without_marking_tpu_failed(client, settings, image_bytes):
    settings.inference_timeout_seconds = 0.02
    started, release = threading.Event(), threading.Event()

    def block():
        started.set()
        release.wait(2)

    future = client.app.state.executor.submit(block)
    assert started.wait(1)
    try:
        assert client.post("/v1/images/analyze", content=image_bytes).status_code == 504
        assert client.get("/health/ready").status_code == 200
        assert client.app.state.pipeline.models.person.calls == 0
    finally:
        release.set()
        future.result(timeout=1)


def test_invalid_and_oversized_image(client, settings, image_bytes):
    assert client.post("/v1/images/analyze", content=b"not an image").status_code == 422
    settings.max_upload_bytes = 1024
    assert client.post("/v1/images/analyze", content=image_bytes).status_code == 413
    settings.max_upload_bytes = 10 * 1024 * 1024
    settings.max_image_pixels = 1024
    assert client.post("/v1/images/analyze", content=image_bytes).status_code == 413


@pytest.mark.parametrize(
    "source",
    [-1, True, "/etc/passwd", "file:///tmp/image.jpg", "http://", "rtsp://user:secret@/invalid"],
)
def test_invalid_source_never_echoes_credentials(client, source):
    response = client.post("/v1/streams", json={"source": source})
    assert response.status_code == 422
    assert "secret" not in response.text


def test_authentication(settings, image_bytes):
    settings.api_key = "test-key"
    with TestClient(create_app(settings, fake_models)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 401
        assert client.get("/docs").status_code == 401
        assert client.post("/v1/images/analyze", content=image_bytes).status_code == 401
        assert (
            client.get("/health/ready", headers={"Authorization": "Bearer test-key"}).status_code
            == 200
        )


def test_exif_orientation_is_applied_before_coordinates():
    image = Image.new("RGB", (80, 40))
    exif = image.getexif()
    exif[274] = 6
    output = io.BytesIO()
    image.save(output, format="JPEG", exif=exif)
    rgb = decode_image(output.getvalue(), 10000)
    assert rgb.shape == (80, 40, 3)


def test_overlap_faces_are_not_reported_twice(client, image_bytes):
    models = client.app.state.pipeline.models
    models.person.boxes *= 2
    result = client.post("/v1/images/analyze", content=image_bytes).json()
    assert sum(len(p["faces"]) for p in result["persons"]) == 1
