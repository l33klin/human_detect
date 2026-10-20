import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from human_detect.app import create_app
from human_detect.config import Settings
from tests.fakes import fake_models


@pytest.fixture
def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (240, 240), "white").save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, database=tmp_path / "identities.sqlite3")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, fake_models)) as client:
        yield client
