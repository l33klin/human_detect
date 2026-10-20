from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from experiments.benchmark import render_frame
from human_detect.identity import IdentityStore
from human_detect.pipeline import Pipeline
from human_detect.runtime import EngineUnavailable
from human_detect.temporal import TemporalSession
from tests.fakes import fake_models


@pytest.fixture
def video(settings):
    store = IdentityStore(settings.database, "test-model-v1", 3)
    models = fake_models(settings)
    pipeline = Pipeline(settings, models, store)
    image = np.random.default_rng(1).integers(0, 255, (240, 240, 3), dtype=np.uint8)
    pipeline.enroll(image, "a", "Alice")
    session = TemporalSession(pipeline)
    base = datetime(2026, 10, 20, tzinfo=timezone.utc)
    yield pipeline, session, image, base
    store.close()


def test_static_video_tracks_and_limits_detection_calls(video):
    pipeline, session, image, base = video
    for i in range(10):
        result = session.analyze(image, base + timedelta(seconds=i / 25))
        assert result.persons[0].faces[0].identity.person_id == "a"
        assert result.temporal["detection_age_ms"] <= 160.01
    assert pipeline.models.person.calls == 2
    assert result.temporal["mode"] == "tracked"


def test_identity_delete_invalidates_video_cache_immediately(video):
    pipeline, session, image, base = video
    session.analyze(image, base)
    pipeline.store.delete("a")
    result = session.analyze(image, base + timedelta(seconds=0.04))
    assert result.temporal["mode"] == "detected"
    assert result.persons[0].faces[0].identity.status == "unknown"


def test_scene_cut_does_not_propagate_identity(video):
    pipeline, session, image, base = video
    session.analyze(image, base)
    pipeline.models.person.boxes = []
    result = session.analyze(np.zeros_like(image), base + timedelta(seconds=0.04))
    assert result.persons == []
    assert result.temporal["mode"] == "detected"


def test_interrupted_timestamps_force_fresh_detection(video):
    pipeline, session, image, base = video
    session.analyze(image, base)
    for timestamp in (base + timedelta(seconds=3), base):
        result = session.analyze(image, timestamp)
        assert result.temporal["mode"] == "detected"
        assert result.temporal["identity_source"] == "current"


def test_sessions_are_independent_and_tpu_failure_is_not_hidden(video):
    pipeline, session, image, base = video
    session.analyze(image, base)
    other = TemporalSession(pipeline)
    assert other.analyze(image, base).temporal["mode"] == "detected"
    pipeline.failure = "device lost"
    with pytest.raises(EngineUnavailable):
        session.analyze(image, base + timedelta(seconds=0.04))


def test_embedding_ttl_forces_refresh_even_during_tracking(video):
    pipeline, session, image, base = video
    result = session.analyze(image, base)
    for i in range(1, 30):
        result = session.analyze(image, base + timedelta(seconds=i / 25))
        assert result.temporal["identity_age_ms"] < 1000


def test_corpus_determinism_and_explicit_changes():
    assets = {
        "reference": np.full((60, 50, 3), 200, np.uint8),
        "alternate": np.full((60, 50, 3), 50, np.uint8),
    }
    first = render_frame(assets, "switch", 1280, 720, 0, 40)
    changed = render_frame(assets, "switch", 1280, 720, 16, 40)
    assert first.shape == (720, 1280, 3)
    assert not np.array_equal(first, changed)
    np.testing.assert_array_equal(first, render_frame(assets, "switch", 1280, 720, 0, 40))
