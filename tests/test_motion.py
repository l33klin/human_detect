from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from human_detect.identity import IdentityStore
from human_detect.motion import EmptySceneSession
from human_detect.pipeline import Pipeline
from human_detect.runtime import EngineUnavailable
from tests.fakes import fake_models


@pytest.fixture
def empty_video(settings):
    settings.motion_enabled = settings.empty_scene_enabled = True
    models = fake_models(settings)
    models.person.boxes = []
    store = IdentityStore(settings.database, "test-model-v1", 3)
    pipeline = Pipeline(settings, models, store)
    session = EmptySceneSession(pipeline)
    rgb = np.zeros((240, 240, 3), np.uint8)
    base = datetime(2026, 10, 21, tzinfo=timezone.utc)
    yield pipeline, session, rgb, base
    store.close()


def test_empty_scene_watchdog_and_age(empty_video):
    pipeline, session, rgb, base = empty_video
    for i in range(20):
        result = session.analyze(rgb, base + timedelta(seconds=i / 5))
        assert result.persons == []
        assert result.temporal["detection_age_ms"] < 2000
        assert result.temporal["identity_source"] == "none" or i in (0, 10)
    assert pipeline.models.person.calls == 2


def test_motion_wakes_on_entry_and_stationary_person_remains(empty_video):
    pipeline, session, rgb, base = empty_video
    session.analyze(rgb, base)
    boxes = fake_models(pipeline.settings).person.boxes
    pipeline.models.person.boxes = boxes
    entered = rgb.copy()
    entered[30:180, 30:180] = 200
    result = session.analyze(entered, base + timedelta(seconds=0.2))
    assert result.persons
    assert result.temporal["mode"] == "detected"
    result = session.analyze(entered, base + timedelta(seconds=0.4))
    assert result.persons
    assert result.temporal["mode"] != "idle"


def test_small_changes_accumulate_against_verified_empty_frame(empty_video):
    pipeline, session, rgb, base = empty_video
    session.analyze(rgb, base)
    for i in range(1, 5):
        changed = np.full_like(rgb, i * 5)
        session.analyze(changed, base + timedelta(seconds=i / 5))
    assert pipeline.models.person.calls == 2


def test_resets_and_tpu_failure_not_hidden_by_idle(empty_video):
    pipeline, session, rgb, base = empty_video
    session.analyze(rgb, base)
    session.analyze(rgb, base)  # Reversed/repeated timestamps cannot reuse emptiness.
    session.analyze(rgb, base + timedelta(seconds=3))
    session.analyze(np.zeros((120, 120, 3), np.uint8), base + timedelta(seconds=3.2))
    assert pipeline.models.person.calls == 4
    pipeline.failure = "device lost"
    with pytest.raises(EngineUnavailable):
        session.analyze(rgb, base + timedelta(seconds=3.4))


def test_identity_store_change_invalidates_empty_result(empty_video):
    pipeline, session, rgb, base = empty_video
    session.analyze(rgb, base)
    pipeline.store.enroll("x", "X", np.array([1, 0, 0], np.float32))
    session.analyze(rgb, base + timedelta(seconds=0.2))
    assert pipeline.models.person.calls == 2


@pytest.mark.parametrize("motion,idle,calls", [(True, False, 7), (False, True, 2)])
def test_factors_can_be_measured_independently(empty_video, motion, idle, calls):
    pipeline, session, rgb, base = empty_video
    pipeline.settings.motion_enabled = motion
    pipeline.settings.empty_scene_enabled = idle
    for i in range(20):
        session.analyze(rgb, base + timedelta(seconds=i / 5))
    assert pipeline.models.person.calls == calls
