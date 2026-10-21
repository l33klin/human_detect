import threading
import time

import pytest

from human_detect.executor import Busy, InferenceExecutor
from human_detect.identity import IdentityStore
from human_detect.pipeline import Pipeline
from human_detect.streams import StreamManager
from tests.fakes import fake_models, synthetic_capture


def test_bounded_admission_and_cancelled_queued_work():
    executor = InferenceExecutor(1)
    started, release = threading.Event(), threading.Event()

    def blocked():
        started.set()
        release.wait(5)

    try:
        first = executor.submit(blocked)
        assert started.wait(2)
        second = executor.submit(lambda: 2)
        with pytest.raises(Busy):
            executor.submit(lambda: 3)
        assert second.cancel()
        third = executor.submit(lambda: 3)
        release.set()
        first.result(timeout=2)
        assert third.result(timeout=2) == 3
    finally:
        release.set()
        executor.close()


@pytest.mark.parametrize(
    "temporal,motion", [(False, False), (False, True), (True, False), (True, True)]
)
def test_spawned_video_capture_json_results_and_cleanup(settings, monkeypatch, temporal, motion):
    monkeypatch.setattr("human_detect.streams.capture_frames", synthetic_capture)
    settings.temporal_enabled = temporal
    settings.empty_scene_enabled = motion
    settings.max_streams = 1
    executor = InferenceExecutor(2)
    store = IdentityStore(settings.database, "test-model-v1", 3)
    pipeline = Pipeline(settings, fake_models(settings), store)
    manager = StreamManager(settings, pipeline, executor)
    try:
        stream = manager.create("rtsp://user:secret@camera/live", 30)
        with pytest.raises(Busy):
            manager.create(0, 1)
        deadline = time.monotonic() + 15
        while stream.snapshot()["processed"] < 2 and time.monotonic() < deadline:
            stream.condition.acquire()
            stream.condition.wait(timeout=0.1)
            stream.condition.release()
        snapshot = stream.snapshot()
        assert snapshot["processed"] >= 2
        assert snapshot["latest"]["width"] == 240
        assert snapshot["latest"]["persons"][0]["faces"]
        assert ("temporal" in snapshot["latest"]) is temporal
        assert "secret" not in str(snapshot)
        process = stream.process
        assert manager.delete(stream.id)
        assert not process.is_alive()
        assert not stream.thread.is_alive()
        assert stream.snapshot()["state"] == "stopped"
        assert manager.list() == []
    finally:
        manager.close()
        executor.close()
        store.close()
