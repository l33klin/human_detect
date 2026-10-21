import time
from types import SimpleNamespace

import numpy as np

from human_detect.schemas import Box
from human_detect.streams import replace_latest


class Detector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0
        self.fail = False

    def detect(self, image, threshold, class_id=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("device disconnected")
        return self.boxes


class Embedder:
    def __init__(self):
        self.calls = 0

    def embed(self, image):
        self.calls += 1
        return np.array([1, 0, 0], dtype=np.float32)


def fake_models(settings):
    return SimpleNamespace(
        person=Detector([(Box(x1=10, y1=20, x2=200, y2=230), 0.95)]),
        face=Detector([(Box(x1=20, y1=30, x2=80, y2=90), 0.9)]),
        embedding=Embedder(),
        dimension=3,
        fingerprint="test-model-v1",
        report={},
        close=lambda: None,
    )


def synthetic_capture(source, timeout_ms, channel, stop, *options):
    """Spawn-safe test video source, no TPU simulation is exposed by production CLI."""
    channel.cancel_join_thread()
    while not stop.is_set():
        replace_latest(channel, (time.time(), np.zeros((240, 240, 3), dtype=np.uint8)))
        stop.wait(0.02)


def synthetic_media(source, url, cfg, requests, responses, stop, shared_name):
    import queue
    from multiprocessing.shared_memory import SharedMemory

    shared = SharedMemory(name=shared_name, create=True, size=240 * 240 * 3)
    requests.cancel_join_thread()
    np.ndarray((240, 240, 3), np.uint8, buffer=shared.buf)[:] = 0
    try:
        index = 0
        while not stop.is_set():
            index += 1
            frame = dict(
                source_epoch="test",
                frame_id=index,
                pts=index * 3600,
                time_base="1/90000",
                seconds=index / 25,
            )
            requests.put(
                dict(
                    type="frame",
                    shared=shared.name,
                    shape=(240, 240, 3),
                    captured=time.time(),
                    frame=frame,
                    rate=25,
                    output_frames=index - 1,
                    dropped_frames=0,
                    audio=True,
                )
            )
            try:
                response = responses.get(timeout=2)
            except queue.Empty:
                break
            assert response["frame"] == frame
            stop.wait(0.04)
    finally:
        shared.close()
        shared.unlink()
