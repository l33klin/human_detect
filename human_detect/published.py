"""One timestamp-aligned annotated publisher per saved RTSP definition."""

import math
import queue
import time
import uuid
from concurrent.futures import TimeoutError
from datetime import datetime, timezone
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from human_detect.executor import Busy
from human_detect.media import media_worker
from human_detect.runtime import EngineUnavailable
from human_detect.streams import Stream


class PublishedStream(Stream):
    def __init__(self, source, fps, settings, pipeline, executor, context, stream_id=None, name=""):
        self.responses = context.Queue(maxsize=1)
        self.shared_name = "hd_" + uuid.uuid4().hex[:20]
        self.media = dict(output_frames=0, dropped_frames=0, audio=False, source_fps=None)
        super().__init__(source, fps, settings, pipeline, executor, context, stream_id, name)

    def _capture_args(self):
        return (
            self.source,
            f"{self.settings.media_publish_url.rstrip('/')}/annotated/{self.id}",
            self.settings,
            self.frames,
            self.responses,
            self.capture_stop,
            self.shared_name,
        )

    def _terminate_capture(self):
        super()._terminate_capture()
        # SIGTERM/SIGKILL bypass the child's finally block. The name is known
        # before it allocates anything, so even a pre-request crash cannot leak.
        try:
            shared = SharedMemory(name=self.shared_name)
        except FileNotFoundError:
            return
        shared.close()
        shared.unlink()

    def _new_process(self):
        return self.context.Process(target=media_worker, args=self._capture_args(), daemon=True)

    def start(self):
        self.process = self._new_process()
        super().start()

    def _restart_capture(self):
        if self.stop.is_set():
            return
        self._terminate_capture()
        self.frames.close()
        self.responses.close()
        self.frames = self.context.Queue(maxsize=1)
        self.responses = self.context.Queue(maxsize=1)
        self.video_pipeline = self._make_video_pipeline()
        self.process = self._new_process()
        self.process.start()

    def _configure_cadence(self, rate):
        # Keep optical tracking at video rate while limiting scheduled full refreshes.
        # Tracking failures can request earlier TPU refreshes for correctness.
        every = max(1, math.ceil(rate / self.fps))
        cfg = self.settings.model_copy(update={"temporal_detect_every": every})
        session = self.video_pipeline
        if session is not self.pipeline:
            session.cfg = cfg
            if hasattr(session, "temporal"):
                session.temporal.cfg = cfg

    def _reply(self, frame, result):
        try:
            self.responses.put(dict(frame=frame, result=result), timeout=1)
            return True
        except queue.Full:
            self._set_state("reconnecting", "media_response_timeout")
            self._restart_capture()
            return False

    def _run(self):
        last_frame = time.monotonic()
        epoch = None
        while not self.stop.is_set():
            try:
                message = self.frames.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() - last_frame > max(10, self.settings.stream_timeout_ms / 500):
                    self._set_state("reconnecting", "media_timeout")
                    self._restart_capture()
                    last_frame, epoch = time.monotonic(), None
                continue
            if message["type"] != "frame":
                self._set_state("reconnecting", message["error"])
                continue
            last_frame = time.monotonic()
            if epoch != message["frame"]["source_epoch"]:
                self.video_pipeline = self._make_video_pipeline()
                epoch = message["frame"]["source_epoch"]
            self._configure_cadence(message["rate"])
            shared, future = None, None
            try:
                shared = SharedMemory(name=message["shared"])
                rgb = np.ndarray(message["shape"], np.uint8, buffer=shared.buf)
                captured = datetime.fromtimestamp(message["captured"], timezone.utc)
                future = self.executor.submit(self.video_pipeline.analyze, rgb, captured)
                result = future.result(timeout=self.settings.inference_timeout_seconds)
                result.frame = message["frame"]
                if not self._reply(message["frame"], result.model_dump(mode="json")):
                    epoch = None
                    continue
                with self.condition:
                    self.latest = result
                    self.sequence += 1
                    self.processed += 1
                    self.media = dict(
                        output_frames=message["output_frames"],
                        dropped_frames=message["dropped_frames"],
                        audio=message["audio"],
                        output_fps=message.get("output_fps", 0),
                        previous_frame_ms=message.get("previous_frame_ms", {}),
                        source_fps=message["rate"],
                        width=rgb.shape[1],
                        height=rgb.shape[0],
                        frame=message["frame"],
                    )
                    self.state, self.error = "running", None
                    self.condition.notify_all()
            except Busy:
                self.skipped += 1
                # No result is safer than a stale identity. Preserve this frame's timing.
                if self._reply(message["frame"], {"persons": []}):
                    self._set_state("busy", "inference_busy")
                else:
                    epoch = None
            except TimeoutError:
                if future and future.running():
                    self.pipeline.failure = "TPU inference timed out; restart required"
                    self._set_state("failed", "tpu_timeout")
                    self.capture_stop.set()
                    break
                if future:
                    future.cancel()
                if self._reply(message["frame"], {"persons": []}):
                    self._set_state("busy", "inference_busy")
                else:
                    epoch = None
            except EngineUnavailable:
                self._set_state("failed", "tpu_unavailable")
                self.capture_stop.set()
                break
            except Exception:
                self._set_state("reconnecting", "media_processing_failed")
                self._restart_capture()
                epoch = None
            finally:
                # A timed-out native inference can still read this buffer. Keep its
                # mapping alive until the future finishes, even if the worker exits.
                if shared:
                    if future and future.running():
                        future.add_done_callback(lambda done, memory=shared: memory.close())
                    else:
                        shared.close()

    def snapshot(self):
        with self.lock:
            value = super().snapshot()
            value.update(
                publish=True,
                media=dict(self.media),
                output=dict(
                    path=f"annotated/{self.id}",
                    video_codec="h264",
                    audio_codec="opus" if self.media.get("audio") else None,
                ),
            )
            if self.state not in {"running", "busy"}:
                value["media"]["output_fps"] = 0
            return value

    def close(self):
        super().close()
        self.responses.close()
