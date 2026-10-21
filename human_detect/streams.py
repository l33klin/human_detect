"""Capture latest frame in a child process so blocked camera I/O can be terminated."""

import multiprocessing
import os
import queue
import signal
import threading
import time
import uuid
from concurrent.futures import TimeoutError
from datetime import datetime, timezone

from human_detect.executor import Busy
from human_detect.runtime import EngineUnavailable


def replace_latest(channel, item):
    try:
        channel.put_nowait(item)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(item)
        except queue.Full:
            pass


def capture_frames(
    source,
    timeout_ms,
    channel,
    stop,
    backend="ffmpeg",
    device="/dev/dri/renderD128",
    max_pixels=20_000_000,
    output_fps=5,
):
    """No source URL or camera credentials are returned in errors or logs."""
    # FFmpeg may log the source, including credentials, directly to stderr.
    if hasattr(os, "setsid"):
        os.setsid()
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 2)
    channel.cancel_join_thread()
    while not stop.is_set():
        capture = None
        try:
            from human_detect.video import FFmpegCapture

            capture = FFmpegCapture(
                source, backend, device, timeout_ms, max_pixels, output_fps=output_fps
            )
            while not stop.is_set():
                rgb = capture.read()
                if rgb is None:
                    raise RuntimeError("read_failed")
                replace_latest(channel, (time.time(), rgb))
        except Exception:
            replace_latest(channel, (time.time(), None))
        finally:
            if capture is not None:
                capture.close()
        stop.wait(2)


class Stream:
    def __init__(self, source, fps, settings, pipeline, executor, context):
        self.id = uuid.uuid4().hex
        self.fps, self.settings = fps, settings
        self.context, self.source = context, source
        self.pipeline, self.executor = pipeline, executor
        self.video_pipeline = self._make_video_pipeline()
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.capture_stop = context.Event()
        self.frames = context.Queue(maxsize=1)
        self.process = context.Process(
            target=capture_frames,
            args=self._capture_args(),
            daemon=True,
        )
        self.state, self.error = "connecting", None
        self.latest = None
        self.sequence = 0
        self.processed = 0
        self.skipped = 0
        self.thread = threading.Thread(target=self._run, name=f"stream-{self.id}", daemon=True)

    def start(self):
        self.process.start()
        self.thread.start()

    def _make_video_pipeline(self):
        if not self.settings.temporal_enabled:
            return self.pipeline
        if self.settings.empty_scene_enabled:
            from human_detect.motion import EmptySceneSession

            return EmptySceneSession(self.pipeline)
        from human_detect.temporal import TemporalSession

        return TemporalSession(self.pipeline)

    def _capture_args(self):
        cfg = self.settings
        return (
            self.source,
            cfg.stream_timeout_ms,
            self.frames,
            self.capture_stop,
            cfg.capture_backend,
            cfg.vaapi_device,
            cfg.max_image_pixels,
            self.fps,
        )

    def _terminate_capture(self):
        # FFmpeg is a child of the capture process. Kill the private group too,
        # including when the parent exited first, to avoid orphaned decoders.
        grouped = hasattr(os, "killpg")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if grouped:
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
            if self.process.is_alive():
                if sig == signal.SIGTERM:
                    self.process.terminate()
                else:
                    self.process.kill()
            self.process.join(timeout=2)

    def _run(self):
        next_at = 0
        stale_after = max(10, self.settings.stream_timeout_ms / 1000 * 2)
        last_frame_at = time.monotonic()
        while not self.stop.is_set():
            if self.stop.wait(max(0, next_at - time.monotonic())):
                break
            try:
                captured, rgb = self.frames.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() - last_frame_at > stale_after:
                    # A camera driver can hang indefinitely; recreate its process.
                    self._set_state("reconnecting", "capture_timeout")
                    self._restart_capture()
                    last_frame_at = time.monotonic()
                continue
            if rgb is None:
                self.video_pipeline = self._make_video_pipeline()
                self._set_state("reconnecting", "source_unavailable")
                continue
            if rgb.shape[0] * rgb.shape[1] > self.settings.max_image_pixels:
                self._set_state("failed", "frame_pixel_limit_exceeded")
                self.capture_stop.set()
                break
            last_frame_at = time.monotonic()
            next_at = time.monotonic() + 1 / self.fps
            future = None
            try:
                future = self.executor.submit(
                    self.video_pipeline.analyze, rgb, datetime.fromtimestamp(captured, timezone.utc)
                )
                result = future.result(timeout=self.settings.inference_timeout_seconds)
                with self.condition:
                    self.latest = result
                    self.sequence += 1
                    self.processed += 1
                    self.state, self.error = "running", None
                    self.condition.notify_all()
            except Busy:
                with self.lock:
                    self.skipped += 1
                self._set_state("busy", "inference_busy")
            except TimeoutError:
                if future is not None and future.running():
                    self.pipeline.failure = "TPU inference timed out; restart required"
                    self._set_state("failed", "tpu_timeout")
                    self.capture_stop.set()
                    break
                if future is not None:
                    future.cancel()
                self._set_state("busy", "inference_busy")
            except EngineUnavailable:
                self._set_state("failed", "tpu_unavailable")
                self.capture_stop.set()
                break
            except Exception:
                self._set_state("failed", "processing_failed")
                self.capture_stop.set()
                break

    def _set_state(self, state, error):
        with self.condition:
            self.state, self.error = state, error
            self.condition.notify_all()

    def _restart_capture(self):
        if self.stop.is_set():
            return
        self._terminate_capture()
        # A terminated process may corrupt its queue; always create a fresh one.
        self.frames.close()
        self.frames = self.context.Queue(maxsize=1)
        self.video_pipeline = self._make_video_pipeline()
        self.process = self.context.Process(
            target=capture_frames,
            args=self._capture_args(),
            daemon=True,
        )
        self.process.start()

    def snapshot(self):
        with self.lock:
            return dict(
                stream_id=self.id,
                state=self.state,
                error=self.error,
                target_fps=self.fps,
                capture_backend=self.settings.capture_backend,
                processed=self.processed,
                skipped=self.skipped,
                sequence=self.sequence,
                latest=self.latest.model_dump(mode="json") if self.latest else None,
            )

    def wait_event(self, after):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > after or self.state in {"failed", "stopped"}, timeout=10
            )
            return self.snapshot()

    def close(self):
        self.stop.set()
        self.capture_stop.set()
        self.thread.join(timeout=self.settings.inference_timeout_seconds + 3)
        self.process.join(timeout=1)
        self._terminate_capture()
        self.frames.close()
        self._set_state("stopped", None)


class StreamManager:
    def __init__(self, settings, pipeline, executor):
        self.settings, self.pipeline, self.executor = settings, pipeline, executor
        self.context = multiprocessing.get_context("spawn")
        self.lock = threading.Lock()
        self.streams = {}

    def create(self, source, fps):
        with self.lock:
            if len(self.streams) >= self.settings.max_streams:
                raise Busy("stream limit reached")
            stream = Stream(
                source,
                fps or self.settings.stream_fps,
                self.settings,
                self.pipeline,
                self.executor,
                self.context,
            )
            stream.start()
            self.streams[stream.id] = stream
            return stream

    def get(self, stream_id):
        with self.lock:
            return self.streams.get(stream_id)

    def list(self):
        with self.lock:
            return [s.snapshot() for s in self.streams.values()]

    def delete(self, stream_id):
        with self.lock:
            stream = self.streams.get(stream_id)
            if not stream:
                return False
            # Keep the capacity reserved until camera/process shutdown completes.
            stream.close()
            del self.streams[stream_id]
            return True

    def close(self):
        with self.lock:
            for stream in self.streams.values():
                stream.close()
            self.streams.clear()
