"""Timestamped A/V transport between FFmpeg and the single TPU owner.

NUT carries RGB video and PCM audio with their original, jointly rebased PTS.
Only one shared RGB frame is in flight. A response belongs to that exact frame;
the worker never paints an unrelated 'latest result' onto a newer frame.
"""

import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from fractions import Fraction
from multiprocessing.shared_memory import SharedMemory

import av
import numpy as np

from human_detect.annotation import Annotator
from human_detect.video import decoder_command


def media_decoder_command(source, cfg):
    command = decoder_command(
        source, cfg.capture_backend, cfg.vaapi_device, cfg.stream_timeout_ms, cfg.stream_fps
    )
    command = command[: command.index("-vf")]
    command += ["-copyts", "-start_at_zero", "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"]
    if cfg.capture_backend == "vaapi":
        command += ["-vf", "hwdownload,format=nv12,format=yuv420p"]
    return command + [
        "-vsync",
        "0",
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-threads:v",
        "1",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-f",
        "nut",
        "-flush_packets",
        "1",
        "pipe:1",
    ]


def encoder_command(url, rate, cfg):
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
    ]
    if cfg.media_encoder == "vaapi":
        command += ["-vaapi_device", cfg.vaapi_device]
    command += [
        "-copyts",
        "-probesize",
        "32768",
        "-analyzeduration",
        "100000",
        "-f",
        "nut",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vsync",
        "0",
    ]
    if cfg.media_encoder == "vaapi":
        command += [
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "h264_vaapi",
            "-profile:v",
            "constrained_baseline",
            "-rc_mode",
            "CQP",
            "-qp",
            str(cfg.media_qp),
        ]
    else:
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-threads:v",
            "1",
        ]
        command += [
            "-b:v",
            str(cfg.media_bitrate),
            "-maxrate",
            str(cfg.media_bitrate),
            "-bufsize",
            str(cfg.media_bitrate // 2),
        ]
    return command + [
        "-bf",
        "0",
        "-g",
        str(max(1, round(rate))),
        "-c:a",
        "libopus",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-b:a",
        "64000",
        "-application",
        "lowdelay",
        "-flush_packets",
        "1",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        url,
    ]


def stop_process(process):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    for pipe in (process.stdin, process.stdout):
        if pipe:
            pipe.close()


class MediaConnection:
    def __init__(self, source, url, cfg):
        self.decoder = self.encoder = self.input = self.output = None
        self.writer = None
        self.writer_stop = threading.Event()
        self.writer_error = None
        self.packets = queue.Queue(maxsize=4)
        self.written = 0
        try:
            self.decoder = subprocess.Popen(
                media_decoder_command(source, cfg),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self.input = av.open(self.decoder.stdout, format="nut", mode="r")
            video = self.input.streams.video[0]
            if video.width * video.height > cfg.max_image_pixels:
                raise ValueError("frame_pixel_limit_exceeded")
            # Rawvideo in NUT often omits average_rate while retaining its
            # nominal rate. Preserve that rate for cadence and output metadata.
            self.rate = video.average_rate or video.guessed_rate or Fraction(25)
            if not 0 < self.rate <= 120:
                raise ValueError("invalid_source_rate")
            self.encoder = subprocess.Popen(
                encoder_command(url, self.rate, cfg),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self.output = av.open(
                self.encoder.stdin,
                format="nut",
                mode="w",
                options={"max_interleave_delta": "100000", "flush_packets": "1"},
            )
            self.video = self.output.add_stream("rawvideo", rate=self.rate)
            self.video.width, self.video.height = video.width, video.height
            self.video.pix_fmt = "rgb24"
            self.video.time_base = Fraction(1, 90000)
            self.audio = {
                stream.index: self.output.add_stream(template=stream)
                for stream in self.input.streams.audio
            }
            # One owner muxes both A/V tracks. A small queue overlaps native
            # pipe writes with the next frame's TPU/tracking work, without
            # reusing the mutable shared inference buffer for queued pixels.
            self.writer = threading.Thread(target=self._write_packets, daemon=True)
            self.writer.start()
        except BaseException:
            self.close()
            raise

    def video_frame(self, rgb, pts, time_base):
        self._enqueue(("video", (rgb, pts, time_base)))

    def _encode_video(self, rgb, pts, time_base):
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts, frame.time_base = pts, time_base
        for packet in self.video.encode(frame):
            self.output.mux(packet)

    def audio_packet(self, packet):
        self._enqueue(("audio", packet))

    def _enqueue(self, item):
        if self.writer_error:
            raise RuntimeError("media_writer_failed")
        self.packets.put(item, timeout=1)
        if self.writer_error:
            raise RuntimeError("media_writer_failed")

    def _write_packets(self):
        try:
            while not self.writer_stop.is_set():
                try:
                    kind, item = self.packets.get(timeout=0.2)
                except queue.Empty:
                    continue
                if kind == "video":
                    self._encode_video(*item)
                    self.written += 1
                else:
                    item.stream = self.audio[item.stream.index]
                    self.output.mux(item)
        except Exception:
            self.writer_error = True

    def close(self):
        # Terminate first: closing a muxer/reader against a blocked pipe can hang.
        self.writer_stop.set()
        stop_process(self.decoder)
        stop_process(self.encoder)
        if self.writer:
            self.writer.join(timeout=2)
        for container in (self.input, self.output):
            if container:
                try:
                    container.close()
                except (OSError, ValueError):
                    pass


def media_worker(source, url, cfg, requests, responses, stop, shared_name):
    """All native media I/O is killable as one private process group."""
    if hasattr(os, "setsid"):
        os.setsid()
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 2)
    requests.cancel_join_thread()
    while not stop.is_set():
        media, shared = None, None
        epoch = uuid.uuid4().hex
        try:
            media = MediaConnection(source, url, cfg)
            annotator = Annotator(cfg.media_font, max(14, min(24, round(media.video.width / 80))))
            clock, first_pts, previous_pts = None, None, None
            dropped = index = 0
            output_times = deque(maxlen=100)
            timings, previous_done = {}, time.monotonic()
            for packet in media.input.demux():
                if stop.is_set():
                    break
                if media.encoder.poll() is not None:
                    raise RuntimeError("encoder_or_publisher_failed")
                if packet.pts is None:
                    continue
                if packet.stream.type == "video":
                    index += 1
                seconds = float(packet.pts * packet.time_base)
                # Keep continuous audio even when overloaded video frames are
                # discarded. Dropping each late audio packet makes speech choppy.
                if packet.stream.type == "audio":
                    media.audio_packet(packet)
                    continue
                if clock is not None and time.monotonic() - clock - (seconds - first_pts) > (
                    cfg.media_max_lag_seconds
                ):
                    dropped += packet.stream.type == "video"
                    continue
                for frame in packet.decode():
                    ready = time.monotonic()
                    if frame.pts is None:
                        raise RuntimeError("source_timestamp_missing")
                    seconds = float(frame.pts * frame.time_base)
                    if previous_pts is not None and not 0 < seconds - previous_pts < 5:
                        raise RuntimeError("source_timestamp_discontinuity")
                    previous_pts = seconds
                    if clock is None:
                        clock, first_pts = time.monotonic(), seconds
                    rgb = frame.to_ndarray(format="rgb24")
                    if rgb.shape[:2] != (media.video.height, media.video.width):
                        raise RuntimeError("source_resolution_changed")
                    if shared is None:
                        shared = SharedMemory(name=shared_name, create=True, size=rgb.nbytes)
                    np.ndarray(rgb.shape, np.uint8, buffer=shared.buf)[:] = rgb
                    reference = dict(
                        source_epoch=epoch,
                        frame_id=index,
                        pts=frame.pts,
                        time_base=str(frame.time_base),
                        seconds=seconds,
                    )
                    requests.put(
                        dict(
                            type="frame",
                            shared=shared.name,
                            shape=rgb.shape,
                            captured=time.time(),
                            frame=reference,
                            rate=float(media.rate),
                            output_frames=media.written,
                            dropped_frames=dropped,
                            audio=bool(media.audio),
                            previous_frame_ms=timings,
                            output_fps=round(
                                (len(output_times) - 1) / (output_times[-1] - output_times[0]), 2
                            )
                            if len(output_times) > 1
                            else 0,
                        ),
                        timeout=1,
                    )
                    response = responses.get(timeout=cfg.inference_timeout_seconds + 2)
                    inferred = time.monotonic()
                    if response.get("frame") != reference or "result" not in response:
                        raise RuntimeError("inference_response_mismatch")
                    annotator.draw(rgb, response["result"]["persons"])
                    annotated = time.monotonic()
                    media.video_frame(rgb, frame.pts, frame.time_base)
                    done = time.monotonic()
                    timings = dict(
                        read=round((ready - previous_done) * 1000, 2),
                        inference=round((inferred - ready) * 1000, 2),
                        annotate=round((annotated - inferred) * 1000, 2),
                        mux=round((done - annotated) * 1000, 2),
                    )
                    previous_done = done
                    output_times.append(time.monotonic())
            if not stop.is_set():
                raise RuntimeError("source_unavailable")
        except Exception:
            try:
                requests.put(dict(type="reconnecting", error="media_connection_failed"), timeout=1)
            except queue.Full:
                pass
        finally:
            if media:
                media.close()
            if shared:
                shared.close()
                shared.unlink()
        stop.wait(2)
