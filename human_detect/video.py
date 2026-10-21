"""FFmpeg RGB frames with explicit VAAPI selection and no software fallback."""

import subprocess

import numpy as np


def decoder_command(source, backend, device, timeout_ms, output_fps):
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
    hardware = backend == "vaapi"
    if hardware:
        command += [
            "-hwaccel",
            "vaapi",
            "-hwaccel_device",
            device,
            "-hwaccel_output_format",
            "vaapi",
        ]
    elif backend != "ffmpeg":
        raise ValueError("expected ffmpeg or vaapi backend")
    if isinstance(source, int):
        command += ["-f", "v4l2", "-i", f"/dev/video{source}"]
    else:
        if source.startswith(("rtsp://", "rtsps://")):
            # RTSP timeout option names differ across FFmpeg releases. Its
            # killable capture process is supervised by the stream watchdog.
            command += ["-rtsp_transport", "tcp"]
        else:
            command += ["-rw_timeout", str(timeout_ms * 1000)]
        command += ["-i", source]
    # This filter keeps hardware surfaces on-device; discarded frames never
    # reach hwdownload, RGB conversion or the inter-process pipe.
    filters = [f"fps=fps={float(output_fps):g}"]
    if hardware:
        # CPU plane rearrangement avoids FFmpeg 4.4's slow direct NV12->RGB
        # conversion. Only selected frames leave the mandatory hardware decoder.
        filters += ["hwdownload", "format=nv12", "format=yuv420p"]
    command += ["-vf", ",".join(filters)]
    # PPM carries dimensions with each frame; no second probe connection or
    # assumption about resolution is needed. The P6 payload is packed RGB24.
    return command + [
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-c:v",
        "ppm",
        "-pix_fmt",
        "rgb24",
        "-threads:v",
        "1",
        "-f",
        "image2pipe",
        "pipe:1",
    ]


class FFmpegCapture:
    def __init__(self, source, backend, device, timeout_ms, max_pixels, output_fps):
        self.max_pixels = max_pixels
        self.command = decoder_command(source, backend, device, timeout_ms, output_fps)
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=65536,
        )

    def read(self):
        stream = self.process.stdout
        magic = stream.readline(128)
        if not magic:
            return None
        if magic != b"P6\n":
            raise RuntimeError("invalid video frame header")
        dimensions = stream.readline(128).split()
        if len(dimensions) != 2:
            raise RuntimeError("invalid video frame dimensions")
        width, height = map(int, dimensions)
        if width <= 0 or height <= 0 or width * height > self.max_pixels:
            raise RuntimeError("frame pixel limit exceeded")
        if stream.readline(128) != b"255\n":
            raise RuntimeError("unsupported video pixel range")
        data = bytearray(width * height * 3)
        view = memoryview(data)
        offset = 0
        while offset < len(data):
            count = stream.readinto(view[offset:])
            if not count:
                raise RuntimeError("truncated video frame")
            offset += count
        return np.frombuffer(data, np.uint8).reshape(height, width, 3)

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.process.stdout.close()
