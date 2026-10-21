import io

import pytest

from human_detect.video import FFmpegCapture, decoder_command


def reader(data, max_pixels=100):
    capture = FFmpegCapture.__new__(FFmpegCapture)
    capture.max_pixels = max_pixels
    capture.process = type("Process", (), {"stdout": io.BytesIO(data)})()
    return capture


def test_frame_framing_and_dynamic_resolution():
    capture = reader(b"P6\n2 1\n255\n" + bytes(range(6)) + b"P6\n1 1\n255\nabc")
    assert capture.read().shape == (1, 2, 3)
    assert capture.read().shape == (1, 1, 3)
    assert capture.read() is None


@pytest.mark.parametrize(
    "data", [b"P6\n999 999\n255\n", b"P6\n2 2\n255\na", b"bad\n", b"P6\n0 1\n255\n"]
)
def test_malformed_and_oversized_frames_fail_before_processing(data):
    with pytest.raises(RuntimeError):
        reader(data).read()


def test_vaapi_requires_hardware_surfaces_and_source_is_single_argument():
    source = "rtsp://user:password@camera/path?x=a;b"
    command = decoder_command(source, "vaapi", "/dev/dri/renderD128", 5000)
    assert command[command.index("-i") + 1] == source
    assert command[command.index("-hwaccel_output_format") + 1] == "vaapi"
    assert command[command.index("-vf") + 1] == "hwdownload,format=nv12"
    assert "-hwaccel" not in decoder_command(source, "ffmpeg", "unused", 5000)
    command = decoder_command(source, "vaapi-i420", "/dev/dri/renderD128", 5000)
    assert command[command.index("-vf") + 1] == "hwdownload,format=nv12,format=yuv420p"
    assert command[command.index("-hwaccel") + 1] == "vaapi"
