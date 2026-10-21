from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HD_", env_file=".env", extra="ignore")

    models_dir: Path = Path("models")
    database: Path = Path("data/identities.sqlite3")
    device: str = "pci:0"
    api_key: str = ""
    person_threshold: float = Field(0.5, gt=0, le=1)
    face_threshold: float = Field(0.6, gt=0, le=1)
    similarity_threshold: float = Field(0.65, gt=0, le=1)
    identity_margin: float = Field(0.08, ge=0, le=1)
    min_face_pixels: int = Field(32, ge=8)
    max_people: int = Field(20, ge=1, le=100)
    max_streams: int = Field(4, ge=1, le=32)
    stream_fps: float = Field(5, gt=0, le=60)
    stream_timeout_ms: int = Field(5000, ge=1000, le=30000)
    max_upload_bytes: int = Field(10 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(20_000_000, ge=1024)
    inference_queue_size: int = Field(4, ge=1, le=100)
    inference_timeout_seconds: float = Field(15, gt=0)
    temporal_enabled: bool = True
    temporal_detect_every: int = Field(5, ge=1, le=30)
    temporal_detection_seconds: float = Field(0.5, gt=0, le=2)
    temporal_identity_seconds: float = Field(1, gt=0, le=5)
    empty_scene_enabled: bool = True
    empty_scene_seconds: float = Field(2, gt=0, le=10)
    motion_pixel_threshold: int = Field(18, ge=1, le=255)
    motion_changed_fraction: float = Field(0.002, gt=0, le=1)
    capture_backend: Literal["ffmpeg", "vaapi"] = "ffmpeg"
    vaapi_device: str = "/dev/dri/renderD128"
    media_enabled: bool = False
    media_inference_fps: float = Field(3, gt=0, le=60)
    media_publish_url: str = "rtsp://127.0.0.1:8554"
    media_webrtc_url: str = "http://127.0.0.1:8889"
    media_public_host: str = ""
    media_rtsp_port: int = Field(8554, ge=1, le=65535)
    media_encoder: Literal["vaapi", "libx264"] = "vaapi"
    media_bitrate: int = Field(4_000_000, ge=100_000, le=50_000_000)
    media_qp: int = Field(24, ge=1, le=51)
    media_max_lag_seconds: float = Field(0.75, gt=0, le=5)
    media_font: str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
