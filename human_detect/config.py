from pathlib import Path

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
