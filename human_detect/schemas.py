from datetime import datetime
from typing import List, Literal, Optional, Union
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator


class Box(BaseModel):
    """Original image pixel coordinates; x2/y2 are exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int


class Identity(BaseModel):
    status: Literal["matched", "unknown", "face_too_small"] = "unknown"
    person_id: Optional[str] = None
    name: Optional[str] = None
    similarity: Optional[float] = None


class Face(BaseModel):
    box: Box
    confidence: float
    identity: Identity


class Person(BaseModel):
    box: Box
    confidence: float
    faces: List[Face] = Field(default_factory=list)


class DetectionResult(BaseModel):
    captured_at: datetime
    width: int
    height: int
    persons: List[Person]
    processing_ms: float
    frame: Optional[dict] = None


class StreamRequest(BaseModel):
    source: Union[int, str]
    fps: Optional[float] = Field(None, gt=0, le=60)
    name: str = Field("", max_length=100)
    publish: Optional[bool] = None

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, value):
        if type(value) is int and 0 <= value <= 64:
            return value
        if isinstance(value, str) and len(value) <= 4096:
            parsed = urlsplit(value)
            if parsed.scheme in {"rtsp", "rtsps", "http", "https"} and parsed.hostname:
                return parsed.geturl()
        raise ValueError("source must be a camera index (0..64) or RTSP/HTTP(S) URL")
