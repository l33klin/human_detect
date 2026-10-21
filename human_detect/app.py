import asyncio
import hmac
import io
import json
import re
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from human_detect.config import Settings
from human_detect.executor import Busy, InferenceExecutor
from human_detect.identity import IdentityStore
from human_detect.pipeline import Pipeline
from human_detect.runtime import EngineUnavailable, ModelBundle
from human_detect.schemas import DetectionResult, StreamRequest
from human_detect.streams import StreamManager


def decode_image(data, max_pixels):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > max_pixels:
                    raise HTTPException(413, "image pixel limit exceeded")
                return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(422, "invalid or oversized image") from exc


def create_app(settings=None, model_factory=ModelBundle):
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        models = model_factory(cfg)
        store = IdentityStore(cfg.database, models.fingerprint, models.dimension)
        executor = InferenceExecutor(cfg.inference_queue_size)
        pipeline = Pipeline(cfg, models, store)
        streams = StreamManager(cfg, pipeline, executor)
        app.state.pipeline, app.state.store = pipeline, store
        app.state.executor, app.state.streams = executor, streams
        try:
            yield
        finally:
            await asyncio.to_thread(streams.close)
            await asyncio.to_thread(executor.close)
            store.close()
            models.close()

    app = FastAPI(title="Human Detect · Coral Edge TPU", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def authorization(request, call_next):
        public = request.url.path in {"/", "/health/live"} or request.url.path.startswith("/ui/")
        if cfg.api_key and not public:
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Bearer {cfg.api_key}".encode()):
                return JSONResponse({"detail": "invalid API key"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI's default error includes the rejected value (possibly an RTSP password).
        errors = [
            dict(loc=error["loc"], msg=error["msg"], type=error["type"]) for error in exc.errors()
        ]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.exception_handler(Busy)
    async def busy_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=429, headers={"Retry-After": "1"})

    @app.exception_handler(EngineUnavailable)
    async def tpu_error(request, exc):
        return JSONResponse(
            {"detail": "TPU unavailable; check server and restart"}, status_code=503
        )

    async def image_body(request):
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > cfg.max_upload_bytes:
                    raise HTTPException(413, "image upload too large")
            except ValueError as exc:
                raise HTTPException(400, "invalid content length") from exc
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > cfg.max_upload_bytes:
                raise HTTPException(413, "image upload too large")
        return await asyncio.to_thread(decode_image, data, cfg.max_image_pixels)

    async def infer(function, *args):
        future = app.state.executor.submit(function, *args)
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=cfg.inference_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            if future.running():
                app.state.pipeline.failure = "TPU inference timed out; restart required"
            future.cancel()
            raise HTTPException(504, "inference timed out") from exc

    def find_stream(stream_id):
        stream = app.state.streams.get(stream_id)
        if stream is None:
            raise HTTPException(404, "stream not found")
        return stream

    def snapshot(stream, request):
        value = dict(stream) if isinstance(stream, dict) else stream.snapshot()
        if value.get("publish"):
            host = cfg.media_public_host or request.url.hostname
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            value["output"]["rtsp_url"] = (
                f"rtsp://{host}:{cfg.media_rtsp_port}/{value['output']['path']}"
            )
            value["output"]["whep_url"] = f"/v1/streams/{value['stream_id']}/whep"
        return value

    web = Path(__file__).with_name("web")
    app.mount("/ui", StaticFiles(directory=web), name="ui")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/v1/config")
    def configuration():
        return dict(
            media_enabled=cfg.media_enabled,
            inference_fps=cfg.media_inference_fps if cfg.media_enabled else cfg.stream_fps,
            max_streams=cfg.max_streams,
        )

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        pipeline = app.state.pipeline
        if pipeline.failure:
            raise HTTPException(503, "TPU unavailable")
        return {
            "status": "ready",
            "device": cfg.device,
            "cpu_fallback": False,
            "models": pipeline.models.report,
        }

    @app.post("/v1/images/analyze", response_model=DetectionResult)
    async def analyze(request: Request):
        rgb = await image_body(request)
        return await infer(app.state.pipeline.analyze, rgb)

    @app.post("/v1/identities/{person_id}/samples", status_code=201)
    async def enroll(
        request: Request, person_id: str, name: str = Query(..., min_length=1, max_length=100)
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", person_id):
            raise HTTPException(422, "person_id must use 1..64 letters, digits, '_' or '-'")
        if not name.strip():
            raise HTTPException(422, "name must not be blank")
        rgb = await image_body(request)
        try:
            return await infer(app.state.pipeline.enroll, rgb, person_id, name.strip())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/identities")
    def identities():
        return app.state.store.list_people()

    @app.delete("/v1/identities/{person_id}")
    def delete_identity(person_id: str):
        if not app.state.store.delete(person_id):
            raise HTTPException(404, "identity not found")
        return {"deleted": True}

    @app.post("/v1/streams", status_code=201)
    async def create_stream(body: StreamRequest, request: Request):
        app.state.pipeline._check()
        try:
            stream = await asyncio.to_thread(
                app.state.streams.create, body.source, body.fps, body.name, body.publish
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return snapshot(stream, request)

    @app.get("/v1/streams")
    def streams(request: Request):
        return [snapshot(s, request) for s in app.state.streams.list()]

    @app.get("/v1/streams/{stream_id}")
    def get_stream(stream_id: str, request: Request):
        return snapshot(find_stream(stream_id), request)

    @app.post("/v1/streams/{stream_id}/start")
    async def start_stream(stream_id: str, request: Request):
        app.state.pipeline._check()
        try:
            stream = await asyncio.to_thread(app.state.streams.start_stream, stream_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if stream is None:
            raise HTTPException(404, "stream not found")
        return snapshot(stream, request)

    @app.post("/v1/streams/{stream_id}/stop")
    async def stop_stream(stream_id: str, request: Request):
        stream = await asyncio.to_thread(app.state.streams.stop_stream, stream_id)
        if stream is None:
            raise HTTPException(404, "stream not found")
        return snapshot(stream, request)

    @app.api_route("/v1/streams/{stream_id}/whep", methods=["POST", "OPTIONS"])
    @app.api_route(
        "/v1/streams/{stream_id}/whep/{session_id}", methods=["PATCH", "DELETE", "OPTIONS"]
    )
    async def whep(stream_id: str, request: Request, session_id: str = ""):
        stream = find_stream(stream_id)
        if not cfg.media_enabled or not stream.snapshot().get("publish"):
            raise HTTPException(409, "stream has no media output")
        if session_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise HTTPException(404, "session not found")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                raise HTTPException(413, "SDP too large")
        path = f"/annotated/{stream_id}/whep" + (f"/{session_id}" if session_id else "")
        headers = {
            key: request.headers[key]
            for key in ("content-type", "if-match")
            if key in request.headers
        }
        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                response = await client.request(
                    request.method,
                    cfg.media_webrtc_url.rstrip("/") + path,
                    content=bytes(body),
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise HTTPException(502, "media server unavailable") from exc
        outgoing = {
            key: response.headers[key]
            for key in ("content-type", "etag", "accept-patch")
            if key in response.headers
        }
        if "location" in response.headers:
            location = urlsplit(response.headers["location"]).path.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", location):
                raise HTTPException(502, "invalid media session")
            outgoing["Location"] = f"/v1/streams/{stream_id}/whep/{location}"
        # Upstream error bodies can contain implementation details; keep responses bounded.
        content = response.content if response.is_success else b"media session unavailable"
        return Response(content, response.status_code, headers=outgoing)

    @app.delete("/v1/streams/{stream_id}")
    async def delete_stream(stream_id: str):
        if not await asyncio.to_thread(app.state.streams.delete, stream_id):
            raise HTTPException(404, "stream not found")
        return {"deleted": True}

    @app.get("/v1/streams/{stream_id}/events")
    async def events(stream_id: str, request: Request):
        stream = find_stream(stream_id)

        async def generate():
            sequence = -1
            while not await request.is_disconnected():
                event = await asyncio.to_thread(stream.wait_event, sequence)
                sequence = event["sequence"]
                yield json.dumps(snapshot(stream, request), ensure_ascii=False) + "\n"
                if event["state"] in {"failed", "stopped"}:
                    break

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
