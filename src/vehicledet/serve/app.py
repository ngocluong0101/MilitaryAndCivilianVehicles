"""API FastAPI: POST /predict, GET /health.

Cấu hình qua biến môi trường: MODEL_PATH (.onnx hoặc .engine), BACKEND (auto|onnx|tensorrt),
DEVICE (auto|cpu|cuda), MAX_UPLOAD_MB (mặc định 10). Chạy: uvicorn vehicledet.serve.app:app
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from vehicledet import __version__
from vehicledet.serve.inference import Detector, load_detector

log = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_MB = 10
MAX_PIXELS = 50_000_000  # chặn ảnh "bom giải nén"
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    class_id: int
    label: str
    score: float
    bbox: BBox


class ImageInfo(BaseModel):
    width: int
    height: int


class PredictResponse(BaseModel):
    image: ImageInfo
    detections: list[DetectionOut]
    inference_ms: float
    total_ms: float
    backend: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str | None = None
    model: str | None = None
    classes: list[str] = []
    version: str = __version__
    error: str | None = None


def _read_limited(upload: UploadFile, limit: int) -> bytes:
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File quá lớn (tối đa {limit // 2**20} MB)")
    if not data:
        raise HTTPException(400, "File rỗng")
    return data


def _decode_image(data: bytes) -> Image.Image:
    import io

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.convert("RGB")
    except Image.DecompressionBombError as e:
        raise HTTPException(413, f"Ảnh quá lớn (tối đa {MAX_PIXELS:,} pixel)") from e
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(400, "File không phải ảnh hợp lệ (hỗ trợ JPEG, PNG, WebP, BMP)") from e


def create_app(detector: Detector | None = None, max_upload_mb: float | None = None) -> FastAPI:
    """Tạo app. Truyền `detector` để tiêm model (stub khi test); mặc định nạp từ MODEL_PATH."""
    limit = int(
        (max_upload_mb or float(os.environ.get("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))) * 2**20
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = detector
        app.state.load_error = None
        path = os.environ.get("MODEL_PATH")
        if detector is None and path:
            try:
                app.state.detector = load_detector(
                    path, os.environ.get("BACKEND", "auto"), os.environ.get("DEVICE", "auto")
                )
                log.info("Đã nạp model %s (%s)", path, app.state.detector.backend)
            except Exception as e:  # noqa: BLE001 - app vẫn chạy, /health báo lỗi rõ ràng
                app.state.load_error = f"{type(e).__name__}: {e}"
                log.error("Không nạp được model: %s", app.state.load_error)
        elif detector is None:
            app.state.load_error = "Chưa đặt biến môi trường MODEL_PATH"
        yield

    app = FastAPI(
        title="Military & Civilian Vehicle Detection", version=__version__, lifespan=lifespan
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        det: Detector | None = app.state.detector
        if det is None:
            return HealthResponse(status="degraded", model_loaded=False, error=app.state.load_error)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            backend=det.backend,
            model=det.model_name,
            classes=list(det.class_names),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        file: Annotated[UploadFile, File(description="Ảnh JPEG/PNG/WebP/BMP")],
        conf: Annotated[float, Query(ge=0.0, le=1.0, description="Ngưỡng điểm tin cậy")] = 0.25,
        iou: Annotated[float, Query(gt=0.0, le=1.0, description="Ngưỡng IoU cho NMS")] = 0.7,
    ) -> PredictResponse:
        det: Detector | None = app.state.detector
        if det is None:
            raise HTTPException(503, f"Model chưa sẵn sàng: {app.state.load_error}")
        image = _decode_image(_read_limited(file, limit))
        try:
            result = det.predict(image, conf=conf, iou=iou)
        except Exception as e:  # noqa: BLE001
            log.exception("Lỗi inference")
            raise HTTPException(500, "Lỗi khi chạy model") from e
        return PredictResponse(**result.to_dict())

    return app


app = create_app()
