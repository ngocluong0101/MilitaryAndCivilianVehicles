"""Lõi inference của demo: ONNX Runtime (mặc định, chạy cả CPU) hoặc TensorRT.

Backend ONNX tự làm tiền/hậu xử lý bằng numpy/Pillow (letterbox → model → lọc ngưỡng → NMS theo lớp
→ đưa toạ độ về ảnh gốc) để không cần torch/ultralytics khi triển khai.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

PAD_VALUE = 114  # màu nền letterbox, giống Ultralytics
MAX_DET = 300


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel của ảnh gốc)

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "class_id": self.class_id,
            "label": self.label,
            "score": round(self.score, 4),
            "bbox": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            },
        }


@dataclass
class DetectionResult:
    detections: list[Detection]
    image_size: tuple[int, int]  # (width, height) ảnh gốc
    inference_ms: float  # riêng phần chạy model
    total_ms: float  # tiền xử lý + model + hậu xử lý
    backend: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image": {"width": self.image_size[0], "height": self.image_size[1]},
            "detections": [d.to_dict() for d in self.detections],
            "inference_ms": round(self.inference_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "backend": self.backend,
        }


class Detector(Protocol):
    """Giao diện chung; demo chỉ phụ thuộc vào giao diện này (dễ thay bằng stub khi test)."""

    backend: str
    model_name: str
    class_names: list[str]

    def predict(
        self, image: Image.Image, conf: float = 0.25, iou: float = 0.7
    ) -> DetectionResult: ...


# ----- tiền/hậu xử lý (hàm thuần, kiểm thử được) -------------------------------------------


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize giữ tỉ lệ rồi đệm cho vuông `size`×`size`.

    img: HxWx3 uint8 (RGB). Trả về (tensor 1x3xSxS float32 trong [0,1], tỉ lệ, (pad_x, pad_y)).
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = max(1, round(w * r)), max(1, round(h * r))
    resized = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return tensor, r, (pad_x, pad_y)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """NMS tham lam. boxes: Nx4 (x1,y1,x2,y2). Trả về chỉ số giữ lại, theo điểm giảm dần."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode_output(
    output: np.ndarray,
    class_names: list[str],
    conf: float,
    iou: float,
    scale: float,
    pad: tuple[int, int],
    image_size: tuple[int, int],
    max_det: int = MAX_DET,
) -> list[Detection]:
    """Giải mã đầu ra YOLO (1, 4+nc, N): cx,cy,w,h trong không gian letterbox + điểm từng lớp."""
    pred = output[0].T  # (N, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(pred)), cls]
    mask = scores >= conf
    if not mask.any():
        return []
    pred, cls, scores = pred[mask], cls[mask], scores[mask]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    # NMS theo từng lớp: dịch box mỗi lớp ra xa nhau để chúng không triệt tiêu lẫn nhau
    keep = nms(boxes + cls[:, None] * 10_000.0, scores, iou)[:max_det]

    img_w, img_h = image_size
    out = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        x1, x2 = (np.clip([x1, x2], 0, None) - pad[0]) / scale
        y1, y2 = (np.clip([y1, y2], 0, None) - pad[1]) / scale
        box = (
            float(np.clip(x1, 0, img_w)),
            float(np.clip(y1, 0, img_h)),
            float(np.clip(x2, 0, img_w)),
            float(np.clip(y2, 0, img_h)),
        )
        c = int(cls[i])
        out.append(
            Detection(c, class_names[c] if c < len(class_names) else str(c), float(scores[i]), box)
        )
    return out


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Vẽ bbox + nhãn lên bản sao của ảnh (dùng chung cho demo/CLI)."""
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (131, 56, 236),
    ]
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(out.size) / 200))
    for d in detections:
        color = palette[d.class_id % len(palette)]
        draw.rectangle(d.box, outline=color, width=width)
        text = f"{d.label} {d.score:.2f}"
        tx, ty = d.box[0], max(0, d.box[1] - 12)
        left, top, right, bottom = draw.textbbox((tx, ty), text)
        draw.rectangle((left - 1, top - 1, right + 1, bottom + 1), fill=color)
        draw.text((tx, ty), text, fill="white")
    return out


# ----- backend ONNX Runtime -----------------------------------------------------------------


def _parse_metadata(meta: dict[str, str]) -> tuple[list[str], int | None]:
    names: list[str] = []
    if "names" in meta:
        parsed = ast.literal_eval(meta["names"])
        names = [parsed[i] for i in sorted(parsed)] if isinstance(parsed, dict) else list(parsed)
    imgsz = None
    if "imgsz" in meta:
        val = json.loads(meta["imgsz"].replace("(", "[").replace(")", "]"))
        imgsz = int(val[0] if isinstance(val, list) else val)
    return names, imgsz


class OnnxDetector:
    """Detector chạy bằng ONNX Runtime (CPU mặc định; dùng CUDA nếu có onnxruntime-gpu)."""

    def __init__(self, model_path: str | Path, device: str = "auto", class_names=None):
        import onnxruntime as ort

        available = ort.get_available_providers()
        want_cuda = device in ("auto", "cuda", "0") and "CUDAExecutionProvider" in available
        if device in ("cuda", "0") and not want_cuda:
            log.warning("Yêu cầu GPU nhưng onnxruntime không có CUDAExecutionProvider; dùng CPU")
        if want_cuda and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # nạp CUDA/cuDNN cài bằng pip (nvidia-*) trước khi tạo session
        providers = (["CUDAExecutionProvider"] if want_cuda else []) + ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.providers = self.session.get_providers()
        self.model_name = Path(model_path).name
        self.backend = f"onnxruntime ({self.providers[0].replace('ExecutionProvider', '')})"

        meta = self.session.get_modelmeta().custom_metadata_map
        if str(meta.get("end2end", "")).lower() == "true":
            raise ValueError("Model xuất dạng end2end (NMS trong graph) chưa được hỗ trợ")
        meta_names, meta_imgsz = _parse_metadata(meta)
        self.class_names = list(class_names or meta_names)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.imgsz = int(h) if isinstance(h, int) and h == w else (meta_imgsz or 640)

    def predict(self, image: Image.Image, conf: float = 0.25, iou: float = 0.7) -> DetectionResult:
        t0 = time.perf_counter()
        rgb = np.asarray(image.convert("RGB"))
        tensor, scale, pad = letterbox(rgb, self.imgsz)
        t1 = time.perf_counter()
        output = self.session.run(None, {self.input_name: tensor})[0]
        t2 = time.perf_counter()
        dets = decode_output(output, self.class_names, conf, iou, scale, pad, image.size)
        t3 = time.perf_counter()
        return DetectionResult(dets, image.size, (t2 - t1) * 1000, (t3 - t0) * 1000, self.backend)


class TensorRTDetector:
    """Detector chạy engine TensorRT qua Ultralytics (cần GPU, tensorrt, ultralytics)."""

    def __init__(self, model_path: str | Path, device: str = "0", class_names=None):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path), task="detect")
        self.device = "0" if device in ("auto", "cuda") else device
        self.model_name = Path(model_path).name
        self.backend = "tensorrt"
        names = self.model.names
        self.class_names = list(class_names or [names[i] for i in sorted(names)])

    def predict(self, image: Image.Image, conf: float = 0.25, iou: float = 0.7) -> DetectionResult:
        t0 = time.perf_counter()
        res = self.model.predict(
            np.asarray(image.convert("RGB"))[:, :, ::-1],
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )[0]
        total = (time.perf_counter() - t0) * 1000
        dets = [
            Detection(int(c), self.class_names[int(c)], float(s), tuple(float(v) for v in b))
            for b, s, c in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
                strict=True,
            )
        ]
        return DetectionResult(dets, image.size, float(res.speed["inference"]), total, self.backend)


def load_detector(model_path: str | Path, backend: str = "auto", device: str = "auto") -> Detector:
    """Tạo detector. backend: 'auto' (theo đuôi file: .onnx/.engine), 'onnx' hoặc 'tensorrt'."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy file model: {path}")
    if backend == "auto":
        backend = {".onnx": "onnx", ".engine": "tensorrt"}.get(path.suffix.lower(), "")
        if not backend:
            raise ValueError(f"Không đoán được backend từ đuôi file '{path.suffix}'")
    if backend == "onnx":
        return OnnxDetector(path, device)
    if backend == "tensorrt":
        return TensorRTDetector(path, "0" if device == "auto" else device)
    raise ValueError(f"backend không hợp lệ: {backend!r} (onnx | tensorrt | auto)")
