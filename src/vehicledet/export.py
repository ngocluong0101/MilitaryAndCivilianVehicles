"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Export model đã train sang ONNX và TensorRT (FP16, INT8).

INT8 hiệu chuẩn bằng ~500 ảnh **train** ngẫu nhiên (seed cố định), không dùng val/test. Ultralytics
đọc ảnh hiệu chuẩn từ khoá `val`, nên sinh một data.yaml riêng trỏ `val` tới danh sách đó.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

import yaml

from vehicledet.data.splits import read_split
from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.paths import EXPORTS_DIR, SPLITS_DIR, class_names, portable, resolve_data_root

log = logging.getLogger(__name__)

ALL_FORMATS = ("onnx", "engine-fp16", "engine-int8")
CALIB_IMAGES = 500


def sample_calibration_images(train_files: list[str], n: int = CALIB_IMAGES, seed: int = 0):
    """Chọn n ảnh ngẫu nhiên (xác định theo seed) từ danh sách train."""
    files = sorted(train_files)
    return sorted(random.Random(seed).sample(files, min(n, len(files))))


def write_calibration_yaml(
    data_root: Path,
    split: str,
    out_dir: Path,
    n: int = CALIB_IMAGES,
    seed: int = 0,
    splits_dir: Path = SPLITS_DIR,
) -> Path:
    """Sinh data.yaml dùng cho calibration INT8: `val` = n ảnh lấy từ TẬP TRAIN của `split`."""
    train = read_split(split, splits_dir)["train"]
    chosen = sample_calibration_images(train, n, seed)
    yolo_train = (
        build_yolo_dataset(data_root, split, splits_dir).data_yaml.parent / "images" / "train"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    list_file = out_dir / "calibration_images.txt"
    list_file.write_text("".join(f"{yolo_train / name}\n" for name in chosen), encoding="utf-8")
    cfg = {
        "path": str(yolo_train.parent.parent),
        "train": str(list_file),
        "val": str(list_file),
        "nc": len(class_names()),
        "names": dict(enumerate(class_names())),
    }
    yaml_path = out_dir / "calibration.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def _tensorrt_version() -> str | None:
    try:
        import tensorrt

        return tensorrt.__version__
    except ImportError:
        return None


def export_formats(
    weights: str | Path,
    tag: str,
    split: str,
    formats: tuple[str, ...] = ALL_FORMATS,
    imgsz: int = 640,
    max_batch: int = 16,
    workspace_gb: float = 4.0,
    data_root: str | None = None,
    device: str = "0",
    out_root: Path = EXPORTS_DIR,
) -> dict:
    """Export vào exports/<tag>/: best.onnx, best_fp16.engine, best_int8.engine.

    Định dạng không export được thì ghi lý do (export_info.json) thay vì làm hỏng cả lệnh.
    """
    from ultralytics import YOLO

    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best_pt = out_dir / "best.pt"
    if Path(weights).resolve() != best_pt.resolve():
        shutil.copy(weights, best_pt)

    results: dict[str, dict] = {}
    # Export engine trước, ONNX sau cùng: export engine ghi đè best.onnx bằng bản của nó
    for fmt in sorted(formats, key=lambda f: f == "onnx"):
        if fmt not in ALL_FORMATS:
            raise ValueError(f"Định dạng không hợp lệ: {fmt}. Chọn trong {ALL_FORMATS}")
        try:
            if fmt == "onnx":
                # dynamic=True: batch (và kích thước) linh hoạt để benchmark batch 1/4/8/16
                path = YOLO(str(best_pt)).export(
                    format="onnx", imgsz=imgsz, dynamic=True, simplify=True, opset=17, device=device
                )
                target = out_dir / "best.onnx"
            else:
                if _tensorrt_version() is None:
                    raise ImportError("chưa cài tensorrt (xem docs/TENSORRT_SETUP.md)")
                kwargs = {"quantize": 16}
                if fmt == "engine-int8":
                    root = resolve_data_root(data_root)
                    calib = write_calibration_yaml(root, split, out_dir)
                    kwargs = {"quantize": 8, "data": str(calib)}
                path = YOLO(str(best_pt)).export(
                    format="engine",
                    imgsz=imgsz,
                    dynamic=True,
                    batch=max_batch,
                    workspace=workspace_gb,
                    device=device,
                    **kwargs,
                )
                target = out_dir / f"best_{fmt.split('-')[1]}.engine"
            Path(path).replace(target)
            results[fmt] = {
                "status": "ok",
                "path": portable(str(target)),
                "size_mb": round(target.stat().st_size / 2**20, 2),
            }
        except Exception as e:  # noqa: BLE001 - ghi lý do, tiếp tục định dạng khác
            log.warning("Export %s thất bại: %s", fmt, e)
            results[fmt] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"[:300]}

    info = {
        "weights": portable(str(weights)),
        "imgsz": imgsz,
        "max_batch": max_batch,
        "calibration": {"source": f"train split of '{split}'", "images": CALIB_IMAGES, "seed": 0},
        "tensorrt": _tensorrt_version(),
        "results": results,
    }
    (out_dir / "export_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return info
