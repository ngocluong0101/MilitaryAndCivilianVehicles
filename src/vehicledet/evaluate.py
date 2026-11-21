"""Đánh giá một weights trên một split bất kỳ; xuất json + csv.

Tập test chỉ dùng được với --final và chỉ ghi vào reports/test_results.json (một lần).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from vehicledet.data.splits import read_split
from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.paths import REPORTS_DIR, class_names, portable, resolve_data_root

log = logging.getLogger(__name__)

TEST_RESULTS_PATH = REPORTS_DIR / "test_results.json"


def confusion_to_dict(matrix: np.ndarray, names: list[str]) -> dict:
    """Ma trận nhầm lẫn của Ultralytics: hàng = dự đoán, cột = nhãn thật, hàng/cột cuối = nền."""
    m = np.asarray(matrix, dtype=float)
    col_sum = m.sum(axis=0, keepdims=True)
    norm = np.divide(m, col_sum, out=np.zeros_like(m), where=col_sum > 0)
    return {
        "labels": [*names, "background"],
        "convention": "rows=predicted, cols=true; last row/col = background",
        "counts": m.astype(int).tolist(),
        "normalized_by_true": np.round(norm, 4).tolist(),
    }


def metrics_to_dict(m, names: list[str]) -> dict:
    """Chuyển đối tượng DetMetrics của Ultralytics thành dict thuần (JSON được)."""
    per_class: dict[str, dict] = {}
    present = [int(i) for i in m.ap_class_index]
    for k, ci in enumerate(present):
        p, r, ap50, ap = m.box.class_result(k)
        per_class[names[ci]] = {
            "precision": float(p),
            "recall": float(r),
            "AP50": float(ap50),
            "AP50-95": float(ap),
        }
    out = {
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": per_class,
        "classes_absent_in_split": [n for i, n in enumerate(names) if i not in present],
        "speed_ms_per_image": {k: round(float(v), 3) for k, v in dict(m.speed).items()},
        "speed_note": "đo trong lúc val (batch, gồm tiền/hậu xử lý); benchmark chuẩn ở bước export",
    }
    cm = getattr(m, "confusion_matrix", None)
    if cm is not None:
        out["confusion_matrix"] = confusion_to_dict(cm.matrix, names)
    return out


def per_class_rows(result: dict) -> list[dict]:
    return [{"class": c, **v} for c, v in result["per_class"].items()]


def write_result(result: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = per_class_rows(result)
    with open(json_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "AP50", "AP50-95"])
        w.writeheader()
        w.writerows(rows)
        w.writerow(
            {
                "class": "ALL",
                "precision": result["precision"],
                "recall": result["recall"],
                "AP50": result["mAP50"],
                "AP50-95": result["mAP50-95"],
            }
        )


def check_test_allowed(part: str, final: bool, allow_rerun: bool, path: Path = TEST_RESULTS_PATH):
    """Bảo vệ tập test: chỉ chạy khi có --final, và không ghi đè kết quả cũ."""
    if part != "test":
        return
    if not final:
        raise PermissionError(
            "Tập test chỉ dùng để báo cáo kết quả cuối: thêm --final khi đã chốt model bằng val."
        )
    if path.exists() and not allow_rerun:
        raise PermissionError(
            f"{path} đã tồn tại: test chỉ được chạy một lần cho model đã chọn "
            "(thêm --allow-rerun nếu bạn chắc chắn)."
        )


def evaluate(
    weights: str | Path,
    data_yaml: Path,
    part: str,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    workers: int = 4,
    conf: float = 0.001,
    iou: float = 0.7,
    plots_dir: Path | None = None,
) -> tuple[dict, Path | None]:
    """Chạy Ultralytics val trên `part` ('val' | 'test'). Trả về (dict kết quả, thư mục plot)."""
    from ultralytics import YOLO

    kwargs = dict(
        data=str(data_yaml),
        split=part,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        conf=conf,
        iou=iou,
        plots=plots_dir is not None,
        verbose=False,
    )
    if device:
        kwargs["device"] = device
    if plots_dir is not None:
        kwargs.update(project=str(plots_dir.parent), name=plots_dir.name, exist_ok=True)
    m = YOLO(str(weights)).val(**kwargs)
    result = metrics_to_dict(m, class_names())
    result["eval_settings"] = {
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "iou": iou,
        "part": part,
    }
    return result, Path(m.save_dir) if plots_dir is not None else None


def run_evaluate(
    weights: str,
    split: str,
    part: str = "val",
    data_root: str | None = None,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    workers: int = 4,
    tag: str | None = None,
    final: bool = False,
    allow_rerun: bool = False,
) -> Path:
    check_test_allowed(part, final, allow_rerun)
    root = resolve_data_root(data_root)
    dataset = build_yolo_dataset(root, split)
    weights_path = Path(weights)
    tag = tag or f"{weights_path.parent.parent.name}_{split}_{part}"
    out = TEST_RESULTS_PATH if part == "test" else REPORTS_DIR / "eval" / f"{tag}.json"
    plots_dir = REPORTS_DIR / "eval" / f"{tag}_plots"

    result, saved = evaluate(
        weights, dataset.data_yaml, part, imgsz, batch, device, plots_dir=plots_dir
    )
    result["weights"] = portable(str(weights))
    result["split"] = split
    result["part"] = part
    result["n_images"] = len(read_split(split)[part])
    write_result(result, out)
    # Giữ lại duy nhất confusion matrix chuẩn hoá (png nhỏ); bỏ phần plot còn lại cho gọn
    keep = saved / "confusion_matrix_normalized.png" if saved else None
    if keep and keep.exists():
        keep.replace(out.with_suffix(".confusion.png"))
    if saved:
        import shutil

        shutil.rmtree(saved, ignore_errors=True)
    log.info("mAP50=%.4f mAP50-95=%.4f -> %s", result["mAP50"], result["mAP50-95"], out)
    return out
