"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Train bằng Ultralytics từ file config YAML (configs/train_*.yaml).
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

import yaml

from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.env_info import env_info
from vehicledet.evaluate import evaluate, write_result
from vehicledet.paths import EXPERIMENTS_DIR, REPO_ROOT, WEIGHTS_DIR, portable, resolve_data_root

log = logging.getLogger(__name__)

_REQUIRED = ("model", "split", "name")
TOP_LEVEL_KEYS = ("model", "split", "name", "seed", "project")


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"Config {path} thiếu khoá: {missing}")
    cfg.setdefault("seed", 0)
    cfg.setdefault("project", "runs/detect")
    cfg.setdefault("train", {})
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Ghi đè bằng 'key=value' (YAML); model/split/name/seed/project nằm ở cấp ngoài."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override phải có dạng key=value, nhận được: {item!r}")
        target = cfg if key in TOP_LEVEL_KEYS else cfg["train"]
        target[key] = yaml.safe_load(value)
    return cfg


def resolve_model(model: str) -> str:
    """Tên weight trần (vd 'yolo11s.pt') -> weights/<tên>, tránh tải file về lẫn trong repo."""
    if Path(model).parent != Path("."):
        return model
    WEIGHTS_DIR.mkdir(exist_ok=True)
    return str(WEIGHTS_DIR / model)


def build_train_args(cfg: dict, data_yaml: Path) -> dict:
    project = Path(cfg["project"])
    if not project.is_absolute():
        project = REPO_ROOT / project
    return {
        **cfg["train"],
        "data": str(data_yaml),
        "seed": cfg["seed"],
        "project": str(project),
        "name": cfg["name"],
    }


def best_epoch_from_results(results_csv: Path) -> tuple[int, int]:
    """(epoch tốt nhất, số epoch đã chạy) từ results.csv; fitness = 0.1*mAP50 + 0.9*mAP50-95."""
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    fitness = [
        0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]) for r in rows
    ]
    best = max(range(len(rows)), key=fitness.__getitem__)
    return int(float(rows[best]["epoch"])), len(rows)


def write_experiment_report(
    name: str,
    run_info: dict,
    val_result: dict,
    results_csv: Path,
    confusion_png: Path,
    root: Path = EXPERIMENTS_DIR,
) -> Path:
    """Ghi val.json, results.csv, confusion_matrix_val.png, run_info.json vào <root>/<name>/."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    write_result(val_result, d / "val.json")
    if results_csv.exists():
        shutil.copy(results_csv, d / "results.csv")
    if confusion_png.exists():
        shutil.copy(confusion_png, d / "confusion_matrix_val.png")
    (d / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return d


def run_train(
    config_path: str | Path,
    data_root: str | None = None,
    overrides=(),
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> Path:
    from ultralytics import YOLO  # import trễ: chỉ cần khi thật sự train

    cfg = apply_overrides(load_config(config_path), list(overrides))
    name = cfg["name"]

    dataset = build_yolo_dataset(resolve_data_root(data_root), cfg["split"])
    log.info("Dataset '%s': %s", cfg["split"], dataset.counts)
    args = build_train_args(cfg, dataset.data_yaml)
    env = env_info()

    model = YOLO(resolve_model(cfg["model"]))

    def save_run_info(trainer) -> None:
        info = {"config": cfg, "train_args": args, "dataset_counts": dataset.counts, "env": env}
        (Path(trainer.save_dir) / "vehicledet_run.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    model.add_callback("on_pretrain_routine_end", save_run_info)

    start = time.time()
    model.train(**args)
    train_minutes = (time.time() - start) / 60
    save_dir = Path(model.trainer.save_dir)

    # Đánh giá model tốt nhất (best.pt) trên val; test KHÔNG đụng tới ở đây.
    best = save_dir / "weights" / "best.pt"
    plots_dir = save_dir / "val_eval"
    val_result, _ = evaluate(
        best,
        dataset.data_yaml,
        "val",
        imgsz=args.get("imgsz", 640),
        batch=args.get("batch", 16),
        device=args.get("device"),
        workers=args.get("workers", 4),
        plots_dir=plots_dir,
    )
    results_csv = save_dir / "results.csv"
    best_epoch, epochs_ran = best_epoch_from_results(results_csv)
    run_info = portable(
        {
            "name": name,
            "config": cfg,
            "train_args": args,
            "dataset_counts": dataset.counts,
            "env": env,
            "train_minutes": round(train_minutes, 1),
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "early_stopped": epochs_ran < args.get("epochs", epochs_ran),
            "weights": str(best),
        }
    )
    report_dir = write_experiment_report(
        name,
        run_info,
        val_result,
        results_csv,
        plots_dir / "confusion_matrix_normalized.png",
        experiments_dir,
    )

    log.info("Xong sau %.1f phút. Kết quả: %s ; báo cáo: %s", train_minutes, save_dir, report_dir)
    return save_dir
