"""Train bằng Ultralytics từ file config YAML (configs/train_*.yaml)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import yaml

from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.env_info import env_info
from vehicledet.paths import REPO_ROOT, WEIGHTS_DIR, resolve_data_root

log = logging.getLogger(__name__)

_REQUIRED = ("model", "split", "name")


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
    """Ghi đè tham số train bằng 'key=value' (value parse theo YAML: 1, 0.5, true...)."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override phải có dạng key=value, nhận được: {item!r}")
        cfg["train"][key] = yaml.safe_load(value)
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


def run_train(config_path: str | Path, data_root: str | None = None, overrides=()) -> Path:
    from ultralytics import YOLO  # import trễ: chỉ cần khi thật sự train

    cfg = apply_overrides(load_config(config_path), list(overrides))
    dataset = build_yolo_dataset(resolve_data_root(data_root), cfg["split"])
    log.info("Dataset '%s': %s", cfg["split"], dataset.counts)

    args = build_train_args(cfg, dataset.data_yaml)
    model = YOLO(resolve_model(cfg["model"]))
    start = time.time()

    def save_run_info(trainer) -> None:
        info = {
            "config": cfg,
            "train_args": args,
            "dataset_counts": dataset.counts,
            "env": env_info(),
        }
        (Path(trainer.save_dir) / "vehicledet_run.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    model.add_callback("on_pretrain_routine_end", save_run_info)
    model.train(**args)
    save_dir = Path(model.trainer.save_dir)
    log.info("Xong sau %.1f phút. Kết quả: %s", (time.time() - start) / 60, save_dir)
    return save_dir
