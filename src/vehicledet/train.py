"""Train bằng Ultralytics từ file config YAML (configs/train_*.yaml)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import yaml

from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.env_info import env_info
from vehicledet.paths import EXPERIMENTS_DIR, REPO_ROOT, WEIGHTS_DIR, portable, resolve_data_root
from vehicledet.tracking import Tracker

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
    """Ghi đè bằng 'key=value' (value parse theo YAML). Khoá model/split/name/seed/project ghi vào
    cấp ngoài cùng, các khoá còn lại là tham số train của Ultralytics."""
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


def _drop_builtin_mlflow(trainer) -> None:
    """Gỡ callback MLflow tích hợp sẵn của Ultralytics (nó log cả weights và đặt tên experiment
    theo đường dẫn); ta dùng Tracker riêng. Callback này được Ultralytics thêm lúc khởi tạo trainer,
    nên phải gỡ ở sự kiện đầu tiên của trainer (on_pretrain_routine_start). Chỉ ảnh hưởng run này,
    không đổi cấu hình toàn cục của Ultralytics."""
    from ultralytics.utils.callbacks import mlflow as ul_mlflow

    for event, fn in ul_mlflow.callbacks.items():
        if fn in trainer.callbacks.get(event, []):
            trainer.callbacks[event].remove(fn)


def run_train(
    config_path: str | Path,
    data_root: str | None = None,
    overrides=(),
    use_mlflow: bool = True,
    skip_if_done: bool = False,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> Path | None:
    from ultralytics import YOLO  # import trễ: chỉ cần khi thật sự train

    from vehicledet.evaluate import evaluate
    from vehicledet.experiments import (
        best_epoch_from_results,
        run_info_path,
        write_experiment_report,
    )

    cfg = apply_overrides(load_config(config_path), list(overrides))
    name = cfg["name"]
    if skip_if_done and run_info_path(name, experiments_dir).exists():
        log.info("Bỏ qua '%s': đã có kết quả ở %s", name, run_info_path(name, experiments_dir))
        return None

    dataset = build_yolo_dataset(resolve_data_root(data_root), cfg["split"])
    log.info("Dataset '%s': %s", cfg["split"], dataset.counts)
    args = build_train_args(cfg, dataset.data_yaml)
    env = env_info()

    tracker = Tracker(use_mlflow)
    tracker.start(
        run_name=name,
        params={"config": cfg, "dataset_counts": dataset.counts, "env": env},
        tags={"split": cfg["split"], "model": cfg["model"], "git_commit": str(env["git_commit"])},
    )

    model = YOLO(resolve_model(cfg["model"]))
    model.add_callback("on_pretrain_routine_start", _drop_builtin_mlflow)
    if tracker.enabled:
        model.add_callback("on_fit_epoch_end", tracker.epoch_callback())

    def save_run_info(trainer) -> None:
        info = {"config": cfg, "train_args": args, "dataset_counts": dataset.counts, "env": env}
        (Path(trainer.save_dir) / "vehicledet_run.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    model.add_callback("on_pretrain_routine_end", save_run_info)

    start = time.time()
    try:
        model.train(**args)
    except BaseException:
        tracker.end("FAILED")
        raise
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
            "mlflow_run_id": tracker.run_id,
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

    final = {
        "val_mAP50": val_result["mAP50"],
        "val_mAP50-95": val_result["mAP50-95"],
        "val_precision": val_result["precision"],
        "val_recall": val_result["recall"],
        "train_minutes": train_minutes,
        "best_epoch": best_epoch,
    }
    for cls, v in val_result["per_class"].items():
        key = cls.replace(" ", "_")
        final[f"val_AP50_{key}"], final[f"val_AP50-95_{key}"] = v["AP50"], v["AP50-95"]
    tracker.log_metrics(final)
    tracker.log_artifacts(
        [
            *report_dir.iterdir(),
            *save_dir.glob("*.png"),
            save_dir / "args.yaml",
            save_dir / "vehicledet_run.json",
        ]
    )
    tracker.end("FINISHED")
    log.info("Xong sau %.1f phút. Kết quả: %s ; báo cáo: %s", train_minutes, save_dir, report_dir)
    return save_dir
