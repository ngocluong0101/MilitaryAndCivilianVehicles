"""MLflow local (sqlite + thư mục mlruns/): log tham số, metric từng epoch, artifact nhỏ."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from vehicledet.paths import MLFLOW_DB, MLRUNS_DIR

log = logging.getLogger(__name__)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # bớt dòng gợi ý khi import mlflow

TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
EXPERIMENT_NAME = "vehicledet"
SMALL_ARTIFACT_SUFFIXES = {".png", ".jpg", ".csv", ".yaml", ".json"}


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Làm phẳng dict lồng nhau thành {'a.b': 'giá trị'} (MLflow param là chuỗi, <= 500 ký tự)."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = str(v)[:500]
    return out


def sanitize_metrics(metrics: dict) -> dict[str, float]:
    """MLflow không nhận dấu ngoặc trong tên metric; bỏ giá trị không phải số."""
    clean = {}
    for k, v in metrics.items():
        try:
            clean[re.sub(r"[()]", "", k)] = float(v)
        except (TypeError, ValueError):
            continue
    return clean


class Tracker:
    """Bao MLflow; khi tắt hoặc lỗi thì không bao giờ làm hỏng quá trình train."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.run_id: str | None = None
        self._mlflow = None

    def start(self, run_name: str, params: dict, tags: dict) -> None:
        if not self.enabled:
            return
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
            mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLRUNS_DIR.as_uri())
        mlflow.set_experiment(EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = run.info.run_id
        mlflow.log_params(flatten(params))
        log.info("MLflow run %s (%s)", self.run_id, TRACKING_URI)

    def _safe(self, fn, *a, **kw):
        if not (self.enabled and self._mlflow):
            return
        try:
            fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            log.warning("MLflow lỗi (bỏ qua, vẫn tiếp tục train): %s", e)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self._mlflow:
            self._safe(self._mlflow.log_metrics, sanitize_metrics(metrics), step=step)

    def log_artifacts(self, paths: list[Path], subdir: str | None = None) -> None:
        if self._mlflow:
            for p in paths:
                if p.is_file() and p.suffix in SMALL_ARTIFACT_SUFFIXES:
                    self._safe(self._mlflow.log_artifact, str(p), subdir)

    def epoch_callback(self):
        """Callback `on_fit_epoch_end` của Ultralytics: metric + lr mỗi epoch."""

        def on_fit_epoch_end(trainer) -> None:
            self.log_metrics(
                {**trainer.metrics, **getattr(trainer, "lr", {})}, step=trainer.epoch + 1
            )

        return on_fit_epoch_end

    def end(self, status: str = "FINISHED") -> None:
        if self._mlflow:
            self._safe(self._mlflow.end_run, status)
