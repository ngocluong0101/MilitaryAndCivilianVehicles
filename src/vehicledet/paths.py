"""Đường dẫn của project; dataset nằm ngoài code, trỏ bằng DATA_ROOT."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
SPLITS_DIR = REPO_ROOT / "splits"
REPORTS_DIR = REPO_ROOT / "reports"
WEIGHTS_DIR = REPO_ROOT / "weights"


def load_data_config() -> dict:
    """Đọc configs/data.yaml (tên lớp + cấu trúc thư mục dataset)."""
    with open(CONFIGS_DIR / "data.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_names() -> list[str]:
    names = load_data_config()["names"]
    return [names[i] for i in sorted(names)]


def resolve_data_root(arg: str | os.PathLike | None = None) -> Path:
    """Thứ tự ưu tiên: tham số > biến môi trường DATA_ROOT > <repo>/data.

    Kiểm tra thư mục ảnh tồn tại để lỗi sớm và dễ hiểu.
    """
    value = arg or os.environ.get("DATA_ROOT") or REPO_ROOT / "data"
    root = Path(value).expanduser().resolve()
    images = root / load_data_config()["images_dir"]
    if not images.is_dir():
        raise FileNotFoundError(
            f"Không thấy thư mục ảnh '{images}'. Đặt DATA_ROOT (hoặc --data-root) "
            "trỏ tới thư mục chứa Images/ và Labels/ của dataset."
        )
    return root


def images_dir(data_root: Path) -> Path:
    return data_root / load_data_config()["images_dir"]


def txt_labels_dir(data_root: Path) -> Path:
    return data_root / load_data_config()["txt_labels_dir"]


def csv_labels_dir(data_root: Path) -> Path:
    return data_root / load_data_config()["csv_labels_dir"]
