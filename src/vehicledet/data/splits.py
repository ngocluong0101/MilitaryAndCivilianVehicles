"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Danh sách chia tập (splits/<tên>/{train,val,test}.txt, mỗi dòng một tên file ảnh).
"""

from __future__ import annotations

from pathlib import Path

from vehicledet.paths import SPLITS_DIR


def write_split_file(path: Path, filenames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n}\n" for n in sorted(filenames)), encoding="utf-8")


def read_split_file(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_split(name: str, splits_dir: Path = SPLITS_DIR) -> dict[str, list[str]]:
    """Đọc mọi file *.txt trong splits/<name>/ (train, val, test nếu có)."""
    d = splits_dir / name
    files = {p.stem: p for p in d.glob("*.txt") if p.stem in {"train", "val", "test"}}
    if "train" not in files or "val" not in files:
        raise FileNotFoundError(f"Thiếu train.txt/val.txt trong {d}")
    return {k: read_split_file(p) for k, p in files.items()}
