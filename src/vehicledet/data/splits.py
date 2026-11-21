"""Danh sách chia tập (splits/<tên>/{train,val,test}.txt, mỗi dòng một tên file ảnh)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from vehicledet.paths import SPLITS_DIR, csv_labels_dir, images_dir


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


def read_csv_filenames(csv_path: Path) -> set[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["filename"] for row in csv.DictReader(f)}


@dataclass
class LegacySplit:
    train: list[str]
    val: list[str]
    overlap_removed: int  # số ảnh của test_labels.csv bị lặp trong train_labels.csv
    missing_images: list[str] = field(default_factory=list)  # có trong CSV, không có trên đĩa


def legacy_split(data_root: Path) -> LegacySplit:
    """Tái tạo cách chia cũ: val = test_labels.csv; train = train_labels.csv trừ val.

    Chỉ giữ ảnh thực sự có trên đĩa. Chia theo tên file, chưa xét ảnh trùng nội dung
    (việc đó thuộc Phase 2).
    """
    csv_dir = csv_labels_dir(data_root)
    train_all = read_csv_filenames(csv_dir / "train_labels.csv")
    val_all = read_csv_filenames(csv_dir / "test_labels.csv")
    train_names = train_all - val_all
    on_disk = {p.name for p in images_dir(data_root).iterdir() if p.is_file()}
    missing = sorted((train_names | val_all) - on_disk)
    return LegacySplit(
        train=sorted(train_names & on_disk),
        val=sorted(val_all & on_disk),
        overlap_removed=len(train_all & val_all),
        missing_images=missing,
    )


def write_legacy_split(data_root: Path, splits_dir: Path = SPLITS_DIR) -> LegacySplit:
    result = legacy_split(data_root)
    out = splits_dir / "legacy"
    write_split_file(out / "train.txt", result.train)
    write_split_file(out / "val.txt", result.val)
    if result.missing_images:
        write_split_file(out / "missing_images.txt", result.missing_images)
    return result
