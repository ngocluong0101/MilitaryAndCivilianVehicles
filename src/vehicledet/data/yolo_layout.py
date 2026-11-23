"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Dựng thư mục kiểu YOLO bằng symlink từ danh sách split (không copy ảnh).

Ultralytics tìm nhãn bằng cách đổi '/images/' thành '/labels/' trong đường dẫn ảnh, nên sinh:

    <DATA_ROOT>/yolo/<split>/images/<train|val|test>/<file>      -> Images/<file>
    <DATA_ROOT>/yolo/<split>/labels/<train|val|test>/<stem>.txt  -> nhãn TXT
    <DATA_ROOT>/yolo/<split>/data.yaml                           (đường dẫn tuyệt đối)
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from vehicledet.data.splits import read_split
from vehicledet.paths import SPLITS_DIR, class_names, images_dir, txt_labels_dir

log = logging.getLogger(__name__)


@dataclass
class YoloDataset:
    data_yaml: Path
    counts: dict[str, int]
    missing_labels: dict[str, list[str]]  # ảnh không có file nhãn TXT (coi như nền)


def build_yolo_dataset(
    data_root: Path, split_name: str, splits_dir: Path = SPLITS_DIR
) -> YoloDataset:
    splits = read_split(split_name, splits_dir)
    img_src, lbl_src = images_dir(data_root), txt_labels_dir(data_root)

    out = data_root / "yolo" / split_name
    if out.exists():
        shutil.rmtree(out)  # thư mục sinh tự động, chỉ chứa symlink + cache

    counts: dict[str, int] = {}
    missing_labels: dict[str, list[str]] = {}
    for part, filenames in splits.items():
        (out / "images" / part).mkdir(parents=True)
        (out / "labels" / part).mkdir(parents=True)
        missing_labels[part] = []
        for name in filenames:
            src_img = img_src / name
            if not src_img.is_file():
                raise FileNotFoundError(
                    f"Ảnh trong split '{split_name}/{part}' không tồn tại: {src_img}"
                )
            (out / "images" / part / name).symlink_to(src_img)
            src_lbl = lbl_src / (Path(name).stem + ".txt")
            if src_lbl.is_file():
                (out / "labels" / part / src_lbl.name).symlink_to(src_lbl)
            else:
                missing_labels[part].append(name)
        counts[part] = len(filenames)
        if missing_labels[part]:
            log.warning(
                "%s/%s: %d ảnh không có file nhãn", split_name, part, len(missing_labels[part])
            )

    cfg = {"path": str(out), "nc": len(class_names()), "names": dict(enumerate(class_names()))}
    for part in splits:
        cfg[part] = f"images/{part}"
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return YoloDataset(data_yaml, counts, missing_labels)
