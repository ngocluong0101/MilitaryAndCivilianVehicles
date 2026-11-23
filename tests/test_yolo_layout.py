"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

import pytest
import yaml

from vehicledet.data.splits import write_split_file
from vehicledet.data.yolo_layout import build_yolo_dataset


@pytest.fixture
def splits_dir(tmp_path):
    d = tmp_path / "splits"
    write_split_file(d / "s1" / "train.txt", ["a.jpg", "c.jpg"])
    write_split_file(d / "s1" / "val.txt", ["b.jpg"])
    write_split_file(d / "s1" / "test.txt", ["d.jpg"])
    return d


def test_build_creates_symlinks_and_yaml(fake_dataset, splits_dir):
    ds = build_yolo_dataset(fake_dataset, "s1", splits_dir)
    out = fake_dataset / "yolo" / "s1"

    assert ds.counts == {"train": 2, "val": 1, "test": 1}
    assert (out / "images" / "train" / "a.jpg").is_symlink()
    assert (out / "labels" / "train" / "a.txt").read_text().startswith("2 ")
    # c.jpg không có nhãn -> ghi nhận là thiếu, không tạo symlink nhãn
    assert ds.missing_labels["train"] == ["c.jpg"]
    assert not (out / "labels" / "train" / "c.txt").exists()

    cfg = yaml.safe_load(ds.data_yaml.read_text())
    assert cfg["path"] == str(out)
    assert cfg["train"] == "images/train" and cfg["val"] == "images/val"
    assert cfg["test"] == "images/test"
    assert cfg["nc"] == 6 and cfg["names"][2] == "military tank"


def test_build_is_idempotent_and_drops_stale_files(fake_dataset, splits_dir):
    build_yolo_dataset(fake_dataset, "s1", splits_dir)
    stale = fake_dataset / "yolo" / "s1" / "labels" / "train.cache"
    stale.write_text("old")
    build_yolo_dataset(fake_dataset, "s1", splits_dir)
    assert not stale.exists()


def test_build_fails_when_image_missing(fake_dataset, splits_dir):
    write_split_file(splits_dir / "s1" / "val.txt", ["zzz.jpg"])
    with pytest.raises(FileNotFoundError, match="zzz.jpg"):
        build_yolo_dataset(fake_dataset, "s1", splits_dir)
