"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

import yaml

from vehicledet.data.splits import write_split_file
from vehicledet.export import sample_calibration_images, write_calibration_yaml


def test_sample_is_seeded_subset_without_duplicates():
    files = [f"img{i}.jpg" for i in range(1000)]
    a = sample_calibration_images(files, 500, seed=0)
    assert a == sample_calibration_images(
        list(reversed(files)), 500, seed=0
    )  # không phụ thuộc thứ tự
    assert a != sample_calibration_images(files, 500, seed=1)
    assert len(a) == len(set(a)) == 500 and set(a) <= set(files)


def test_sample_smaller_than_requested():
    assert len(sample_calibration_images(["a.jpg", "b.jpg"], 500)) == 2


def test_calibration_yaml_uses_only_train_images(fake_dataset, tmp_path):
    splits = tmp_path / "splits"
    write_split_file(splits / "s" / "train.txt", ["a.jpg", "b.jpg", "c.jpg"])
    write_split_file(splits / "s" / "val.txt", ["d.jpg"])
    write_split_file(splits / "s" / "test.txt", ["e.jpg"])

    path = write_calibration_yaml(
        fake_dataset, "s", tmp_path / "calib", n=2, seed=0, splits_dir=splits
    )
    cfg = yaml.safe_load(path.read_text())
    listed = [line.rsplit("/", 1)[1] for line in open(cfg["val"]).read().splitlines()]
    assert len(listed) == 2 and set(listed) <= {"a.jpg", "b.jpg", "c.jpg"}  # không có val/test
    assert cfg["train"] == cfg["val"] and cfg["nc"] == 6
    assert all("/images/train/" in line for line in open(cfg["val"]).read().splitlines())
