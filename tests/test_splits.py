"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

import pytest

from vehicledet.data.splits import read_split, write_split_file


def test_read_split_requires_train_and_val(tmp_path):
    write_split_file(tmp_path / "x" / "train.txt", ["a.jpg"])
    with pytest.raises(FileNotFoundError):
        read_split("x", tmp_path)


def test_write_and_read_split(tmp_path):
    write_split_file(tmp_path / "x" / "train.txt", ["b.jpg", "a.jpg"])
    write_split_file(tmp_path / "x" / "val.txt", ["c.jpg"])
    assert read_split("x", tmp_path) == {"train": ["a.jpg", "b.jpg"], "val": ["c.jpg"]}
