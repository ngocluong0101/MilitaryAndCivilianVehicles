"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

import pytest

from vehicledet.data.splits import (
    legacy_split,
    read_split,
    write_legacy_split,
    write_split_file,
)


def test_legacy_split_removes_overlap_and_missing(fake_dataset):
    r = legacy_split(fake_dataset)
    assert r.val == ["d.jpg", "e.jpg"]
    assert r.train == ["a.jpg", "b.jpg", "c.jpg"]  # d bị loại vì thuộc val
    assert r.overlap_removed == 1
    assert r.missing_images == ["g.jpg"]
    assert not set(r.train) & set(r.val)


def test_write_and_read_legacy_split(fake_dataset, tmp_path):
    splits_dir = tmp_path / "splits"
    write_legacy_split(fake_dataset, splits_dir)
    loaded = read_split("legacy", splits_dir)
    assert loaded["train"] == ["a.jpg", "b.jpg", "c.jpg"]
    assert loaded["val"] == ["d.jpg", "e.jpg"]
    assert (splits_dir / "legacy" / "missing_images.txt").read_text() == "g.jpg\n"


def test_read_split_requires_train_and_val(tmp_path):
    write_split_file(tmp_path / "x" / "train.txt", ["a.jpg"])
    with pytest.raises(FileNotFoundError):
        read_split("x", tmp_path)
