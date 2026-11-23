"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

import itertools
from collections import Counter

import pytest

from vehicledet.data.audit import ImageRecord
from vehicledet.data.labels import Box, LabelProblem
from vehicledet.data.make_split import (
    BACKGROUND,
    assign_splits,
    build_split,
    eligibility,
    main_class,
)

RATIOS = (0.8, 0.1, 0.1)


def rec(name, boxes=(), size=(100, 100), has_txt=True, problems=()):
    return ImageRecord(name, size, None if size else "err", has_txt, list(boxes), list(problems))


def b(cls, w=0.2):
    return Box(cls, 0.5, 0.5, w, w)


def test_main_class_rules():
    assert main_class([]) == BACKGROUND
    assert main_class([b(1), b(2), b(2)]) == 2  # nhiều bbox nhất
    assert main_class([b(1, 0.5), b(2, 0.1)]) == 1  # hoà số bbox -> diện tích lớn hơn
    assert main_class([b(3), b(1)]) == 1  # hoà hết -> id nhỏ hơn


def test_eligibility():
    assert eligibility(rec("a")) is None
    assert eligibility(rec("a", size=None)) == "unreadable_image"
    assert eligibility(rec("a", has_txt=False)) == "no_label_file"
    err = LabelProblem("a.txt", 1, "class_out_of_range", "")
    assert eligibility(rec("a", problems=[err])) == "label_error"
    warn = LabelProblem("a.txt", 1, "duplicate_box", "")
    assert eligibility(rec("a", boxes=[b(0)], problems=[warn])) is None


def make_data(n_per_class=300, n_classes=3, n_background=30, group_size=3):
    """Dữ liệu giả: mỗi lớp là các nhóm gần trùng cỡ group_size + vài ảnh đơn."""
    strata, groups = {}, []
    for c in range(n_classes):
        names = [f"c{c}_{i}.jpg" for i in range(n_per_class)]
        strata.update({n: c for n in names})
        for i in range(0, n_per_class - group_size, group_size * 2):
            groups.append(names[i : i + group_size])
    strata.update({f"bg_{i}.jpg": BACKGROUND for i in range(n_background)})
    return strata, groups


def test_no_group_spans_splits_and_every_image_assigned():
    strata, groups = make_data()
    result = assign_splits(strata, groups, RATIOS, seed=0)
    assert set(result) == set(strata)
    for g in groups:
        assert len({result[m] for m in g}) == 1


def test_ratios_and_stratification_are_respected():
    strata, groups = make_data()
    result = assign_splits(strata, groups, RATIOS, seed=0)
    total = len(strata)
    counts = Counter(result.values())
    for split, ratio in zip(("train", "val", "test"), RATIOS, strict=True):
        assert abs(counts[split] / total - ratio) < 0.02
    for cls in {*range(3), BACKGROUND}:
        in_cls = [n for n, s in strata.items() if s == cls]
        val_share = sum(result[n] == "val" for n in in_cls) / len(in_cls)
        assert abs(val_share - 0.1) < 0.06  # mỗi lớp (kể cả ảnh nền) đều có mặt trong val


def test_deterministic_and_seed_sensitive():
    strata, groups = make_data()
    a = assign_splits(strata, groups, RATIOS, seed=7)
    assert a == assign_splits(strata, groups, RATIOS, seed=7)
    assert a != assign_splits(strata, groups, RATIOS, seed=8)


def test_input_order_does_not_matter():
    strata, groups = make_data(n_per_class=60)
    shuffled = dict(reversed(list(strata.items())))
    assert assign_splits(strata, groups, RATIOS, 3) == assign_splits(
        shuffled, list(reversed(groups)), RATIOS, 3
    )


def test_bad_ratios_rejected():
    with pytest.raises(ValueError):
        assign_splits({"a": 0}, [], (0.5, 0.5, 0.5), 0)


def test_build_split_excludes_bad_images_and_filters_groups():
    err = LabelProblem("x.txt", 1, "zero_size", "")
    records = {f"i{i}.jpg": rec(f"i{i}.jpg", [b(i % 2)]) for i in range(40)}
    records["bad.jpg"] = rec("bad.jpg", problems=[err])
    records["gone.jpg"] = rec("gone.jpg", size=None)
    groups = [["i0.jpg", "bad.jpg"], ["i1.jpg", "i2.jpg", "gone.jpg"]]
    assignment, excluded = build_split(records, groups, RATIOS, 0)
    assert excluded == {"bad.jpg": "label_error", "gone.jpg": "unreadable_image"}
    assert "bad.jpg" not in assignment and len(assignment) == 40
    assert assignment["i1.jpg"] == assignment["i2.jpg"]


def test_pairs_of_splits_are_disjoint():
    strata, groups = make_data()
    result = assign_splits(strata, groups, RATIOS, 0)
    by = {s: {n for n, v in result.items() if v == s} for s in ("train", "val", "test")}
    for x, y in itertools.combinations(by, 2):
        assert not by[x] & by[y]
