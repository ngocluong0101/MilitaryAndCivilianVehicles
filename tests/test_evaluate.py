import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vehicledet.evaluate import (
    check_test_allowed,
    confusion_to_dict,
    metrics_to_dict,
    write_result,
)

NAMES = ["a", "b", "c"]


def fake_metrics(present=(0, 1, 2)):
    ap = {0: (0.9, 0.8, 0.85, 0.6), 1: (0.7, 0.6, 0.65, 0.4), 2: (0.5, 0.4, 0.45, 0.2)}
    box = SimpleNamespace(
        map50=0.65,
        map=0.4,
        mp=0.7,
        mr=0.6,
        class_result=lambda k: ap[list(present)[k]],
    )
    cm = SimpleNamespace(matrix=np.array([[8, 1, 0, 0], [2, 9, 0, 0], [0, 0, 5, 0], [1, 0, 1, 0]]))
    return SimpleNamespace(
        box=box,
        ap_class_index=np.array(present),
        speed={"preprocess": 0.4, "inference": 3.2},
        confusion_matrix=cm,
    )


def test_metrics_to_dict_per_class_and_overall():
    r = metrics_to_dict(fake_metrics(), NAMES)
    assert r["mAP50"] == 0.65 and r["mAP50-95"] == 0.4
    assert r["per_class"]["a"] == {"precision": 0.9, "recall": 0.8, "AP50": 0.85, "AP50-95": 0.6}
    assert r["speed_ms_per_image"]["inference"] == 3.2
    assert r["classes_absent_in_split"] == []
    json.dumps(r)  # phải JSON-serializable (không còn numpy)


def test_metrics_to_dict_marks_absent_classes():
    r = metrics_to_dict(fake_metrics(present=(0, 2)), NAMES)
    assert list(r["per_class"]) == ["a", "c"]
    assert r["classes_absent_in_split"] == ["b"]


def test_confusion_normalized_by_true_columns():
    cm = confusion_to_dict(np.array([[8, 1, 0], [2, 9, 0], [0, 0, 0]]), ["a", "b"])
    assert cm["labels"] == ["a", "b", "background"]
    assert cm["counts"][0] == [8, 1, 0]
    assert cm["normalized_by_true"][0][0] == pytest.approx(0.8)  # 8 / (8 + 2)
    assert cm["normalized_by_true"][1][1] == pytest.approx(0.9)
    assert cm["normalized_by_true"][0][2] == 0  # cột không có mẫu -> 0, không chia cho 0


def test_write_result_json_and_csv(tmp_path):
    result = metrics_to_dict(fake_metrics(), NAMES)
    write_result(result, tmp_path / "out" / "r.json")
    assert json.loads((tmp_path / "out" / "r.json").read_text())["mAP50"] == 0.65
    rows = list(csv.DictReader(open(tmp_path / "out" / "r.csv")))
    assert [r["class"] for r in rows] == ["a", "b", "c", "ALL"]
    assert float(rows[-1]["AP50-95"]) == 0.4


def test_val_is_always_allowed(tmp_path):
    check_test_allowed("val", final=False, allow_rerun=False, path=tmp_path / "t.json")


def test_test_requires_final_flag(tmp_path):
    with pytest.raises(PermissionError, match="--final"):
        check_test_allowed("test", final=False, allow_rerun=False, path=tmp_path / "t.json")


def test_test_is_run_only_once(tmp_path):
    path = tmp_path / "t.json"
    check_test_allowed("test", final=True, allow_rerun=False, path=path)  # lần đầu: được
    path.write_text("{}")
    with pytest.raises(PermissionError, match="một lần"):
        check_test_allowed("test", final=True, allow_rerun=False, path=path)
    check_test_allowed("test", final=True, allow_rerun=True, path=path)
