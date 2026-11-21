import json

import pytest

from vehicledet.experiments import (
    MARK_END,
    MARK_START,
    best_epoch_from_results,
    load_rows,
    pick_best,
    render_table,
    replace_marked_block,
    run_info_path,
    weakest_class,
    write_experiment_report,
)

RESULT_HEADER = "                  epoch,       metrics/mAP50(B),    metrics/mAP50-95(B)\n"


def fake_result(map50=0.8, map5095=0.6, weak=0.3):
    return {
        "mAP50": map50,
        "mAP50-95": map5095,
        "precision": 0.7,
        "recall": 0.7,
        "per_class": {
            "a": {"precision": 1, "recall": 1, "AP50": 0.9, "AP50-95": 0.8},
            "b": {"precision": 1, "recall": 1, "AP50": 0.5, "AP50-95": weak},
        },
    }


def fake_info(name, model="yolo11s.pt", split="v2", imgsz=640, minutes=75.0):
    return {
        "name": name,
        "config": {"model": model, "split": split},
        "train_args": {"imgsz": imgsz, "epochs": 60},
        "train_minutes": minutes,
        "epochs_ran": 42,
        "mlflow_run_id": "abcdef1234567890",
    }


def test_best_epoch_uses_ultralytics_fitness(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text(RESULT_HEADER + "1,0.5,0.30\n2,0.6,0.40\n3,0.9,0.35\n4,0.6,0.39\n")
    # fitness = 0.1*mAP50 + 0.9*mAP50-95: epoch 2 = 0.42 > epoch 4 = 0.411 > epoch 3 = 0.405
    assert best_epoch_from_results(p) == (2, 4)


def test_weakest_class():
    assert weakest_class(fake_result(weak=0.3)["per_class"]) == ("b", 0.3)


def test_write_report_writes_run_info_last(tmp_path):
    d = write_experiment_report("exp", fake_info("exp"), fake_result(), None, None, tmp_path)
    assert (d / "val.json").exists() and (d / "val.csv").exists()
    assert run_info_path("exp", tmp_path).exists()


def test_load_rows_table_and_best(tmp_path):
    exp, test_path = tmp_path / "experiments", tmp_path / "test_results.json"
    write_experiment_report(
        "exp_a", fake_info("exp_a", "yolo11n.pt"), fake_result(0.7, 0.5), None, None, exp
    )
    write_experiment_report("exp_b", fake_info("exp_b"), fake_result(0.8, 0.6), None, None, exp)
    write_experiment_report(
        "exp_c", fake_info("exp_c", split="legacy"), fake_result(0.95, 0.9), None, None, exp
    )
    (exp / "exp_unfinished").mkdir()  # chưa có run_info -> bị bỏ qua
    rows = load_rows(exp, test_path)
    assert [r["name"] for r in rows] == ["exp_a", "exp_b", "exp_c"]

    assert pick_best(rows, "v2", 640)["name"] == "exp_b"  # exp_c là legacy nên không xét

    test_path.write_text(
        json.dumps(
            {**fake_result(0.75, 0.55), "weights": "<repo>/runs/detect/exp_b/weights/best.pt"}
        )
    )
    rows = load_rows(exp, test_path)
    assert rows[-1]["name"] == "TEST: exp_b" and rows[-1]["part"] == "test"
    assert rows[-1]["mAP50-95"] == 0.55 and rows[-1]["split"] == "v2 (test)"

    table = render_table(rows)
    assert (
        "| exp_b | yolo11s | v2 (val) | 640 | 0.800 | 0.600 | b (0.300) "
        "| 1 giờ 15 phút | 42 | `abcdef12` |" in table
    )
    assert table.count("\n") == 2 + len(rows)


def test_render_table_when_empty():
    assert "chưa chạy" in render_table([])


def test_replace_marked_block():
    text = f"trước\n{MARK_START}\ncũ\n{MARK_END}\nsau\n"
    out = replace_marked_block(text, "mới\n")
    assert out == f"trước\n{MARK_START}\nmới\n{MARK_END}\nsau\n"
    assert replace_marked_block(out, "mới\n") == out  # idempotent
    with pytest.raises(ValueError):
        replace_marked_block("không có marker", "x")
