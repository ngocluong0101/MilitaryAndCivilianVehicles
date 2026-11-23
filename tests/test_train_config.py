"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

from pathlib import Path

import pytest

from vehicledet.paths import CONFIGS_DIR, REPO_ROOT
from vehicledet.train import apply_overrides, build_train_args, load_config, resolve_model


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("[et]*_*.yaml")), ids=lambda p: p.name)
def test_all_train_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg["split"] and cfg["model"].endswith(".pt")
    assert cfg["train"]["imgsz"] in (320, 640, 960)


def test_missing_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model: yolo11n.pt\n")
    with pytest.raises(ValueError, match="split"):
        load_config(p)


def test_overrides_parse_yaml_values():
    cfg = {"train": {"epochs": 100}}
    apply_overrides(cfg, ["epochs=1", "fraction=0.5", "plots=false"])
    assert cfg["train"] == {"epochs": 1, "fraction": 0.5, "plots": False}
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["oops"])


def test_build_train_args_is_absolute_and_seeded():
    cfg = load_config(CONFIGS_DIR / "exp3c_yolo11n_v2_640_30ep.yaml")
    args = build_train_args(cfg, Path("/x/data.yaml"))
    assert args["data"] == "/x/data.yaml"
    assert args["seed"] == 0
    assert args["project"] == str(REPO_ROOT / "runs/detect")


def test_resolve_model_puts_bare_names_in_weights_dir():
    assert resolve_model("yolo11s.pt").endswith("weights/yolo11s.pt")
    assert resolve_model("/abs/best.pt") == "/abs/best.pt"


def test_best_epoch_and_report(tmp_path):
    from vehicledet.train import best_epoch_from_results, write_experiment_report

    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch, metrics/mAP50(B), metrics/mAP50-95(B)\n1,0.5,0.30\n2,0.6,0.40\n3,0.6,0.35\n"
    )
    assert best_epoch_from_results(csv_path) == (2, 3)
    val = {
        "mAP50": 0.6,
        "mAP50-95": 0.4,
        "precision": 0.7,
        "recall": 0.5,
        "per_class": {"a": {"precision": 0.7, "recall": 0.5, "AP50": 0.6, "AP50-95": 0.4}},
    }
    d = write_experiment_report(
        "x", {"name": "x"}, val, csv_path, tmp_path / "no.png", root=tmp_path / "exp"
    )
    assert (d / "run_info.json").exists() and (d / "val.json").exists()
    assert (d / "results.csv").exists() and not (d / "confusion_matrix_val.png").exists()
