from vehicledet import tracking
from vehicledet.paths import REPO_ROOT, portable
from vehicledet.train import apply_overrides


def test_flatten_nested_and_truncates():
    flat = tracking.flatten({"a": {"b": 1, "c": {"d": "x"}}, "e": "y" * 1000})
    assert flat["a.b"] == "1" and flat["a.c.d"] == "x"
    assert len(flat["e"]) == 500


def test_sanitize_metrics_drops_parens_and_non_numbers():
    clean = tracking.sanitize_metrics({"metrics/mAP50(B)": 0.5, "name": "x", "lr/pg0": "0.01"})
    assert clean == {"metrics/mAP50B": 0.5, "lr/pg0": 0.01}


def test_disabled_tracker_is_noop():
    t = tracking.Tracker(enabled=False)
    t.start("run", {"a": 1}, {})
    t.log_metrics({"x": 1})
    t.end()
    assert t.run_id is None


def test_tracker_logs_to_local_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    monkeypatch.setattr(tracking, "MLRUNS_DIR", tmp_path / "mlruns")
    art = tmp_path / "cm.png"
    art.write_bytes(b"png")
    t = tracking.Tracker()
    t.start("run1", {"cfg": {"seed": 0}}, {"split": "v2"})
    t.log_metrics({"val_mAP50": 0.5}, step=1)
    t.log_artifacts([art, tmp_path / "missing.png"])
    t.end()

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking.TRACKING_URI)
    run = client.get_run(t.run_id)
    assert run.data.params["cfg.seed"] == "0" and run.data.tags["split"] == "v2"
    assert run.data.metrics["val_mAP50"] == 0.5
    assert [a.path for a in client.list_artifacts(t.run_id)] == ["cm.png"]
    assert run.info.status == "FINISHED"


def test_overrides_route_top_level_keys():
    cfg = {"model": "a.pt", "split": "v2", "name": "n", "train": {"epochs": 60}}
    apply_overrides(cfg, ["model=yolo11n.pt", "name=x", "imgsz=960", "epochs=5"])
    assert cfg["model"] == "yolo11n.pt" and cfg["name"] == "x"
    assert cfg["train"] == {"epochs": 5, "imgsz": 960}


def test_portable_hides_machine_paths():
    value = {"p": f"{REPO_ROOT}/runs/x", "l": [f"{REPO_ROOT}/a", 3], "n": 1}
    assert portable(value) == {"p": "<repo>/runs/x", "l": ["<repo>/a", 3], "n": 1}
    assert "ngocluong" not in portable(str(REPO_ROOT / "data"))
