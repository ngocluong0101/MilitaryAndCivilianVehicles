import pytest

from vehicledet.cli import build_parser, main


def test_parser_requires_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_train_args():
    a = build_parser().parse_args(["train", "--config", "c.yaml", "--override", "epochs=1"])
    assert a.command == "train" and a.override == ["epochs=1"]


def test_parser_data_commands():
    pr = build_parser()
    assert pr.parse_args(["audit"]).command == "audit"
    assert pr.parse_args(["find-duplicates"]).threshold == 56
    a = pr.parse_args(
        ["make-split", "--name", "v3", "--seed", "5", "--ratios", "0.7", "0.2", "0.1"]
    )
    assert (a.name, a.seed, a.ratios) == ("v3", 5, [0.7, 0.2, 0.1])


def test_parser_evaluate_and_train_flags():
    pr = build_parser()
    a = pr.parse_args(
        ["evaluate", "--weights", "w.pt", "--split", "v2", "--part", "test", "--final"]
    )
    assert (a.part, a.final, a.allow_rerun, a.workers) == ("test", True, False, 4)
    t = pr.parse_args(["train", "--config", "c.yaml", "--no-mlflow", "--skip-if-done"])
    assert t.no_mlflow and t.skip_if_done
    with pytest.raises(SystemExit):
        pr.parse_args(["evaluate", "--weights", "w.pt", "--split", "v2", "--part", "other"])


def test_make_legacy_split_end_to_end(fake_dataset, tmp_path, monkeypatch, capsys):
    # default SPLITS_DIR được gán lúc định nghĩa hàm -> ép ghi vào tmp_path
    from vehicledet.data import splits

    monkeypatch.setattr(
        "vehicledet.cli.write_legacy_split",
        lambda root: splits.write_legacy_split(root, tmp_path / "splits"),
    )
    assert main(["make-legacy-split", "--data-root", str(fake_dataset)]) == 0
    out = capsys.readouterr().out
    assert "train=3 val=2" in out and "g.jpg" in out
    assert (tmp_path / "splits" / "legacy" / "train.txt").exists()
