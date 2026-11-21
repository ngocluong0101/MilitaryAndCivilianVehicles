import pytest

from vehicledet.paths import class_names, resolve_data_root


def test_class_names_order():
    names = class_names()
    assert len(names) == 6
    assert names[0] == "military truck"
    assert names[5] == "military helicopter"


def test_resolve_prefers_argument_over_env(fake_dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "nope"))
    assert resolve_data_root(fake_dataset) == fake_dataset.resolve()


def test_resolve_uses_env(fake_dataset, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(fake_dataset))
    assert resolve_data_root() == fake_dataset.resolve()


def test_resolve_missing_images_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATA_ROOT"):
        resolve_data_root(tmp_path)
