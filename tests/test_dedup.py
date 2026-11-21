import numpy as np
import pytest
from PIL import Image

from vehicledet.data import dedup


def noise_image(seed: int, size=(96, 96)) -> Image.Image:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)  # ảnh khối thô, 8x8
    return Image.fromarray(blocks).resize(size, Image.Resampling.NEAREST)


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "Images"
    d.mkdir()
    base = noise_image(1)
    base.save(d / "a.png")
    base.save(d / "a_copy.png")  # trùng byte
    base.resize((200, 200)).save(d / "a_big.jpg", quality=80)  # bản resize + nén lại
    noise_image(2).save(d / "b.png")
    noise_image(3).save(d / "c.png")
    return d


def test_hamming_pairs_and_grouping():
    hashes = ["0" * 16, "0" * 15 + "1", "f" * 16, "f" * 15 + "e"]  # 2 cặp, mỗi cặp cách 1 bit
    pairs = dedup.hamming_pairs(hashes, 64, 2)
    assert sorted((a, b, d) for a, b, d in pairs) == [(0, 1, 1), (2, 3, 1)]
    assert dedup.group_pairs(4, pairs, 2) == [[0, 1], [2, 3]]
    assert dedup.group_pairs(4, pairs, 0) == []


def test_group_pairs_is_transitive():
    pairs = [(0, 1, 5), (1, 2, 5), (3, 4, 50)]
    assert dedup.group_pairs(5, pairs, 10) == [[0, 1, 2]]


def test_hamming_pairs_handles_multiword_hashes():
    a = "0" * 64
    b = "0" * 63 + "1"  # khác ở từ 64-bit thấp nhất
    c = "1" + "0" * 63  # khác ở từ 64-bit cao nhất
    pairs = dedup.hamming_pairs([a, b, c], 256, 1)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (0, 2)]


def test_find_duplicate_groups(image_dir, tmp_path):
    names = sorted(p.name for p in image_dir.iterdir())
    hashes = dedup.compute_hashes(image_dir, names, tmp_path / "cache.json")
    res = dedup.find_duplicate_groups(hashes)
    assert [g["members"] for g in res["groups"]] == [["a.png", "a_big.jpg", "a_copy.png"]]
    assert res["summary"]["exact_duplicate_groups_md5"] == 1
    assert res["summary"]["images_not_in_any_group"] == 2


def test_compute_hashes_uses_cache_and_skips_unreadable(image_dir, tmp_path):
    (image_dir / "broken.jpg").write_bytes(b"nope")
    cache = tmp_path / "cache.json"
    first = dedup.compute_hashes(image_dir, ["a.png", "broken.jpg"], cache)
    assert set(first) == {"a.png"}
    second = dedup.compute_hashes(image_dir, ["a.png"], cache)
    assert second == first


def test_cross_split_leakage():
    groups = [["a", "b"], ["c", "d", "e"], ["f", "g"]]
    split_of = {
        "a": "train",
        "b": "val",
        "c": "train",
        "d": "train",
        "e": "test",
        "f": "val",
        "g": "val",
    }
    r = dedup.cross_split_leakage(groups, split_of)
    assert r["groups_spanning_splits"] == 2
    assert r["images_with_dup_in"]["train"] == {"val": 1, "test": 2}
    assert r["images_with_dup_in"]["val"] == {"train": 1}
    assert r["images_with_dup_in_same_split"] == {"train": 2, "val": 2}
