"""Phát hiện ảnh trùng / gần trùng bằng perceptual hash (pHash) và gom nhóm.

- Hai ảnh là "gần trùng" nếu pHash 16x16 (256 bit) cách nhau <= THRESHOLD, HOẶC pHash 8x8
  (64 bit) cách nhau <= COARSE_THRESHOLD (bắt thêm ảnh bị zoom/crop). Nhóm là bao đóng bắc
  cầu (union-find).
- dHash 8x8 là hash độc lập, chỉ dùng để kiểm tra chéo kết quả chia tập.
- md5 của file để đếm ảnh trùng byte-by-byte.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

HASH_SIZE = 16
BITS = HASH_SIZE * HASH_SIZE
THRESHOLD = 56  # Hamming trên 256 bit (~22%); xem bảng độ nhạy trong reports/duplicates.md
SENSITIVITY = (0, 8, 16, 24, 32, 40, 48, 56, 64, 72)
COARSE_HASH_SIZE = 8
COARSE_THRESHOLD = 4  # Hamming trên pHash 8x8 (64 bit)
CHECK_THRESHOLD = 4  # Hamming trên dHash 8x8, chỉ để kiểm tra chéo
CACHE_VERSION = 2


def _hash_int(im: Image.Image, kind: str, hash_size: int) -> int:
    import imagehash

    fn = {"phash": imagehash.phash, "dhash": imagehash.dhash}[kind]
    return int(str(fn(im, hash_size=hash_size)), 16)


def compute_hashes(
    image_dir: Path, names: list[str], cache_path: Path | None = None
) -> dict[str, dict]:
    """Trả về {tên ảnh: {md5, phash, phash8, dhash8}}; ảnh không đọc được bị bỏ qua (ghi log).

    Kết quả được cache theo (tên, kích thước file) để chạy lại nhanh.
    """
    cache: dict = {}
    if cache_path and cache_path.is_file():
        loaded = json.loads(cache_path.read_text())
        if loaded.get("version") == CACHE_VERSION:
            cache = loaded["items"]

    out: dict[str, dict] = {}
    for i, name in enumerate(names, 1):
        path = image_dir / name
        size = path.stat().st_size
        hit = cache.get(name)
        if hit and hit["file_size"] == size:
            out[name] = hit
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                item = {
                    "file_size": size,
                    "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                    "phash": f"{_hash_int(im, 'phash', HASH_SIZE):0{BITS // 4}x}",
                    "phash8": f"{_hash_int(im, 'phash', COARSE_HASH_SIZE):016x}",
                    "dhash8": f"{_hash_int(im, 'dhash', 8):016x}",
                }
        except Exception as e:  # noqa: BLE001
            log.warning("bỏ qua ảnh không đọc được %s: %s", name, e)
            continue
        out[name] = item
        if i % 1000 == 0:
            log.info("đã hash %d/%d ảnh", i, len(names))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"version": CACHE_VERSION, "items": out}))
    return out


def _to_words(hex_hashes: list[str], bits: int) -> np.ndarray:
    """Chuỗi hex -> mảng (n, bits/64) uint64."""
    n_words = bits // 64
    arr = np.zeros((len(hex_hashes), n_words), dtype=np.uint64)
    for i, h in enumerate(hex_hashes):
        v = int(h, 16)
        for k in range(n_words):
            arr[i, k] = (v >> (64 * k)) & 0xFFFFFFFFFFFFFFFF
    return arr


def hamming_pairs(hex_hashes: list[str], bits: int, max_dist: int) -> list[tuple[int, int, int]]:
    """Mọi cặp (i, j, khoảng cách) với i < j và khoảng cách Hamming <= max_dist."""
    arr = _to_words(hex_hashes, bits)
    pairs: list[tuple[int, int, int]] = []
    for i in range(len(arr) - 1):
        d = np.bitwise_count(arr[i] ^ arr[i + 1 :]).sum(axis=1)
        for j in np.flatnonzero(d <= max_dist):
            pairs.append((i, i + 1 + int(j), int(d[j])))
    return pairs


def group_pairs(n: int, pairs: list[tuple[int, int, int]], threshold: int) -> list[list[int]]:
    """Union-find trên các cặp có khoảng cách <= threshold; chỉ trả nhóm có >= 2 ảnh."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, d in pairs:
        if d <= threshold:
            parent[find(a)] = find(b)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def find_duplicate_groups(hashes: dict[str, dict], threshold: int = THRESHOLD) -> dict:
    """Gom nhóm gần trùng. Trả về dict sẵn sàng ghi JSON (nhóm theo tên file, đã sắp xếp)."""
    names = sorted(hashes)
    pairs = hamming_pairs([hashes[n]["phash"] for n in names], BITS, max(SENSITIVITY))
    coarse = hamming_pairs([hashes[n]["phash8"] for n in names], 64, COARSE_THRESHOLD)
    coarse_edges = [(a, b, 0) for a, b, _ in coarse]

    def summarize(g: list[list[int]]) -> dict:
        sizes = sorted((len(x) for x in g), reverse=True)
        return {"groups": len(g), "images_in_groups": sum(sizes), "largest_group": sizes[0]}

    sensitivity = {
        str(t): summarize(group_pairs(len(names), pairs, t) or [[0]]) for t in SENSITIVITY
    }
    groups = group_pairs(len(names), pairs + coarse_edges, threshold)
    out_groups = [{"members": [names[i] for i in g]} for g in groups]
    only_16 = group_pairs(len(names), pairs, threshold)

    md5_count: dict[str, int] = defaultdict(int)
    for n in names:
        md5_count[hashes[n]["md5"]] += 1
    exact = [c for c in md5_count.values() if c > 1]
    in_groups = sum(len(g["members"]) for g in out_groups)
    return {
        "params": {
            "hash": "phash",
            "hash_size": HASH_SIZE,
            "bits": BITS,
            "threshold": threshold,
            "coarse_phash8_threshold": COARSE_THRESHOLD,
        },
        "summary": {
            "images_hashed": len(names),
            "near_duplicate_groups": len(out_groups),
            "images_in_groups": in_groups,
            "images_not_in_any_group": len(names) - in_groups,
            "approx_distinct_pictures": len(names) - in_groups + len(out_groups),
            "largest_group": max((len(g["members"]) for g in out_groups), default=0),
            "extra_images_from_coarse_phash8": in_groups - sum(len(g) for g in only_16),
            "exact_duplicate_groups_md5": len(exact),
            "images_in_exact_duplicate_groups": sum(exact),
            "sensitivity_phash16_only": sensitivity,
        },
        "groups": out_groups,
    }


def cross_split_leakage(groups: list[list[str]], split_of: dict[str, str]) -> dict:
    """Đếm trùng lặp trong và giữa các tập dựa trên nhóm gần trùng.

    - `images_with_dup_in[a][b]`: số ảnh thuộc tập a có >= 1 ảnh gần trùng trong tập b (a != b).
    - `images_with_dup_in_same_split[a]`: số ảnh tập a có >= 1 ảnh gần trùng khác trong chính tập a.
    Ảnh không có trong `split_of` (không thuộc cách chia) bị bỏ qua.
    """
    spanning = 0
    leaked: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    same: dict[str, int] = defaultdict(int)
    for members in groups:
        in_split = [m for m in members if m in split_of]
        count = {
            p: sum(split_of[m] == p for m in in_split) for p in {split_of[m] for m in in_split}
        }
        if len(count) > 1:
            spanning += 1
        for m in in_split:
            a = split_of[m]
            if count[a] > 1:
                same[a] += 1
            for b in set(count) - {a}:
                leaked[a][b] += 1
    return {
        "groups_spanning_splits": spanning,
        "images_with_dup_in": {a: dict(v) for a, v in leaked.items()},
        "images_with_dup_in_same_split": dict(same),
    }


def secondary_cross_split_pairs(
    hashes: dict[str, dict], split_of: dict[str, str], threshold: int = CHECK_THRESHOLD
) -> int:
    """Kiểm tra chéo bằng dHash 8x8 (độc lập pHash): số cặp gần trùng ở hai tập khác nhau."""
    names = sorted(n for n in hashes if n in split_of)
    pairs = hamming_pairs([hashes[n]["dhash8"] for n in names], 64, threshold)
    return sum(split_of[names[a]] != split_of[names[b]] for a, b, _ in pairs)


def save_group_montages(
    image_dir: Path,
    groups: list[dict],
    out_dir: Path,
    split_of: dict[str, str] | None = None,
    n_examples: int = 8,
    seed: int = 0,
    tile_h: int = 150,
) -> list[str]:
    """Lưu ảnh ghép nhỏ (JPEG) của một số nhóm tiêu biểu: nhóm lớn nhất + ngẫu nhiên."""
    from PIL import ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("group_*.jpg"):
        old.unlink()
    rng = random.Random(seed)
    by_size = sorted(groups, key=lambda g: -len(g["members"]))
    picks = by_size[:2] + rng.sample(by_size[2:], min(n_examples - 2, max(0, len(by_size) - 2)))
    if split_of:  # ưu tiên có ví dụ nhóm trải qua nhiều tập
        spanning = [g for g in groups if len({split_of.get(m) for m in g["members"]}) > 1]
        picks = picks[: n_examples - 2] + rng.sample(spanning, min(2, len(spanning)))
    saved = []
    for k, g in enumerate(picks):
        tiles = []
        for m in g["members"][:5]:
            with Image.open(image_dir / m) as im:
                im = im.convert("RGB")
                im.thumbnail((tile_h * 2, tile_h))
                tiles.append((m, im.copy()))
        canvas = Image.new(
            "RGB", (sum(t.width for _, t in tiles) + 4 * len(tiles), tile_h + 14), "white"
        )
        x, draw = 0, ImageDraw.Draw(canvas)
        for m, t in tiles:
            canvas.paste(t, (x, 14))
            label = (split_of or {}).get(m, "")
            draw.text((x + 2, 1), label, fill="red")
            x += t.width + 4
        name = f"group_{k:02d}_n{len(g['members'])}.jpg"
        canvas.save(out_dir / name, quality=70)
        saved.append(name)
    return saved
