"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Các lệnh dữ liệu cấp cao: audit, find-duplicates, make-split (được gọi từ CLI).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vehicledet.data import dedup
from vehicledet.data.audit import list_images, scan_dataset, write_audit
from vehicledet.data.make_split import (
    SPLIT_NAMES,
    build_split,
    compare_report,
    describe_split,
    render_markdown,
    write_split,
)
from vehicledet.data.splits import read_split
from vehicledet.paths import REPORTS_DIR, SPLITS_DIR, class_names, images_dir

log = logging.getLogger(__name__)


def hash_cache_path(data_root: Path) -> Path:
    return data_root / "cache" / "phash.json"


def run_audit(data_root: Path, out_dir: Path = REPORTS_DIR) -> dict:
    return write_audit(data_root, out_dir)


def _split_of(name: str) -> dict[str, str] | None:
    d = SPLITS_DIR / name
    if not (d / "train.txt").is_file():
        return None
    return {n: p for p, names in read_split(name).items() for n in names}


def _duplicates_markdown(res: dict, legacy: dict | None, montages: list[str]) -> str:
    s, p = res["summary"], res["params"]
    L = [
        "# Ảnh trùng / gần trùng",
        "",
        "Sinh bởi `python -m vehicledet find-duplicates` (danh sách nhóm đầy đủ: `duplicates.json`).",
        "",
        "## Phương pháp",
        "",
        f"- pHash {p['hash_size']}x{p['hash_size']} ({p['bits']} bit) trên ảnh RGB→xám; hai ảnh cùng "
        f"nhóm nếu khoảng cách Hamming ≤ **{p['threshold']}**, **hoặc** pHash 8x8 (64 bit) cách nhau ≤ "
        f"**{p['coarse_phash8_threshold']}** (bắt thêm ảnh bị zoom/crop); nhóm lấy bao đóng bắc cầu (union-find).",
        "- Ngưỡng chọn bằng cách xem trực quan các cặp theo từng mức khoảng cách (mẫu 8–12 cặp mỗi mức): "
        "đến ~44 (/256) vẫn là cùng một bức ảnh (khác watermark, màu, độ phân giải, crop); từ ~45 bắt đầu "
        "lẫn ảnh cùng mẫu nhưng khác nội dung (vd ô tô nhìn từ trên xuống khác màu).",
        "- Chọn **56**: số nhóm gần như phẳng trong khoảng 8–56 (xem bảng độ nhạy); từ 64 trở lên nhóm bắt đầu "
        "bị nối dây chuyền (nhóm lớn nhất tăng). Ưu tiên chống rò rỉ nên chấp nhận gộp thêm vài ảnh chỉ giống "
        "mẫu: hậu quả duy nhất là chúng buộc phải nằm chung một tập.",
        "",
        "## Kết quả",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---|",
        f"| Ảnh đã hash | {s['images_hashed']:,} |",
        f"| Nhóm gần trùng (≥2 ảnh) | {s['near_duplicate_groups']:,} |",
        f"| Ảnh nằm trong nhóm gần trùng | {s['images_in_groups']:,} "
        f"({100 * s['images_in_groups'] / max(1, s['images_hashed']):.1f}%) |",
        f"| Ảnh không có bản gần trùng | {s['images_not_in_any_group']:,} |",
        f'| Số "bức ảnh khác nhau" ước lượng | {s["approx_distinct_pictures"]:,} |',
        f"| Nhóm lớn nhất | {s['largest_group']} ảnh |",
        f"| Ảnh chỉ được gom nhờ tiêu chí pHash 8x8 | {s['extra_images_from_coarse_phash8']} |",
        f"| Nhóm trùng byte (md5) | {s['exact_duplicate_groups_md5']:,} nhóm "
        f"({s['images_in_exact_duplicate_groups']:,} ảnh) |",
        "",
        "### Độ nhạy theo ngưỡng (chỉ pHash 16x16)",
        "",
        "| Ngưỡng Hamming (/256) | Nhóm | Ảnh trong nhóm | Nhóm lớn nhất |",
        "|---|---|---|---|",
    ]
    for t, v in s["sensitivity_phash16_only"].items():
        L.append(f"| {t} | {v['groups']:,} | {v['images_in_groups']:,} | {v['largest_group']} |")
    if legacy:
        lk = legacy["leakage_phash16"]
        L += [
            "",
            "## Rò rỉ trong cách chia cũ (`splits/legacy`)",
            "",
            f"- Nhóm gần trùng nằm ở cả train và val: **{lk['groups_spanning_splits']}**",
        ]
        for a, v in lk["images_with_dup_in"].items():
            for b, n in v.items():
                L.append(f"- Ảnh `{a}` có bản gần trùng trong `{b}`: **{n}**")
        for a, n in lk["images_with_dup_in_same_split"].items():
            L.append(f"- Ảnh `{a}` có bản gần trùng khác trong chính `{a}`: **{n}**")
    if montages:
        L += ["", "## Ví dụ (chữ đỏ = tập theo cách chia cũ, nếu có)", ""]
        L += [f"![{m}](duplicates/{m})" for m in montages]
    return "\n".join(L) + "\n"


def run_find_duplicates(
    data_root: Path, out_dir: Path = REPORTS_DIR, threshold: int = dedup.THRESHOLD
) -> dict:
    names = [p.name for p in list_images(data_root)]
    hashes = dedup.compute_hashes(images_dir(data_root), names, hash_cache_path(data_root))
    res = dedup.find_duplicate_groups(hashes, threshold)

    legacy_of = _split_of("legacy")
    legacy = None
    if legacy_of:
        groups = [g["members"] for g in res["groups"]]
        legacy = {"leakage_phash16": dedup.cross_split_leakage(groups, legacy_of)}
        res["legacy_split_leakage"] = legacy["leakage_phash16"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "duplicates.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    montages = dedup.save_group_montages(
        images_dir(data_root), res["groups"], out_dir / "duplicates", legacy_of
    )
    (out_dir / "duplicates.md").write_text(
        _duplicates_markdown(res, legacy, montages), encoding="utf-8"
    )
    return res


def run_make_split(
    data_root: Path,
    name: str = "v2",
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    out_dir: Path = REPORTS_DIR,
) -> dict:
    dup_path = out_dir / "duplicates.json"
    if not dup_path.is_file():
        raise FileNotFoundError(f"Chưa có {dup_path}; chạy `find-duplicates` trước.")
    dup = json.loads(dup_path.read_text(encoding="utf-8"))
    dup_groups = [g["members"] for g in dup["groups"]]

    records = scan_dataset(data_root)
    assignment, excluded = build_split(records, dup_groups, ratios, seed)
    split = {p: sorted(n for n, s in assignment.items() if s == p) for p in SPLIT_NAMES}

    params = {
        "name": name,
        "seed": seed,
        "ratios": list(ratios),
        "dup_threshold": dup["params"]["threshold"],
        "dup_hash": dup["params"]["hash"],
    }
    info = {
        **params,
        "counts": {p: len(v) for p, v in split.items()},
        "excluded": len(excluded),
        "stratify_by": "main class of image (most boxes; background = no boxes)",
    }
    write_split(SPLITS_DIR / name, assignment, excluded, info)

    names = [p.name for p in list_images(data_root)]
    hashes = dedup.compute_hashes(images_dir(data_root), names, hash_cache_path(data_root))
    splits = {name: split}
    legacy = _split_of("legacy")
    if legacy:
        splits = {"legacy": read_split("legacy"), **splits}
    report = compare_report(splits, records, class_names(), dup_groups, hashes)
    (out_dir / "split_stats.json").write_text(
        json.dumps({"params": params, "splits": report}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "split_stats.md").write_text(render_markdown(report, params), encoding="utf-8")
    return {"report": report, "excluded": excluded}


__all__ = ["run_audit", "run_find_duplicates", "run_make_split", "describe_split"]
