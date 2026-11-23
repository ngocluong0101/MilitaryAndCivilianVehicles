"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Chia train/val/test: stratify theo lớp chính của ảnh, nhóm gần trùng nằm chung một tập.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from vehicledet.data.audit import ImageRecord
from vehicledet.data.dedup import cross_split_leakage, secondary_cross_split_pairs
from vehicledet.data.labels import Box

SPLIT_NAMES = ("train", "val", "test")
BACKGROUND = -1  # "lớp" của ảnh không có bbox


def main_class(boxes: list[Box]) -> int:
    """Lớp có nhiều bbox nhất (hoà: diện tích lớn hơn, rồi id nhỏ hơn); không bbox -> BACKGROUND."""
    if not boxes:
        return BACKGROUND
    count: Counter[int] = Counter(b.cls for b in boxes)
    area: dict[int, float] = defaultdict(float)
    for b in boxes:
        area[b.cls] += b.w * b.h
    return min(count, key=lambda c: (-count[c], -area[c], c))


def eligibility(rec: ImageRecord) -> str | None:
    """None nếu ảnh dùng được; ngược lại trả lý do loại (không sửa nhãn, chỉ loại)."""
    if not rec.readable:
        return "unreadable_image"
    if not rec.has_txt:
        return "no_label_file"
    if rec.has_error:
        return "label_error"
    return None


def assign_splits(
    strata: dict[str, int],
    groups: list[list[str]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Gán mỗi nhóm gần trùng vào một tập sao cho phân bố từng stratum sát tỉ lệ mục tiêu.

    `strata`: {ảnh -> lớp chính hoặc BACKGROUND}; ảnh không thuộc nhóm nào là nhóm 1 ảnh.
    Tham lam: nhóm lớn trước, vào tập đang "thiếu" nhiều nhất.
    """
    if abs(sum(ratios) - 1) > 1e-6:
        raise ValueError(f"Tổng tỉ lệ phải bằng 1, nhận được {ratios}")
    grouped = {m for g in groups for m in g}
    units = [list(g) for g in groups] + [[n] for n in strata if n not in grouped]
    rng = random.Random(seed)
    units.sort()  # thứ tự xác định trước khi xáo
    rng.shuffle(units)
    units.sort(key=len, reverse=True)  # sort ổn định: giữ thứ tự ngẫu nhiên trong cùng kích thước

    total = Counter(strata.values())
    target = {s: {c: r * total[c] for c in total} for s, r in zip(SPLIT_NAMES, ratios, strict=True)}
    cur: dict[str, Counter] = {s: Counter() for s in SPLIT_NAMES}
    assignment: dict[str, str] = {}

    for unit in units:
        need = Counter(strata[n] for n in unit)
        scores = {
            s: sum(
                k * (target[s][c] - cur[s][c]) / max(target[s][c], 1e-9) for c, k in need.items()
            )
            for s in SPLIT_NAMES
        }
        best = max(scores.values())
        chosen = rng.choice([s for s in SPLIT_NAMES if scores[s] >= best - 1e-12])
        for n in unit:
            assignment[n] = chosen
        cur[chosen].update(need)
    return assignment


def describe_split(
    split: dict[str, list[str]], records: dict[str, ImageRecord], class_names: list[str]
) -> dict:
    out: dict = {}
    for part, names in split.items():
        boxes = Counter()
        main = Counter()
        for n in names:
            rec = records[n]
            boxes.update(b.cls for b in rec.boxes)
            main[main_class(rec.boxes)] += 1
        out[part] = {
            "images": len(names),
            "boxes": sum(boxes.values()),
            "background_images": main[BACKGROUND],
            "boxes_per_class": {class_names[c]: boxes[c] for c in range(len(class_names))},
            "images_per_main_class": {class_names[c]: main[c] for c in range(len(class_names))},
        }
    return out


def write_split(
    out_dir: Path,
    assignment: dict[str, str],
    excluded: dict[str, str],
    info: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in SPLIT_NAMES:
        names = sorted(n for n, s in assignment.items() if s == part)
        (out_dir / f"{part}.txt").write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    excl = out_dir / "excluded.tsv"
    if excluded:
        excl.write_text(
            "".join(f"{n}\t{r}\n" for n, r in sorted(excluded.items())), encoding="utf-8"
        )
    elif excl.exists():
        excl.unlink()
    (out_dir / "split_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_split(
    records: dict[str, ImageRecord],
    dup_groups: list[list[str]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Trả về (assignment, excluded)."""
    excluded = {n: r for n, rec in records.items() if (r := eligibility(rec))}
    strata = {n: main_class(rec.boxes) for n, rec in records.items() if n not in excluded}
    groups = [[m for m in g if m in strata] for g in dup_groups]
    groups = [g for g in groups if len(g) > 1]
    return assign_splits(strata, groups, ratios, seed), excluded


def compare_report(
    splits: dict[str, dict[str, list[str]]],
    records: dict[str, ImageRecord],
    class_names: list[str],
    dup_groups: list[list[str]],
    hashes: dict[str, dict],
) -> dict:
    """Thống kê + rò rỉ cho nhiều cách chia (v2, ...)."""
    out = {}
    for name, split in splits.items():
        split_of = {n: p for p, names in split.items() for n in names}
        out[name] = {
            "stats": describe_split(split, records, class_names),
            "leakage_phash16": cross_split_leakage(dup_groups, split_of),
            "cross_split_pairs_dhash8_check": secondary_cross_split_pairs(hashes, split_of),
        }
    return out


def render_markdown(report: dict, params: dict) -> str:
    L = [
        "# Thống kê chia tập & rò rỉ",
        "",
        "Sinh bởi `python -m vehicledet make-split`. "
        f"Tham số: {json.dumps(params, ensure_ascii=False)}",
        "",
    ]
    for name, r in report.items():
        L += [
            f"## Cách chia `{name}`",
            "",
            "| Tập | Ảnh | Bbox | Ảnh nền (không bbox) |",
            "|---|---|---|---|",
        ]
        for part, s in r["stats"].items():
            L.append(f"| {part} | {s['images']:,} | {s['boxes']:,} | {s['background_images']} |")
        classes = list(next(iter(r["stats"].values()))["boxes_per_class"])
        L += [
            "",
            "Số ảnh theo lớp chính / số bbox theo lớp:",
            "",
            "| Lớp | " + " | ".join(r["stats"]) + " |",
            "|---|" + "---|" * len(r["stats"]),
        ]
        for c in classes:
            cells = " | ".join(
                f"{r['stats'][p]['images_per_main_class'][c]} / "
                f"{r['stats'][p]['boxes_per_class'][c]}"
                for p in r["stats"]
            )
            L.append(f"| {c} | {cells} |")
        lk = r["leakage_phash16"]
        L += [
            "",
            f"- Nhóm gần trùng (pHash 16x16) nằm ở >1 tập: **{lk['groups_spanning_splits']}**",
        ]
        for a, v in lk["images_with_dup_in"].items():
            for b, n in v.items():
                L.append(f"- Ảnh tập `{a}` có bản gần trùng trong `{b}`: **{n}**")
        L += [
            "- Kiểm tra chéo độc lập bằng dHash 8x8 (Hamming ≤ 4): "
            f"**{r['cross_split_pairs_dhash8_check']}** "
            "cặp nằm ở hai tập khác nhau",
            "",
        ]
    return "\n".join(L) + "\n"
