"""Audit chất lượng dữ liệu: thống kê, nhãn hỏng, ảnh hỏng, CSV vs TXT."""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from vehicledet.data.labels import (
    ERROR_KINDS,
    WARNING_KINDS,
    Box,
    LabelProblem,
    compare_boxes,
    parse_yolo_txt,
    read_csv_boxes,
)
from vehicledet.paths import class_names, csv_labels_dir, images_dir, txt_labels_dir

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SMALL_AREA, MEDIUM_AREA = 32**2, 96**2  # chuẩn COCO (pixel^2, trên ảnh gốc)
MAX_EXAMPLES = 20


@dataclass
class ImageRecord:
    name: str
    size: tuple[int, int] | None  # (w, h); None nếu ảnh không đọc được
    read_error: str | None
    has_txt: bool
    boxes: list[Box] = field(default_factory=list)
    problems: list[LabelProblem] = field(default_factory=list)

    @property
    def readable(self) -> bool:
        return self.size is not None

    @property
    def has_error(self) -> bool:
        return any(p.is_error for p in self.problems)


def list_images(data_root: Path) -> list[Path]:
    return sorted(
        p
        for p in images_dir(data_root).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def scan_dataset(data_root: Path) -> dict[str, ImageRecord]:
    """Đọc mọi ảnh (giải mã đầy đủ) và nhãn TXT tương ứng."""
    txt_dir = txt_labels_dir(data_root)
    n_cls = len(class_names())
    records: dict[str, ImageRecord] = {}
    images = list_images(data_root)
    for i, img in enumerate(images, 1):
        size, err = None, None
        try:
            with Image.open(img) as im:
                im.load()
                size = im.size
        except Exception as e:  # noqa: BLE001 - mọi lỗi giải mã đều là "ảnh hỏng"
            err = f"{type(e).__name__}: {e}"
        txt = txt_dir / (img.stem + ".txt")
        rec = ImageRecord(img.name, size, err, txt.is_file())
        if rec.has_txt:
            rec.boxes, rec.problems = parse_yolo_txt(txt, n_cls, size)
        records[img.name] = rec
        if i % 1000 == 0:
            log.info("đã quét %d/%d ảnh", i, len(images))
    return records


def box_size_category(box: Box, size: tuple[int, int]) -> str:
    area = box.w * size[0] * box.h * size[1]
    return "small" if area < SMALL_AREA else "medium" if area < MEDIUM_AREA else "large"


def _dist(values: list[int]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": s[0],
        "mean": round(statistics.fmean(s), 2),
        "median": statistics.median(s),
        "p95": s[min(len(s) - 1, int(0.95 * len(s)))],
        "max": s[-1],
    }


def _histogram(counts: list[int]) -> dict[str, int]:
    buckets = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, ">10": 0}
    for n in counts:
        buckets[
            "1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else ">10"
        ] += 1
    return buckets


def compare_csv_txt(data_root: Path, records: dict[str, ImageRecord]) -> dict:
    names = class_names()
    class_to_id = {n: i for i, n in enumerate(names)}
    csv_dir = csv_labels_dir(data_root)
    csv_boxes, csv_sizes, unknown = read_csv_boxes(
        [csv_dir / "train_labels.csv", csv_dir / "test_labels.csv"], class_to_id
    )
    status: dict[str, list[str]] = defaultdict(list)
    size_mismatch: list[str] = []

    for fn, cboxes in csv_boxes.items():
        rec = records.get(fn)
        if rec is None:
            status["csv_image_missing_on_disk"].append(fn)
            continue
        if not rec.has_txt:
            status["csv_only_no_txt"].append(fn)
            continue
        if rec.size and rec.size not in csv_sizes[fn]:
            size_mismatch.append(fn)
        status[compare_boxes(cboxes, rec.boxes)].append(fn)
    for fn, rec in records.items():
        if fn in csv_boxes:
            continue
        if not rec.has_txt:
            status["no_txt_and_not_in_csv"].append(fn)
        else:
            status[
                "txt_only_not_in_csv_labeled" if rec.boxes else "txt_only_not_in_csv_empty"
            ].append(fn)

    # Ảnh có trong CSV nhưng thiếu trên đĩa: có thể đã bị đổi tên -> tìm ảnh có TXT khớp bbox
    unmatched = [r for r in records.values() if r.name not in csv_boxes and r.boxes]
    renamed = {}
    for fn in status.get("csv_image_missing_on_disk", []):
        for rec in unmatched:
            same_size = rec.size is None or rec.size in csv_sizes[fn]
            if same_size and compare_boxes(csv_boxes[fn], rec.boxes) == "match":
                renamed[fn] = rec.name
                break

    return {
        "images_in_csv": len(csv_boxes),
        "csv_missing_probably_renamed": renamed,
        "status_counts": {k: len(v) for k, v in sorted(status.items())},
        "status_examples": {
            k: sorted(v)[:MAX_EXAMPLES] for k, v in sorted(status.items()) if k != "match"
        },
        "status_files": {k: sorted(v) for k, v in status.items() if k != "match"},
        "image_size_mismatch_count": len(size_mismatch),
        "image_size_mismatch_examples": sorted(size_mismatch)[:MAX_EXAMPLES],
        "unknown_csv_classes": unknown,
    }


def run_audit(data_root: Path, records: dict[str, ImageRecord] | None = None) -> dict:
    names = class_names()
    records = records or scan_dataset(data_root)
    txt_dir = txt_labels_dir(data_root)
    image_stems = {Path(n).stem for n in records}
    txt_stems = {p.stem for p in txt_dir.glob("*.txt")}

    unreadable = {n: r.read_error for n, r in records.items() if not r.readable}
    no_txt = sorted(n for n, r in records.items() if not r.has_txt)
    txt_without_image = sorted(txt_stems - image_stems)
    no_boxes = sorted(n for n, r in records.items() if r.has_txt and not r.boxes)

    boxes_per_class: Counter[int] = Counter()
    size_per_class: dict[int, Counter] = defaultdict(Counter)
    images_per_class: Counter[int] = Counter()
    boxes_per_image: list[int] = []
    for r in records.values():
        if not r.boxes:
            continue
        boxes_per_image.append(len(r.boxes))
        images_per_class.update({b.cls for b in r.boxes})
        for b in r.boxes:
            boxes_per_class[b.cls] += 1
            if r.size:
                size_per_class[b.cls][box_size_category(b, r.size)] += 1

    problems = [p for r in records.values() for p in r.problems]
    by_kind: dict[str, list[LabelProblem]] = defaultdict(list)
    for p in problems:
        by_kind[p.kind].append(p)
    problem_counts = {k: len(by_kind.get(k, [])) for k in (*ERROR_KINDS, *WARNING_KINDS)}
    problem_examples = {
        k: [f"{p.file}:{p.line} {p.detail}" for p in v[:MAX_EXAMPLES]] for k, v in by_kind.items()
    }
    images_with_error = sorted(n for n, r in records.items() if r.has_error)

    total_boxes = sum(boxes_per_class.values())
    return {
        "data_root_images": len(records),
        "classes": names,
        "images": {
            "total": len(records),
            "unreadable": len(unreadable),
            "with_boxes": len(boxes_per_image),
            "no_txt_file": len(no_txt),
            "txt_empty_or_no_valid_box": len(no_boxes),
            "txt_without_image": len(txt_without_image),
            "with_label_error": len(images_with_error),
        },
        "boxes": {
            "total": total_boxes,
            "per_class": {names[c]: boxes_per_class[c] for c in range(len(names))},
            "images_per_class": {names[c]: images_per_class[c] for c in range(len(names))},
            "per_image": _dist(boxes_per_image),
            "per_image_histogram": _histogram(boxes_per_image),
            "size_per_class": {
                names[c]: {k: size_per_class[c][k] for k in ("small", "medium", "large")}
                for c in range(len(names))
            },
        },
        "label_problems": {
            "counts": problem_counts,
            "examples": problem_examples,
            "images_with_error": images_with_error,
        },
        "unreadable_images": unreadable,
        "image_lists": {
            "no_txt_file": no_txt,
            "txt_empty_or_no_valid_box": no_boxes,
            "txt_without_image": txt_without_image,
        },
        "csv_vs_txt": compare_csv_txt(data_root, records),
    }


def render_markdown(a: dict) -> str:
    img, bx, pr, cv = a["images"], a["boxes"], a["label_problems"], a["csv_vs_txt"]
    L = [
        "# Báo cáo kiểm tra dữ liệu",
        "",
        "Sinh tự động bởi `python -m vehicledet audit`; danh sách đầy đủ trong `data_audit.json`.",
        "",
        "## 1. Tổng quan ảnh",
        "",
        "| Chỉ số | Số lượng |",
        "|---|---|",
    ]
    labels = {
        "total": "Tổng số ảnh",
        "unreadable": "Ảnh hỏng (không giải mã được)",
        "with_boxes": "Ảnh có ≥1 bbox hợp lệ",
        "no_txt_file": "Ảnh không có file nhãn TXT",
        "txt_empty_or_no_valid_box": "Ảnh có file TXT nhưng không có bbox hợp lệ",
        "txt_without_image": "File TXT không có ảnh tương ứng",
        "with_label_error": "Ảnh có ≥1 dòng nhãn lỗi",
    }
    L += [f"| {labels[k]} | {v:,} |" for k, v in img.items()]

    L += [
        "",
        "## 2. Bbox theo lớp",
        "",
        f"Tổng số bbox hợp lệ: **{bx['total']:,}**",
        "",
        "| Lớp | Số bbox | % bbox | Số ảnh chứa lớp | small | medium | large |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in a["classes"]:
        n, s = bx["per_class"][c], bx["size_per_class"][c]
        tot = max(1, sum(s.values()))
        L.append(
            f"| {c} | {n:,} | {100 * n / max(1, bx['total']):.1f}% "
            f"| {bx['images_per_class'][c]:,} | "
            + " | ".join(
                f"{s[k]:,} ({100 * s[k] / tot:.0f}%)" for k in ("small", "medium", "large")
            )
            + " |"
        )
    L += [
        "",
        "small/medium/large theo COCO: diện tích bbox < 32², < 96², còn lại (pixel², ảnh gốc).",
        "",
        "Số bbox mỗi ảnh (ảnh có nhãn): "
        + ", ".join(f"{k}={v}" for k, v in bx["per_image"].items()),
        "",
        "| Số bbox/ảnh | " + " | ".join(bx["per_image_histogram"]) + " |",
        "|---|" + "---|" * len(bx["per_image_histogram"]),
        "| Số ảnh | " + " | ".join(str(v) for v in bx["per_image_histogram"].values()) + " |",
    ]

    L += ["", "## 3. Nhãn hỏng / đáng ngờ", "", "| Loại | Mức | Số dòng |", "|---|---|---|"]
    for k, v in pr["counts"].items():
        L.append(f"| `{k}` | {'lỗi' if k in ERROR_KINDS else 'cảnh báo'} | {v:,} |")
    for k, ex in pr["examples"].items():
        L += ["", f"Ví dụ `{k}`:", ""] + [f"- `{e}`" for e in ex[:5]]
    if a["unreadable_images"]:
        L += ["", "Ảnh hỏng:", ""] + [
            f"- `{n}`: {e}" for n, e in list(a["unreadable_images"].items())[:10]
        ]

    L += [
        "",
        "## 4. So sánh CSV với TXT",
        "",
        f"Số ảnh có trong CSV: **{cv['images_in_csv']:,}**",
        "",
        "| Trạng thái | Số ảnh |",
        "|---|---|",
    ]
    L += [f"| `{k}` | {v:,} |" for k, v in cv["status_counts"].items()]
    L += [
        "",
        f"Ảnh có kích thước thật khác width/height trong CSV: "
        f"**{cv['image_size_mismatch_count']}**",
    ]
    for k, ex in cv["status_examples"].items():
        L += ["", f"Ví dụ `{k}`:", ""] + [f"- `{e}`" for e in ex[:5]]
    if cv["csv_missing_probably_renamed"]:
        L += [
            "",
            "Ảnh có trong CSV nhưng thiếu trên đĩa, khả năng cao đã bị đổi tên "
            "(bbox + kích thước khớp 1-1 với một ảnh có TXT nhưng không có trong CSV):",
            "",
        ]
        L += [f"- `{a}` → `{b}`" for a, b in cv["csv_missing_probably_renamed"].items()]
    if cv["unknown_csv_classes"]:
        L += ["", f"Tên lớp lạ trong CSV: {cv['unknown_csv_classes']}"]
    return "\n".join(L) + "\n"


def write_audit(data_root: Path, out_dir: Path) -> dict:
    result = run_audit(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "data_audit.md").write_text(render_markdown(result), encoding="utf-8")
    return result
