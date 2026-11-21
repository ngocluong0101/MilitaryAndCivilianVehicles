"""Đọc & kiểm tra nhãn YOLO (.txt) và CSV của dataset."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

EDGE_TOLERANCE = 1e-3  # sai số cho phép khi bbox chạm/vượt biên ảnh
TINY_SIDE_PX = 4  # cạnh bbox nhỏ hơn số này (pixel) coi là đáng ngờ

ERROR_KINDS = ("bad_format", "class_out_of_range", "coord_out_of_range", "zero_size")
WARNING_KINDS = ("box_exceeds_image", "duplicate_box", "tiny_box")


@dataclass(frozen=True)
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float


@dataclass(frozen=True)
class LabelProblem:
    file: str
    line: int
    kind: str
    detail: str

    @property
    def is_error(self) -> bool:
        return self.kind in ERROR_KINDS


def parse_yolo_txt(
    path: Path, num_classes: int, image_size: tuple[int, int] | None = None
) -> tuple[list[Box], list[LabelProblem]]:
    """Đọc một file nhãn YOLO. Trả về (box hợp lệ, danh sách vấn đề).

    Box có lỗi mức 'error' không nằm trong danh sách box trả về. `image_size` = (w, h) pixel,
    dùng để phát hiện bbox quá nhỏ.
    """
    boxes: list[Box] = []
    problems: list[LabelProblem] = []
    name = path.name
    seen: set[Box] = set()

    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError(f"cần 5 cột, có {len(parts)}")
            cls_f = float(parts[0])
            xc, yc, w, h = (float(x) for x in parts[1:])
        except ValueError as e:
            problems.append(LabelProblem(name, lineno, "bad_format", f"{e}: {line[:60]!r}"))
            continue

        if cls_f != int(cls_f) or not 0 <= int(cls_f) < num_classes:
            problems.append(LabelProblem(name, lineno, "class_out_of_range", f"class={parts[0]}"))
            continue
        if any(not 0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            problems.append(LabelProblem(name, lineno, "coord_out_of_range", line[:60]))
            continue
        if w <= 0 or h <= 0:
            problems.append(LabelProblem(name, lineno, "zero_size", f"w={w} h={h}"))
            continue

        box = Box(int(cls_f), xc, yc, w, h)
        if box in seen:
            problems.append(LabelProblem(name, lineno, "duplicate_box", line[:60]))
        seen.add(box)
        if (
            xc - w / 2 < -EDGE_TOLERANCE
            or yc - h / 2 < -EDGE_TOLERANCE
            or xc + w / 2 > 1 + EDGE_TOLERANCE
            or yc + h / 2 > 1 + EDGE_TOLERANCE
        ):
            problems.append(LabelProblem(name, lineno, "box_exceeds_image", line[:60]))
        if image_size and min(w * image_size[0], h * image_size[1]) < TINY_SIDE_PX:
            problems.append(LabelProblem(name, lineno, "tiny_box", line[:60]))
        boxes.append(box)
    return boxes, problems


def read_csv_boxes(csv_paths: list[Path], class_to_id: dict[str, int]):
    """Đọc các CSV (loại dòng trùng giữa các file). Trả về (boxes theo filename, vấn đề).

    Box được chuẩn hoá về dạng YOLO theo width/height ghi trong CSV. Trả thêm kích thước ảnh
    khai báo trong CSV để so với ảnh thật.
    """
    rows: set[tuple] = set()
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.add(
                    (
                        r["filename"],
                        int(r["width"]),
                        int(r["height"]),
                        r["class"].strip(),
                        int(float(r["xmin"])),
                        int(float(r["ymin"])),
                        int(float(r["xmax"])),
                        int(float(r["ymax"])),
                    )
                )
    boxes: dict[str, list[Box]] = defaultdict(list)
    sizes: dict[str, set[tuple[int, int]]] = defaultdict(set)
    unknown_classes: dict[str, int] = defaultdict(int)
    for fn, w, h, cname, x1, y1, x2, y2 in sorted(rows):
        sizes[fn].add((w, h))
        if cname not in class_to_id:
            unknown_classes[cname] += 1
            continue
        boxes[fn].append(
            Box(
                class_to_id[cname],
                (x1 + x2) / 2 / w,
                (y1 + y2) / 2 / h,
                (x2 - x1) / w,
                (y2 - y1) / h,
            )
        )
    for fn in sizes:
        boxes.setdefault(fn, [])
    return dict(boxes), dict(sizes), dict(unknown_classes)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xc - a.w / 2, a.yc - a.h / 2, a.xc + a.w / 2, a.yc + a.h / 2
    bx1, by1, bx2, by2 = b.xc - b.w / 2, b.yc - b.h / 2, b.xc + b.w / 2, b.yc + b.h / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def compare_boxes(
    csv_boxes: list[Box], txt_boxes: list[Box], iou_good: float = 0.9, iou_min: float = 0.5
) -> str:
    """So khớp nhãn CSV với TXT của cùng một ảnh.

    Trả về một trong:
    - 'count_mismatch': khác số bbox;
    - 'box_mismatch': có bbox không ghép được (IoU < iou_min);
    - 'class_mismatch': vị trí khớp nhưng khác lớp;
    - 'minor_box_diff': khớp lớp nhưng có bbox lệch nhẹ (iou_min <= IoU < iou_good);
    - 'match': mọi bbox khớp (IoU >= iou_good, cùng lớp).
    Ghép tham lam theo IoU cao nhất.
    """
    if len(csv_boxes) != len(txt_boxes):
        return "count_mismatch"
    remaining = list(txt_boxes)
    class_mismatch = False
    worst = 1.0
    for cb in csv_boxes:
        best = max(remaining, key=lambda tb: iou(cb, tb))
        best_iou = iou(cb, best)
        if best_iou < iou_min:
            return "box_mismatch"
        worst = min(worst, best_iou)
        class_mismatch |= best.cls != cb.cls
        remaining.remove(best)
    if class_mismatch:
        return "class_mismatch"
    return "match" if worst >= iou_good else "minor_box_diff"
