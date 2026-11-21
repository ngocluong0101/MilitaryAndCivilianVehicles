"""Tổng hợp reports/experiments/*/ thành bảng so sánh (md + csv) và cập nhật ROADMAP.md.

Dùng: python scripts/summarize_runs.py [--roadmap ROADMAP.md]
"""

import argparse
from pathlib import Path

from vehicledet.experiments import (
    load_rows,
    pick_best,
    render_table,
    replace_marked_block,
    write_summary,
)
from vehicledet.paths import REPORTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roadmap", type=Path, help="Cập nhật khối bảng trong file này")
    args = ap.parse_args()

    rows = load_rows()
    write_summary(rows, REPORTS_DIR)
    table = render_table(rows)
    print(table)
    best = pick_best(rows, "v2", 640)
    if best:
        print(
            f"Tốt nhất theo mAP50-95 trên val (split v2, imgsz 640): {best['name']} "
            f"({best['mAP50-95']:.3f})"
        )
    if args.roadmap:
        text = args.roadmap.read_text(encoding="utf-8")
        args.roadmap.write_text(replace_marked_block(text, table), encoding="utf-8")
        print(f"Đã cập nhật {args.roadmap}")


if __name__ == "__main__":
    main()
