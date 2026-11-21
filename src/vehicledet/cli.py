"""CLI thống nhất: python -m vehicledet <lệnh> ..."""

from __future__ import annotations

import argparse
import json
import logging

from vehicledet.data.splits import write_legacy_split
from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.paths import resolve_data_root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vehicledet", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_data_root(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--data-root", help="Thư mục chứa Images/ và Labels/ (mặc định: $DATA_ROOT)"
        )

    sp = sub.add_parser(
        "make-legacy-split", help="Tái tạo cách chia train/val cũ -> splits/legacy/"
    )
    add_data_root(sp)

    sp = sub.add_parser(
        "prepare-data", help="Dựng thư mục YOLO (symlink) + data.yaml cho một split"
    )
    add_data_root(sp)
    sp.add_argument("--split", required=True, help="Tên thư mục trong splits/ (vd: legacy)")

    sp = sub.add_parser("audit", help="Kiểm tra chất lượng dữ liệu -> reports/data_audit.{md,json}")
    add_data_root(sp)

    sp = sub.add_parser(
        "find-duplicates", help="Tìm ảnh trùng/gần trùng (pHash) -> reports/duplicates.*"
    )
    add_data_root(sp)
    sp.add_argument("--threshold", type=int, default=56, help="Ngưỡng Hamming trên pHash 16x16")

    sp = sub.add_parser(
        "make-split", help="Chia train/val/test theo nhóm gần trùng -> splits/<name>/"
    )
    add_data_root(sp)
    sp.add_argument("--name", default="v2")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument(
        "--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST")
    )

    sp = sub.add_parser("train", help="Train từ file config")
    add_data_root(sp)
    sp.add_argument("--config", required=True, help="vd: configs/train_yolo11s.yaml")
    sp.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ghi đè tham số train, vd: --override epochs=1",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "make-legacy-split":
        r = write_legacy_split(resolve_data_root(args.data_root))
        print(
            f"train={len(r.train)} val={len(r.val)} "
            f"(ảnh val lặp trong train_labels.csv đã loại: {r.overlap_removed})"
        )
        if r.missing_images:
            print(
                f"CẢNH BÁO: {len(r.missing_images)} ảnh có trong CSV nhưng không có trên đĩa: "
                f"{r.missing_images}"
            )
    elif args.command == "prepare-data":
        ds = build_yolo_dataset(resolve_data_root(args.data_root), args.split)
        print(f"{ds.data_yaml}  counts={ds.counts}")
    elif args.command == "audit":
        from vehicledet.data.commands import run_audit

        r = run_audit(resolve_data_root(args.data_root))
        print(
            json.dumps(
                {
                    "images": r["images"],
                    "boxes": r["boxes"]["total"],
                    "label_problems": r["label_problems"]["counts"],
                    "csv_vs_txt": r["csv_vs_txt"]["status_counts"],
                },
                ensure_ascii=False,
                indent=1,
            )
        )
    elif args.command == "find-duplicates":
        from vehicledet.data.commands import run_find_duplicates

        r = run_find_duplicates(resolve_data_root(args.data_root), threshold=args.threshold)
        print(json.dumps(r["summary"], ensure_ascii=False, indent=1))
    elif args.command == "make-split":
        from vehicledet.data.commands import run_make_split

        r = run_make_split(
            resolve_data_root(args.data_root), args.name, args.seed, tuple(args.ratios)
        )
        print(
            json.dumps(
                {
                    k: v["stats"] and {p: s["images"] for p, s in v["stats"].items()}
                    for k, v in r["report"].items()
                },
                indent=1,
            )
        )
        print(f"loại khỏi split: {len(r['excluded'])} ảnh")
    elif args.command == "train":
        from vehicledet.train import run_train

        print(run_train(args.config, args.data_root, args.override))
    return 0
