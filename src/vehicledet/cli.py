"""CLI thống nhất: python -m vehicledet <lệnh> ..."""

from __future__ import annotations

import argparse
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
    elif args.command == "train":
        from vehicledet.train import run_train

        print(run_train(args.config, args.data_root, args.override))
    return 0
