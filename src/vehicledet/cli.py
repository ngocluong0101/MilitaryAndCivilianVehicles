"""CLI thống nhất: python -m vehicledet <lệnh> ..."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
    sp.add_argument("--config", required=True, help="vd: configs/exp3b_yolo11s_v2_640.yaml")
    sp.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ghi đè tham số train hoặc model/split/name/seed, vd: --override epochs=1",
    )
    sp.add_argument("--no-mlflow", action="store_true", help="Không ghi MLflow")
    sp.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Bỏ qua nếu reports/experiments/<name>/ đã có kết quả",
    )

    sp = sub.add_parser("evaluate", help="Đánh giá weights trên một split -> reports/eval/*.json")
    add_data_root(sp)
    sp.add_argument("--weights", required=True)
    sp.add_argument("--split", required=True, help="Tên thư mục trong splits/ (vd: v2)")
    sp.add_argument("--part", choices=["val", "test"], default="val")
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--device", default=None)
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--tag", default=None)
    sp.add_argument(
        "--final",
        action="store_true",
        help="BẮT BUỘC cho --part test: xác nhận đây là lần báo cáo cuối",
    )
    sp.add_argument("--allow-rerun", action="store_true", help="Cho phép ghi đè test_results.json")

    sp = sub.add_parser(
        "export", help="Export best.pt -> ONNX, TensorRT FP16/INT8 trong exports/<tag>/"
    )
    add_data_root(sp)
    sp.add_argument("--weights", required=True)
    sp.add_argument("--tag", required=True, help="Tên thư mục trong exports/ (vd: exp3b)")
    sp.add_argument("--split", default="v2", help="Split lấy ảnh TRAIN để hiệu chuẩn INT8")
    sp.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "engine-fp16", "engine-int8"],
        choices=["onnx", "engine-fp16", "engine-int8"],
    )
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--device", default="0")

    sp = sub.add_parser(
        "benchmark", help="Benchmark tốc độ/VRAM/mAP các định dạng -> reports/benchmark.csv"
    )
    add_data_root(sp)
    sp.add_argument("--model-dir", required=True, help="Thư mục exports/<tag>")
    sp.add_argument("--split", default="v2")
    sp.add_argument("--device", default="0", help="'0' (GPU) hoặc 'cpu'")
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--warmup", type=int, default=20)
    sp.add_argument("--iters", type=int, default=200)
    sp.add_argument(
        "--map-part",
        choices=["val", "test", "none"],
        default="val",
        help="Tập đo mAP; 'test' cần --final",
    )
    sp.add_argument("--final", action="store_true", help="Xác nhận dùng tập test cho báo cáo cuối")
    sp.add_argument("--only", nargs="+", help="Chỉ đo các định dạng này (vd: onnx_fp32)")
    sp.add_argument("--map-batch", type=int, default=8, help="Batch khi đo mAP")
    sp.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Số vòng đo lặp (xen kẽ các định dạng); báo cáo trung vị và khoảng min–max",
    )
    sp.add_argument(
        "--no-isolate",
        action="store_true",
        help="Đo mọi định dạng trong một tiến trình (VRAM sẽ lẫn giữa các định dạng)",
    )
    sp.add_argument(
        "--row-out", help=argparse.SUPPRESS
    )  # nội bộ: tiến trình con ghi một dòng kết quả
    sp.add_argument("--roadmap", help="Cập nhật bảng benchmark trong file này (ROADMAP.md)")

    sp = sub.add_parser("analyze-errors", help="Phân tích lỗi trên VAL -> reports/error_analysis/")
    add_data_root(sp)
    sp.add_argument("--weights", required=True)
    sp.add_argument("--split", default="v2")
    sp.add_argument("--conf", type=float, default=0.25, help="Ngưỡng điểm khi phân tích lỗi")
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--device", default=None)
    sp.add_argument("--out", default=None, help="Thư mục xuất (mặc định reports/error_analysis)")

    def add_model_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", required=True, help="File .onnx hoặc .engine")
        sp.add_argument("--backend", choices=["auto", "onnx", "tensorrt"], default="auto")
        sp.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        sp.add_argument("--host", default="127.0.0.1")

    sp = sub.add_parser("serve", help="Chạy API FastAPI (POST /predict, GET /health)")
    add_model_args(sp)
    sp.add_argument("--port", type=int, default=8000)

    sp = sub.add_parser("demo", help="Chạy demo Gradio")
    add_model_args(sp)
    sp.add_argument("--port", type=int, default=7860)
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

        print(
            run_train(
                args.config,
                args.data_root,
                args.override,
                use_mlflow=not args.no_mlflow,
                skip_if_done=args.skip_if_done,
            )
        )
    elif args.command == "evaluate":
        from vehicledet.evaluate import run_evaluate

        print(
            run_evaluate(
                args.weights,
                args.split,
                args.part,
                args.data_root,
                args.imgsz,
                args.batch,
                args.device,
                args.workers,
                args.tag,
                args.final,
                args.allow_rerun,
            )
        )
    elif args.command == "export":
        from vehicledet.export import export_formats

        info = export_formats(
            args.weights,
            args.tag,
            args.split,
            tuple(args.formats),
            args.imgsz,
            data_root=args.data_root,
            device=args.device,
        )
        print(json.dumps(info["results"], ensure_ascii=False, indent=1))
    elif args.command == "benchmark":
        from vehicledet.benchmark import (
            BENCH_MARK_END,
            BENCH_MARK_START,
            render_benchmark_table,
            run_benchmark,
        )
        from vehicledet.experiments import replace_marked_block

        rows = run_benchmark(
            args.model_dir,
            args.split,
            args.device,
            args.imgsz,
            args.warmup,
            args.iters,
            None if args.map_part == "none" else args.map_part,
            args.final,
            map_batch=args.map_batch,
            only=args.only,
            data_root=args.data_root,
            isolate=not args.no_isolate,
            row_out=Path(args.row_out) if args.row_out else None,
            repeats=args.repeats,
        )
        if args.row_out:
            return 0
        table = render_benchmark_table(rows)
        print(table)
        if args.roadmap and not Path(args.roadmap).is_file():
            print(f"Bỏ qua cập nhật: không có file {args.roadmap} (file này giữ cục bộ, ngoài git)")
        elif args.roadmap:
            path = Path(args.roadmap)
            path.write_text(
                replace_marked_block(
                    path.read_text(encoding="utf-8"), table, BENCH_MARK_START, BENCH_MARK_END
                ),
                encoding="utf-8",
            )
    elif args.command == "analyze-errors":
        from vehicledet.error_analysis import run_error_analysis

        kwargs = {"out_dir": Path(args.out)} if args.out else {}
        summary = run_error_analysis(
            args.weights, args.split, args.conf, args.imgsz, args.device, args.data_root, **kwargs
        )
        print(json.dumps(summary["counts"], ensure_ascii=False, indent=1))
    elif args.command == "serve":
        import os

        import uvicorn

        os.environ.update(MODEL_PATH=args.model, BACKEND=args.backend, DEVICE=args.device)
        uvicorn.run("vehicledet.serve.app:app", host=args.host, port=args.port)
    elif args.command == "demo":
        from vehicledet.serve.demo import launch

        launch(args.model, args.backend, args.device, args.host, args.port)
    return 0
