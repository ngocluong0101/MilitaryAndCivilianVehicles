"""So khớp pipeline ONNX tự viết (API/demo) với Ultralytics trên cùng ảnh val.

Dùng: python scripts/check_serve_parity.py --model exports/<tag>/best.onnx [--split v2] [--n 100]
Chỉ dùng ảnh VAL. Ghi reports/serve_parity.json: tỉ lệ bbox khớp (IoU > 0.9) theo khoảng điểm.
Lưu ý: tiền xử lý khác nhau chút (Pillow vs OpenCV resize) nên phát hiện điểm thấp có thể lệch.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vehicledet.data.splits import read_split
from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.paths import REPORTS_DIR, portable, resolve_data_root
from vehicledet.serve.inference import load_detector


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="v2")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--data-root")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "serve_parity.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    root = resolve_data_root(args.data_root)
    img_dir = build_yolo_dataset(root, args.split).data_yaml.parent / "images" / "val"
    mine, ref = load_detector(args.model, device="cpu"), YOLO(args.model, task="detect")

    rows = []  # (điểm ultralytics, IoU tốt nhất với pipeline tự viết, cùng lớp)
    for name in read_split(args.split)["val"][: args.n]:
        im = Image.open(img_dir / name).convert("RGB")
        dets = mine.predict(im, args.conf, args.iou).detections
        r = ref.predict(
            np.asarray(im)[:, :, ::-1],
            imgsz=mine.imgsz,
            conf=args.conf,
            iou=args.iou,
            device="cpu",
            verbose=False,
        )[0]
        for b, c, s in zip(
            r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy(), strict=True
        ):
            best = max(dets, key=lambda d: iou(d.box, b.tolist()), default=None)
            rows.append(
                (
                    float(s),
                    iou(best.box, b.tolist()) if best else 0.0,
                    bool(best and best.class_id == int(c)),
                )
            )
    arr = np.array(rows, dtype=float).reshape(-1, 3)
    buckets = {}
    for lo, hi in ((args.conf, 0.4), (0.4, 0.6), (0.6, 1.01)):
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        if m.sum():
            buckets[f"score_{lo:.2f}-{min(hi, 1.0):.2f}"] = {
                "n_ultralytics_boxes": int(m.sum()),
                "matched_iou_gt_0.9": round(float(np.mean(arr[m, 1] > 0.9)), 3),
                "matched_iou_gt_0.5": round(float(np.mean(arr[m, 1] > 0.5)), 3),
            }
    out = {
        "model": portable(args.model),
        "split": args.split,
        "part": "val",
        "n_images": args.n,
        "conf": args.conf,
        "iou": args.iou,
        "buckets": buckets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
