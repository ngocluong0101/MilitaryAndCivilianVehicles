"""Phân tích lỗi phát hiện trên tập VAL: nhầm lớp, bỏ sót, phát hiện thừa, định vị sai, nhãn đáng ngờ.

Quy tắc ghép (theo thứ tự điểm giảm dần, ngưỡng IoU 0,5):
- đúng lớp + IoU >= 0,5 với GT chưa được ghép            -> tp
- IoU >= 0,5 với GT cùng lớp đã được ghép                -> duplicate (dự đoán trùng)
- IoU >= 0,5 với GT khác lớp                             -> class_confusion (nhầm lớp)
- chỉ có GT cùng lớp với 0,1 <= IoU < 0,5                -> localization (định vị sai)
- chỉ có GT khác lớp với 0,1 <= IoU < 0,5                -> other_fp
- không chạm GT nào (IoU < 0,1)                          -> background_fp (phát hiện thừa)
GT không được ghép và không được giải thích bởi nhầm lớp/định vị -> miss (bỏ sót), kèm thẻ: small
(COCO), occluded (bị GT khác che > 30% diện tích), crowded (ảnh có >= 6 vật thể).
Không sửa nhãn: dự đoán điểm cao không khớp GT chỉ được ghi vào danh sách "nhãn đáng ngờ".
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vehicledet.data.labels import parse_yolo_txt
from vehicledet.data.splits import read_split
from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.paths import REPORTS_DIR, class_names, portable, resolve_data_root, txt_labels_dir

IOU_MATCH = 0.5
IOU_LOC = 0.1
SMALL_AREA = 32**2
CROWDED_N = 6
OCCLUDED_FRAC = 0.3
SUSPECT_FP_SCORE = 0.7  # dự đoán thừa điểm cao: có thể thiếu nhãn
SUSPECT_CONF_SCORE = 0.8  # nhầm lớp điểm cao: có thể nhãn sai

KINDS = ("tp", "class_confusion", "localization", "duplicate", "background_fp", "other_fp", "miss")
GALLERY_CATEGORIES = (
    "class_confusion",
    "miss_small",
    "miss_occluded",
    "miss_crowded",
    "miss_other",
    "background_fp",
    "duplicate",
    "localization",
    "suspect_label",
)
CATEGORY_TITLES = {
    "class_confusion": "Nhầm lớp",
    "miss_small": "Bỏ sót: vật thể nhỏ",
    "miss_occluded": "Bỏ sót: bị che/chồng",
    "miss_crowded": "Bỏ sót: ảnh nhiều vật thể",
    "miss_other": "Bỏ sót: không thuộc nhóm trên",
    "background_fp": "Phát hiện thừa (nền)",
    "duplicate": "Dự đoán trùng lặp",
    "localization": "Định vị sai (đúng lớp, IoU 0,1–0,5)",
    "suspect_label": "Nhãn đáng ngờ (dự đoán điểm cao không khớp nhãn)",
}


@dataclass(frozen=True)
class Box:
    cls: int
    xyxy: tuple[float, float, float, float]
    score: float = 1.0  # GT: 1.0


@dataclass
class ErrorEvent:
    image: str
    kind: str
    pred: Box | None = None
    gt: Box | None = None
    tags: list[str] = field(default_factory=list)  # chỉ cho miss
    iou: float = 0.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa mọi cặp (N,4) x (M,4) box xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _area(xyxy) -> float:
    return max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])


def _occlusion(gts: list[Box], i: int) -> float:
    """Tỉ lệ diện tích GT i bị các GT khác phủ lên (lấy giá trị lớn nhất)."""
    a = gts[i].xyxy
    area = _area(a)
    if area <= 0:
        return 0.0
    best = 0.0
    for j, other in enumerate(gts):
        if j == i:
            continue
        w = min(a[2], other.xyxy[2]) - max(a[0], other.xyxy[0])
        h = min(a[3], other.xyxy[3]) - max(a[1], other.xyxy[1])
        if w > 0 and h > 0:
            best = max(best, w * h / area)
    return best


def analyze_image(name: str, gts: list[Box], preds: list[Box]) -> list[ErrorEvent]:
    """Phân loại mọi dự đoán và mọi GT của một ảnh theo quy tắc ở đầu file."""
    events: list[ErrorEvent] = []
    g = np.array([b.xyxy for b in gts], dtype=float).reshape(-1, 4)
    order = sorted(range(len(preds)), key=lambda k: -preds[k].score)
    ious = iou_matrix(np.array([p.xyxy for p in preds], dtype=float).reshape(-1, 4), g)
    matched = [False] * len(gts)
    explained = [False] * len(gts)

    for k in order:
        p, row = preds[k], ious[k]
        same = [j for j in range(len(gts)) if gts[j].cls == p.cls]
        free = [j for j in same if not matched[j] and row[j] >= IOU_MATCH]
        if free:
            j = max(free, key=lambda j: row[j])
            matched[j] = True
            events.append(ErrorEvent(name, "tp", p, gts[j], iou=float(row[j])))
            continue
        hits = [j for j in range(len(gts)) if row[j] >= IOU_MATCH]
        if hits:
            j = max(hits, key=lambda j: row[j])
            if gts[j].cls == p.cls:
                events.append(ErrorEvent(name, "duplicate", p, gts[j], iou=float(row[j])))
            else:
                explained[j] = True
                events.append(ErrorEvent(name, "class_confusion", p, gts[j], iou=float(row[j])))
            continue
        loc_same = [j for j in same if row[j] >= IOU_LOC]
        if loc_same:
            j = max(loc_same, key=lambda j: row[j])
            explained[j] = True
            events.append(ErrorEvent(name, "localization", p, gts[j], iou=float(row[j])))
        elif any(row[j] >= IOU_LOC for j in range(len(gts))):
            j = max(range(len(gts)), key=lambda j: row[j])
            events.append(ErrorEvent(name, "other_fp", p, gts[j], iou=float(row[j])))
        else:
            events.append(ErrorEvent(name, "background_fp", p, None))

    for j, gt in enumerate(gts):
        if matched[j] or explained[j]:
            continue
        tags = []
        if _area(gt.xyxy) < SMALL_AREA:
            tags.append("small")
        if _occlusion(gts, j) > OCCLUDED_FRAC:
            tags.append("occluded")
        if len(gts) >= CROWDED_N:
            tags.append("crowded")
        events.append(ErrorEvent(name, "miss", None, gt, tags=tags))
    return events


def summarize(events: list[ErrorEvent], n_images: int, names: list[str]) -> dict:
    """Đếm theo loại lỗi, cặp nhầm lớp, thẻ bỏ sót, thống kê từng lớp và danh sách nhãn đáng ngờ."""
    kinds = Counter(e.kind for e in events)
    pairs: Counter[tuple[str, str]] = Counter()
    miss_tags: Counter[str] = Counter()
    per_class: dict[str, Counter] = {n: Counter() for n in names}
    suspect: list[dict] = []
    for e in events:
        if e.kind == "tp":
            per_class[names[e.gt.cls]]["tp"] += 1
        elif e.kind == "class_confusion":
            pairs[(names[e.gt.cls], names[e.pred.cls])] += 1
            per_class[names[e.gt.cls]]["confused_as_other"] += 1
            per_class[names[e.pred.cls]]["predicted_wrongly"] += 1
            if e.pred.score >= SUSPECT_CONF_SCORE:
                suspect.append(_suspect(e, "high-confidence class disagreement", names))
        elif e.kind == "miss":
            per_class[names[e.gt.cls]]["miss"] += 1
            for t in e.tags or ["other"]:
                miss_tags[t] += 1
        elif e.kind in ("background_fp", "other_fp", "duplicate", "localization"):
            per_class[names[e.pred.cls]][e.kind] += 1
            if e.kind == "background_fp" and e.pred.score >= SUSPECT_FP_SCORE:
                suspect.append(_suspect(e, "high-confidence detection without label", names))
    n_gt = kinds["tp"] + kinds["class_confusion"] + kinds["localization"] + kinds["miss"]
    return {
        "n_images": n_images,
        "n_gt_boxes_approx": n_gt,  # tp + nhầm lớp + định vị + bỏ sót (một GT có thể xuất hiện >1 lần)
        "counts": {k: kinds.get(k, 0) for k in KINDS},
        "confusion_pairs": [
            {"true": t, "predicted": p, "count": c} for (t, p), c in pairs.most_common()
        ],
        "miss_tags": dict(miss_tags),
        "per_class": {n: dict(c) for n, c in per_class.items()},
        "suspect_labels": sorted(suspect, key=lambda s: -s["score"]),
    }


def _suspect(e: ErrorEvent, reason: str, names: list[str]) -> dict:
    return {
        "image": e.image,
        "reason": reason,
        "score": round(e.pred.score, 3),
        "predicted": names[e.pred.cls],
        "labeled": names[e.gt.cls] if e.gt else None,
    }


# ----- gallery ----------------------------------------------------------------------------------


def _category_of(e: ErrorEvent) -> list[str]:
    cats = []
    if e.kind == "miss":
        cats.append(
            "miss_small"
            if "small" in e.tags
            else "miss_occluded"
            if "occluded" in e.tags
            else "miss_crowded"
            if "crowded" in e.tags
            else "miss_other"
        )
    elif e.kind in ("class_confusion", "background_fp", "duplicate", "localization"):
        cats.append(e.kind)
        if (e.kind == "background_fp" and e.pred.score >= SUSPECT_FP_SCORE) or (
            e.kind == "class_confusion" and e.pred.score >= SUSPECT_CONF_SCORE
        ):
            cats.append("suspect_label")
    return cats


def select_examples(events: list[ErrorEvent], per_category: int = 6, seed: int = 0) -> dict:
    """Chọn ảnh tiêu biểu mỗi nhóm: lỗi do dự đoán -> điểm cao nhất; bỏ sót -> ngẫu nhiên (seed)."""
    by_cat: dict[str, list[ErrorEvent]] = defaultdict(list)
    for e in events:
        for c in _category_of(e):
            by_cat[c].append(e)
    rng = random.Random(seed)
    chosen: dict[str, list[ErrorEvent]] = {}
    for cat in GALLERY_CATEGORIES:
        items = by_cat.get(cat, [])
        if cat.startswith("miss"):
            items = sorted(items, key=lambda e: (e.image, e.gt.xyxy))
            rng.shuffle(items)
        else:
            items = sorted(items, key=lambda e: (-e.pred.score, e.image))
        seen, picked = set(), []
        for e in items:  # mỗi ảnh chỉ một lần trong cùng nhóm
            if e.image not in seen:
                seen.add(e.image)
                picked.append(e)
            if len(picked) == per_category:
                break
        chosen[cat] = picked
    return chosen


def draw_tile(
    img: Image.Image,
    gts: list[Box],
    preds: list[Box],
    names: list[str],
    caption: str,
    height: int = 300,
) -> Image.Image:
    """GT màu xanh lá (nét liền), dự đoán màu đỏ; ảnh thu nhỏ về chiều cao `height`."""
    scale = height / img.height
    im = img.convert("RGB").resize((max(1, round(img.width * scale)), height))
    d = ImageDraw.Draw(im)

    def rect(b: Box, color, label):
        x1, y1, x2, y2 = (v * scale for v in b.xyxy)
        d.rectangle((x1, y1, x2, y2), outline=color, width=2)
        d.text((x1 + 2, max(0, y1 - 11)), label, fill=color)

    for b in gts:
        rect(b, (0, 170, 0), f"GT {names[b.cls]}")
    for b in preds:
        rect(b, (220, 0, 0), f"{names[b.cls]} {b.score:.2f}")
    canvas = Image.new("RGB", (im.width, height + 14), "white")
    canvas.paste(im, (0, 14))
    ImageDraw.Draw(canvas).text((2, 1), caption, fill="black")
    return canvas


def save_gallery(chosen, images: dict, gts: dict, preds: dict, names, out_dir: Path) -> dict:
    """Mỗi nhóm một ảnh ghép JPEG nhỏ (3 cột). Trả về {nhóm: [tên ảnh]}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    listing = {}
    for cat, events in chosen.items():
        listing[cat] = [e.image for e in events]
        if not events:
            continue
        tiles = []
        for e in events:
            with Image.open(images[e.image]) as im:
                cap = f"{e.image[:34]} ({e.kind})"
                tiles.append(draw_tile(im, gts[e.image], preds[e.image], names, cap))
        cols = 3
        rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
        w = max(sum(t.width for t in r) for r in rows)
        canvas = Image.new("RGB", (w, sum(max(t.height for t in r) for r in rows)), "white")
        y = 0
        for r in rows:
            x = 0
            for t in r:
                canvas.paste(t, (x, y))
                x += t.width
            y += max(t.height for t in r)
        canvas.save(out_dir / f"{cat}.jpg", quality=70)
    return listing


# ----- chạy trên dữ liệu thật ---------------------------------------------------------------------


def load_ground_truth(data_root: Path, split: str, part: str = "val"):
    """GT (pixel) và đường dẫn ảnh của một phần split."""
    names_files = read_split(split)[part]
    img_dir = build_yolo_dataset(data_root, split).data_yaml.parent / "images" / part
    lbl_dir = txt_labels_dir(data_root)
    n_cls = len(class_names())
    gts, paths, sizes = {}, {}, {}
    for name in names_files:
        paths[name] = img_dir / name
        with Image.open(paths[name]) as im:
            w, h = im.size
        boxes, _ = parse_yolo_txt(lbl_dir / (Path(name).stem + ".txt"), n_cls, (w, h))
        gts[name] = [
            Box(
                b.cls,
                (
                    (b.xc - b.w / 2) * w,
                    (b.yc - b.h / 2) * h,
                    (b.xc + b.w / 2) * w,
                    (b.yc + b.h / 2) * h,
                ),
            )
            for b in boxes
        ]
        sizes[name] = (w, h)
    return gts, paths, sizes


def predict_images(weights: str, paths: dict, conf: float, imgsz: int, device: str | None):
    """Dự đoán trên danh sách ảnh bằng Ultralytics (iou NMS 0,7)."""
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    out = {}
    names = list(paths)
    for i in range(0, len(names), 32):
        batch = names[i : i + 32]
        kwargs = {"conf": conf, "iou": 0.7, "imgsz": imgsz, "verbose": False}
        if device:
            kwargs["device"] = device
        for name, r in zip(
            batch, model.predict([str(paths[n]) for n in batch], **kwargs), strict=True
        ):
            out[name] = [
                Box(int(c), tuple(float(v) for v in b), float(s))
                for b, s, c in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                    strict=True,
                )
            ]
    return out


def render_markdown(summary: dict, listing: dict, params: dict) -> str:
    c, n = summary["counts"], summary["n_images"]
    total_pred = (
        c["tp"]
        + c["class_confusion"]
        + c["localization"]
        + c["duplicate"]
        + c["background_fp"]
        + c["other_fp"]
    )
    L = [
        "# Phân tích lỗi (tập val)",
        "",
        f"Sinh bởi `python -m vehicledet analyze-errors`. Tham số: {json.dumps(params, ensure_ascii=False)}",
        "",
        f"- Ảnh phân tích: **{n}**; số dự đoán: **{total_pred}**; "
        f"GT (xấp xỉ): **{summary['n_gt_boxes_approx']}**.",
        '- Chỉ dùng tập **val** (không dùng test). Không sửa nhãn nào; mục "nhãn đáng ngờ" chỉ ghi nhận.',
        "",
        "## 1. Số lượng theo loại lỗi",
        "",
        "| Loại | Số lượng | Giải thích |",
        "|---|---|---|",
    ]
    expl = {
        "tp": "đúng lớp, IoU ≥ 0,5",
        "class_confusion": "IoU ≥ 0,5 nhưng sai lớp",
        "localization": "đúng lớp nhưng IoU 0,1–0,5",
        "duplicate": "trùng với GT đã được ghép",
        "background_fp": "dự đoán không chạm GT nào (nền)",
        "other_fp": "chạm GT khác lớp ở mức IoU thấp",
        "miss": "GT không được phát hiện (bỏ sót)",
    }
    L += [f"| {k} | {c[k]:,} | {expl[k]} |" for k in KINDS]

    L += ["", "## 2. Nhầm lớp (GT → dự đoán)", ""]
    if summary["confusion_pairs"]:
        L += ["| GT | Dự đoán | Số lần |", "|---|---|---|"]
        L += [
            f"| {p['true']} | {p['predicted']} | {p['count']} |"
            for p in summary["confusion_pairs"][:10]
        ]
    else:
        L += ["Không có."]

    L += [
        "",
        "## 3. Bỏ sót (theo thẻ; một GT có thể có nhiều thẻ)",
        "",
        "| Thẻ | Số GT |",
        "|---|---|",
    ]
    L += [f"| {t} | {v} |" for t, v in sorted(summary["miss_tags"].items(), key=lambda x: -x[1])]

    L += [
        "",
        "## 4. Theo từng lớp",
        "",
        "| Lớp | tp | bị nhầm sang lớp khác | bỏ sót | dự đoán nhầm vào lớp này | thừa (nền) |",
        "|---|---|---|---|---|---|",
    ]
    for name, v in summary["per_class"].items():
        L.append(
            f"| {name} | {v.get('tp', 0)} | {v.get('confused_as_other', 0)} | {v.get('miss', 0)} | "
            f"{v.get('predicted_wrongly', 0)} | {v.get('background_fp', 0)} |"
        )

    L += ["", "## 5. Nhãn đáng ngờ (chỉ ghi nhận, KHÔNG sửa)", ""]
    sl = summary["suspect_labels"]
    L += [
        f"Số ca: **{len(sl)}** (dự đoán điểm ≥ {SUSPECT_FP_SCORE} không có nhãn, hoặc ≥ {SUSPECT_CONF_SCORE} "
        "nhưng khác lớp nhãn). Cần người kiểm tra tay trước khi kết luận nhãn sai."
    ]
    L += [
        f"- `{s['image']}`: dự đoán `{s['predicted']}` ({s['score']}); nhãn: `{s['labeled']}` — {s['reason']}"
        for s in sl[:15]
    ]

    L += ["", "## 6. Gallery (xanh = nhãn thật, đỏ = dự đoán)", ""]
    for cat in GALLERY_CATEGORIES:
        if listing.get(cat):
            L += [
                f"### {CATEGORY_TITLES[cat]}",
                "",
                f"![{cat}]({cat}.jpg)",
                "",
                "Ảnh: " + ", ".join(f"`{x}`" for x in listing[cat]),
                "",
            ]
    return "\n".join(L) + "\n"


def run_error_analysis(
    weights: str,
    split: str = "v2",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    data_root: str | None = None,
    out_dir: Path = REPORTS_DIR / "error_analysis",
    seed: int = 0,
) -> dict:
    root = resolve_data_root(data_root)
    names = class_names()
    gts, paths, _ = load_ground_truth(root, split, "val")
    preds = predict_images(weights, paths, conf, imgsz, device)
    events = [e for n in gts for e in analyze_image(n, gts[n], preds.get(n, []))]
    summary = summarize(events, len(gts), names)
    listing = save_gallery(select_examples(events, seed=seed), paths, gts, preds, names, out_dir)
    params = {
        "weights": portable(str(weights)),
        "split": split,
        "part": "val",
        "conf": conf,
        "iou_match": IOU_MATCH,
        "imgsz": imgsz,
        "seed": seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis.json").write_text(
        json.dumps({"params": params, **summary, "gallery": listing}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "error_analysis.md").write_text(
        render_markdown(summary, listing, params), encoding="utf-8"
    )
    return summary
