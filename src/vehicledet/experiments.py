"""Lưu kết quả mỗi thí nghiệm vào reports/experiments/<tên>/ và tổng hợp thành bảng so sánh."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from vehicledet.evaluate import TEST_RESULTS_PATH, write_result
from vehicledet.paths import EXPERIMENTS_DIR

MARK_START = "<!-- EXPERIMENTS:START (sinh bởi scripts/summarize_runs.py, đừng sửa tay) -->"
MARK_END = "<!-- EXPERIMENTS:END -->"


def run_info_path(name: str, root: Path = EXPERIMENTS_DIR) -> Path:
    return root / name / "run_info.json"


def best_epoch_from_results(results_csv: Path) -> tuple[int, int]:
    """(epoch tốt nhất, số epoch đã chạy) từ results.csv của Ultralytics.

    Epoch tốt nhất theo fitness của Ultralytics = 0.1*mAP50 + 0.9*mAP50-95 (đánh giá trên val).
    """
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    fitness = [
        0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]) for r in rows
    ]
    best = max(range(len(rows)), key=fitness.__getitem__)
    return int(float(rows[best]["epoch"])), len(rows)


def write_experiment_report(
    name: str,
    run_info: dict,
    val_result: dict,
    results_csv: Path | None,
    confusion_png: Path | None,
    root: Path = EXPERIMENTS_DIR,
) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    write_result(val_result, d / "val.json")
    if results_csv and results_csv.exists():
        shutil.copy(results_csv, d / "results.csv")
    if confusion_png and confusion_png.exists():
        shutil.copy(confusion_png, d / "confusion_matrix_val.png")
    # run_info.json ghi cuối cùng: có file này nghĩa là thí nghiệm đã hoàn tất
    (d / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return d


def weakest_class(per_class: dict) -> tuple[str, float]:
    name = min(per_class, key=lambda c: per_class[c]["AP50-95"])
    return name, per_class[name]["AP50-95"]


def _row(name: str, info: dict, result: dict, part: str) -> dict:
    weak, weak_ap = weakest_class(result["per_class"])
    cfg = info["config"]
    return {
        "name": name,
        "model": cfg["model"].removesuffix(".pt"),
        "split": f"{cfg['split']} ({part})",
        "imgsz": info["train_args"].get("imgsz"),
        "mAP50": result["mAP50"],
        "mAP50-95": result["mAP50-95"],
        "weakest_class": weak,
        "weakest_AP50-95": weak_ap,
        "train_minutes": info.get("train_minutes"),
        "epochs_ran": info.get("epochs_ran"),
        "mlflow_run_id": info.get("mlflow_run_id"),
        "part": part,
    }


def load_rows(root: Path = EXPERIMENTS_DIR, test_path: Path = TEST_RESULTS_PATH) -> list[dict]:
    rows, infos = [], {}
    for info_path in sorted(root.glob("*/run_info.json")):
        name = info_path.parent.name
        val_path = info_path.parent / "val.json"
        if not val_path.exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        infos[name] = info
        rows.append(_row(name, info, json.loads(val_path.read_text(encoding="utf-8")), "val"))
    if test_path.exists():  # dòng cuối: kết quả test của model được chọn
        res = json.loads(test_path.read_text(encoding="utf-8"))
        src = Path(res["weights"]).parent.parent.name
        if src in infos:
            row = _row(src, infos[src], res, "test")
            row["name"] = f"TEST: {src}"
            rows.append(row)
    return rows


def pick_best(rows: list[dict], split_prefix: str, imgsz: int = 640) -> dict | None:
    """Model tốt nhất theo mAP50-95 trên val, trong số các run cùng split và imgsz."""
    cand = [
        r
        for r in rows
        if r["part"] == "val" and r["split"].startswith(split_prefix) and r["imgsz"] == imgsz
    ]
    return max(cand, key=lambda r: r["mAP50-95"], default=None)


def _minutes(m: float | None) -> str:
    if m is None:
        return "–"
    if m >= 60:
        return f"{int(m // 60)} giờ {int(m % 60):02d} phút"
    return f"{m:.1f} phút" if m < 10 else f"{m:.0f} phút"


def render_table(rows: list[dict]) -> str:
    head = (
        "| # | Model | Split | imgsz | mAP50 | mAP50-95 | Lớp yếu nhất (AP50-95) | "
        "Thời gian train | Epoch | MLflow run |\n|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if not rows:
        return head + "| – | chưa chạy | | | | | | | | |\n"
    lines = []
    for r in rows:
        run = (r["mlflow_run_id"] or "–")[:8]
        lines.append(
            f"| {r['name']} | {r['model']} | {r['split']} | {r['imgsz']} | {r['mAP50']:.3f} | "
            f"{r['mAP50-95']:.3f} | {r['weakest_class']} ({r['weakest_AP50-95']:.3f}) | "
            f"{_minutes(r['train_minutes'])} | {r['epochs_ran'] or '–'} | `{run}` |"
        )
    return head + "\n".join(lines) + "\n"


def write_summary(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiments_summary.md").write_text(render_table(rows), encoding="utf-8")
    if rows:
        with open(out_dir / "experiments_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def replace_marked_block(text: str, block: str) -> str:
    """Thay nội dung giữa MARK_START và MARK_END (giữ nguyên hai dòng đánh dấu)."""
    if MARK_START not in text or MARK_END not in text:
        raise ValueError(f"File thiếu cặp đánh dấu {MARK_START!r} … {MARK_END!r}")
    head, rest = text.split(MARK_START, 1)
    _, tail = rest.split(MARK_END, 1)
    return f"{head}{MARK_START}\n{block}{MARK_END}{tail}"
