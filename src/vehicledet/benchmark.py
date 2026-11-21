"""Benchmark tốc độ và độ chính xác: PyTorch FP32/FP16, ONNX Runtime, TensorRT FP16/INT8.

Đo qua cùng một đường Ultralytics `predict` cho mọi định dạng nên tiền/hậu xử lý giống nhau:
- `latency` (model-only) = `speed['inference']` của Ultralytics, ms cho một ảnh, batch 1;
- `e2e` = thời gian cả lời gọi predict (tiền xử lý + model + NMS), batch 1;
- `throughput` = ảnh/giây của cả lời gọi predict (end-to-end) với batch 1/4/8/16.
Định dạng không chạy được trên máy (thiếu TensorRT, onnxruntime-gpu, GPU...) được ghi là "chưa đo"
kèm lý do, không bao giờ ước đoán số.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vehicledet.data.splits import read_split
from vehicledet.data.yolo_layout import build_yolo_dataset
from vehicledet.paths import REPORTS_DIR, portable, resolve_data_root

log = logging.getLogger(__name__)

BATCHES = (1, 4, 8, 16)
BENCH_MARK_START = "<!-- BENCHMARK:START (sinh bởi `python -m vehicledet benchmark --roadmap`) -->"
BENCH_MARK_END = "<!-- BENCHMARK:END -->"


@dataclass(frozen=True)
class FormatSpec:
    key: str
    label: str
    precision: str
    kind: str  # 'pt' | 'onnx' | 'engine'
    filename: str
    quantize: int | None = None  # chỉ cho PyTorch: 16 = FP16


SPECS = (
    FormatSpec("pytorch_fp32", "PyTorch", "FP32", "pt", "best.pt"),
    FormatSpec("pytorch_fp16", "PyTorch", "FP16", "pt", "best.pt", 16),
    FormatSpec("onnx_fp32", "ONNX Runtime", "FP32", "onnx", "best.onnx"),
    FormatSpec("tensorrt_fp16", "TensorRT", "FP16", "engine", "best_fp16.engine"),
    FormatSpec("tensorrt_int8", "TensorRT", "INT8", "engine", "best_int8.engine"),
)

CSV_FIELDS = [
    "format",
    "precision",
    "device",
    "status",
    "latency_b1_p50_ms",
    "latency_b1_p95_ms",
    "e2e_b1_p50_ms",
    "e2e_b1_p95_ms",
    "throughput_b1",
    "throughput_b4",
    "throughput_b8",
    "throughput_b16",
    "vram_mb",
    "mAP50",
    "mAP50-95",
    "map_part",
    "file_mb",
    "note",
]


# ----- thống kê & tiện ích thuần (kiểm thử được không cần GPU) -----------------------------


def percentile(values: list[float], q: float) -> float:
    """Phân vị q (0–100) theo nội suy tuyến tính."""
    if not values:
        raise ValueError("Danh sách rỗng")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def latency_stats(ms: list[float]) -> dict:
    return {"p50": percentile(ms, 50), "p95": percentile(ms, 95), "mean": float(np.mean(ms))}


def throughput(batch: int, wall_ms: list[float]) -> float:
    """Ảnh/giây, tính từ thời gian trung vị của mỗi lời gọi (ít nhạy với giá trị ngoại lai)."""
    return batch * 1000.0 / percentile(wall_ms, 50)


def parse_gpu_memory(text: str, pid: int) -> float | None:
    """Phân tích đầu ra của nvidia-smi (`--query-compute-apps=pid,used_memory` dạng csv, không
    header, không đơn vị); trả về MiB của tiến trình `pid` (None nếu không thấy)."""
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def process_gpu_memory_mb(pid: int | None = None) -> float | None:
    """VRAM của riêng tiến trình này (gồm cả CUDA context), đo bằng nvidia-smi.

    Không dùng `torch.cuda.memory_allocated` vì không thấy bộ nhớ của ONNX Runtime/TensorRT.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_gpu_memory(out, pid or os.getpid())


def check_available(spec: FormatSpec, path: Path, device: str) -> str | None:
    """None nếu đo được; ngược lại trả lý do 'chưa đo'."""
    on_cuda = device != "cpu"
    if not path.exists():
        return f"chưa có file {path.name} (chạy `python -m vehicledet export` trước)"
    if spec.quantize == 16 and not on_cuda:
        return "FP16 trên CPU không có ý nghĩa"
    if spec.kind == "engine":
        if not on_cuda:
            return "TensorRT cần GPU"
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            return "chưa cài tensorrt (xem docs/TENSORRT_SETUP.md)"
    if spec.kind == "onnx" and on_cuda:
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                return (
                    "onnxruntime chưa có CUDAExecutionProvider (cần onnxruntime-gpu; "
                    f"hiện có {ort.get_available_providers()})"
                )
        except ImportError:
            return "chưa cài onnxruntime"
    return None


def env_report(imgsz: int, device: str, warmup: int, iters: int) -> dict:
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input": f"1x3x{imgsz}x{imgsz} (letterbox), batch 1/4/8/16",
        "device": device,
        "warmup_iters": warmup,
        "latency_iters": iters,
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        info["gpu_driver_memory"] = out
    except (OSError, subprocess.SubprocessError):
        info["gpu_driver_memory"] = None
    for mod in ("torch", "ultralytics", "onnxruntime", "tensorrt"):
        try:
            m = __import__(mod)
            info[mod] = m.__version__
        except ImportError:
            info[mod] = None
    try:
        import torch

        info["cuda_torch"] = torch.version.cuda
    except ImportError:
        info["cuda_torch"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime_providers"] = None
    return info


# ----- đo thật ------------------------------------------------------------------------------


def load_sample_images(data_root: Path, split: str, n: int = 16) -> list[np.ndarray]:
    """n ảnh BGR đầu tiên của tập VAL (chỉ để đo thời gian, không dùng nhãn)."""
    import cv2

    root = build_yolo_dataset(data_root, split).data_yaml.parent / "images" / "val"
    images = []
    for name in read_split(split)["val"][:n]:
        img = cv2.imread(str(root / name))
        if img is not None:
            images.append(img)
    if len(images) < max(BATCHES):
        raise RuntimeError(f"Chỉ đọc được {len(images)} ảnh mẫu, cần {max(BATCHES)}")
    return images


def _predict(model, imgs, imgsz, device, quantize):
    kwargs = dict(imgsz=imgsz, device=device, verbose=False, conf=0.25)
    if quantize:
        kwargs["quantize"] = quantize
    return model.predict(imgs, **kwargs)


def measure(model, images, imgsz, device, quantize, warmup: int, iters: int) -> dict:
    """Warm-up rồi đo latency batch 1 (model-only và e2e) và throughput các batch; theo dõi VRAM."""
    vram = []
    sample = lambda: vram.append(process_gpu_memory_mb()) if device != "cpu" else None  # noqa: E731

    for _ in range(warmup):
        _predict(model, images[:1], imgsz, device, quantize)
    sample()

    infer_ms, e2e_ms = [], []
    for i in range(iters):
        t = time.perf_counter()
        res = _predict(
            model, images[i % len(images) : i % len(images) + 1], imgsz, device, quantize
        )
        e2e_ms.append((time.perf_counter() - t) * 1000)
        infer_ms.append(res[0].speed["inference"])
    sample()

    out = {
        "latency_b1": latency_stats(infer_ms),
        "e2e_b1": latency_stats(e2e_ms),
        "throughput": {},
    }
    for b in BATCHES:
        batch = images[:b]
        for _ in range(max(3, warmup // 4)):
            _predict(model, batch, imgsz, device, quantize)
        walls = []
        for _ in range(max(10, iters // 4)):
            t = time.perf_counter()
            _predict(model, batch, imgsz, device, quantize)
            walls.append((time.perf_counter() - t) * 1000)
        out["throughput"][b] = throughput(b, walls)
        sample()
    vram = [v for v in vram if v is not None]
    out["vram_mb"] = max(vram) if vram else None
    return out


def measure_map(path: Path, data_yaml: Path, part: str, imgsz, device, quantize, batch: int):
    from ultralytics import YOLO

    kwargs = dict(
        data=str(data_yaml),
        split=part,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=False,
        verbose=False,
    )
    if quantize:
        kwargs["quantize"] = quantize
    m = YOLO(str(path), task="detect").val(**kwargs)
    return float(m.box.map50), float(m.box.map)


def empty_row(spec: FormatSpec, device: str, note: str) -> dict:
    row = {k: "" for k in CSV_FIELDS}
    row.update(
        format=spec.label, precision=spec.precision, device=device, status="chưa đo", note=note
    )
    return row


def benchmark_format(
    spec, path, images, imgsz, device, warmup, iters, data_yaml, map_part, map_batch
):
    from ultralytics import YOLO

    why = check_available(spec, path, device)
    if why:
        return empty_row(spec, device, why)
    row = empty_row(spec, device, "")
    try:
        model = YOLO(str(path), task="detect")
        m = measure(model, images, imgsz, device, spec.quantize, warmup, iters)
        row.update(
            status="đã đo",
            latency_b1_p50_ms=round(m["latency_b1"]["p50"], 3),
            latency_b1_p95_ms=round(m["latency_b1"]["p95"], 3),
            e2e_b1_p50_ms=round(m["e2e_b1"]["p50"], 3),
            e2e_b1_p95_ms=round(m["e2e_b1"]["p95"], 3),
            vram_mb=m["vram_mb"] if m["vram_mb"] is not None else "",
            file_mb=round(path.stat().st_size / 2**20, 2),
            **{f"throughput_b{b}": round(v, 2) for b, v in m["throughput"].items()},
        )
        del model
        if map_part:
            row["mAP50"], row["mAP50-95"] = (
                round(v, 4)
                for v in measure_map(
                    path, data_yaml, map_part, imgsz, device, spec.quantize, map_batch
                )
            )
            row["map_part"] = map_part
    except Exception as e:  # noqa: BLE001
        row.update(status="lỗi", note=f"{type(e).__name__}: {e}"[:300])
    return row


def run_benchmark(
    model_dir: str | Path,
    split: str,
    device: str = "0",
    imgsz: int = 640,
    warmup: int = 20,
    iters: int = 200,
    map_part: str | None = "val",
    final: bool = False,
    map_batch: int = 8,
    only: list[str] | None = None,
    data_root: str | None = None,
    out_dir: Path = REPORTS_DIR,
) -> list[dict]:
    """Chạy benchmark mọi định dạng trong model_dir; ghi reports/benchmark.csv + benchmark.json."""
    if map_part == "test" and not final:
        raise PermissionError(
            "mAP trên tập test chỉ dùng cho báo cáo cuối: thêm --final khi đã chốt model bằng val."
        )
    root = resolve_data_root(data_root)
    model_dir = Path(model_dir)
    images = load_sample_images(root, split)
    data_yaml = build_yolo_dataset(root, split).data_yaml

    rows = []
    for spec in SPECS:
        if only and spec.key not in only:
            continue
        log.info("Benchmark %s %s ...", spec.label, spec.precision)
        rows.append(
            benchmark_format(
                spec,
                model_dir / spec.filename,
                images,
                imgsz,
                device,
                warmup,
                iters,
                data_yaml,
                map_part,
                map_batch,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "env": env_report(imgsz, device, warmup, iters),
                "model_dir": portable(str(model_dir)),
                "split": split,
                "map_part": map_part,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def render_benchmark_table(rows: list[dict]) -> str:
    head = (
        "| Định dạng | Precision | Latency p50 (ms) | p95 (ms) | Throughput b16 (img/s) | "
        "VRAM (MB) | mAP50-95 | Ghi chú |\n|---|---|---|---|---|---|---|---|\n"
    )
    if not rows:
        return head + "| chưa đo | | | | | | | |\n"
    lines = []
    for r in rows:
        if r["status"] != "đã đo":
            lines.append(f"| {r['format']} | {r['precision']} | chưa đo | | | | | {r['note']} |")
            continue
        part = f" ({r['map_part']})" if r["map_part"] else ""
        map_cell = f"{r['mAP50-95']}{part}" if r["mAP50-95"] != "" else "chưa đo"
        lines.append(
            f"| {r['format']} | {r['precision']} | {r['latency_b1_p50_ms']} | "
            f"{r['latency_b1_p95_ms']} | {r['throughput_b16']} | "
            f"{r['vram_mb'] or 'n/a'} | {map_cell} | |"
        )
    return head + "\n".join(lines) + "\n"
