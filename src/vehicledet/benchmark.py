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
import statistics
import subprocess
import sys
import tempfile
import threading
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
    "repeats",
    "latency_b1_p50_ms",
    "latency_b1_p50_min_ms",
    "latency_b1_p50_max_ms",
    "latency_b1_p95_ms",
    "e2e_b1_p50_ms",
    "e2e_b1_p95_ms",
    "throughput_b1",
    "throughput_b4",
    "throughput_b8",
    "throughput_b16",
    "throughput_b16_min",
    "throughput_b16_max",
    "vram_mb",
    "sm_clock_mhz_median",
    "sm_clock_mhz_max",
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


def _ort_session_providers(path: Path) -> list[str]:
    """Provider mà một session ONNX Runtime THỰC SỰ dùng (không chỉ provider được biên dịch sẵn).

    ONNX Runtime lặng lẽ rơi về CPU nếu không nạp được thư viện CUDA, nên phải tạo session thật để
    kiểm tra; `preload_dlls()` nạp CUDA/cuDNN cài bằng pip (gói nvidia-*) trước khi tạo session.
    """
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    sess = ort.InferenceSession(
        str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return sess.get_providers()


class ClockSampler:
    """Lấy mẫu xung nhịp SM của GPU (NVML) khi đo; GPU laptop đổi xung theo tải/nhiệt/công suất.

    Không có NVML/GPU thì không ghi gì (stats() trả về None)."""

    def __init__(self, interval: float = 0.1):
        self.interval, self.samples = interval, []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self._stop.is_set():
                self.samples.append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
                time.sleep(self.interval)
        except Exception:  # noqa: BLE001 - không có NVML: bỏ qua
            return

    def __enter__(self) -> ClockSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def stats(self) -> dict | None:
        if not self.samples:
            return None
        return {"median": statistics.median(self.samples), "max": max(self.samples)}


_MEDIAN_FIELDS = (
    "latency_b1_p50_ms",
    "latency_b1_p95_ms",
    "e2e_b1_p50_ms",
    "e2e_b1_p95_ms",
    "throughput_b1",
    "throughput_b4",
    "throughput_b8",
    "throughput_b16",
    "vram_mb",
    "sm_clock_mhz_median",
)


def aggregate_rows(rows: list[dict]) -> dict:
    """Gộp nhiều lần đo (vòng lặp) của cùng một định dạng: trung vị + khoảng min–max.

    Lần đo không thành công bị bỏ qua; nếu không lần nào thành công thì trả về dòng đầu tiên.
    mAP và map_part lấy từ lần đo đầu tiên có giá trị.
    """
    ok = [r for r in rows if r["status"] == "đã đo"]
    if not ok:
        return rows[0]
    out = dict(ok[0])
    out["repeats"] = len(ok)
    for f in _MEDIAN_FIELDS:
        vals = [float(r[f]) for r in ok if r.get(f) not in ("", None)]
        out[f] = round(statistics.median(vals), 3) if vals else ""
    p50 = [float(r["latency_b1_p50_ms"]) for r in ok]
    thr = [float(r["throughput_b16"]) for r in ok]
    out["latency_b1_p50_min_ms"], out["latency_b1_p50_max_ms"] = min(p50), max(p50)
    out["throughput_b16_min"], out["throughput_b16_max"] = min(thr), max(thr)
    clocks = [
        float(r["sm_clock_mhz_max"]) for r in ok if r.get("sm_clock_mhz_max") not in ("", None)
    ]
    out["sm_clock_mhz_max"] = max(clocks) if clocks else ""
    for f in ("mAP50", "mAP50-95", "map_part"):
        out[f] = next((r[f] for r in ok if r.get(f) not in ("", None)), "")
    return out


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
            providers = _ort_session_providers(path)
        except Exception as e:  # noqa: BLE001
            return f"không tạo được session ONNX Runtime: {type(e).__name__}: {e}"[:300]
        if "CUDAExecutionProvider" not in providers:
            return (
                "ONNX Runtime không nạp được CUDAExecutionProvider (session chỉ dùng "
                f"{providers}; thiếu thư viện CUDA/cuDNN khớp với onnxruntime-gpu?)"
            )
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


def letterbox_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize giữ tỉ lệ rồi đệm xám cho vuông size×size (ảnh BGR)."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = max(1, round(w * r)), max(1, round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def load_sample_images(data_root: Path, split: str, n: int = 16, imgsz: int = 640):
    """n ảnh BGR đầu tiên của tập VAL, đã letterbox về đúng imgsz×imgsz (chỉ để đo thời gian).

    Dùng một kích thước cố định để model đầu vào động (ONNX/TensorRT) không phải tối ưu lại theo
    từng hình dạng ảnh, nhờ vậy mọi định dạng được đo trên cùng input 1×3×imgsz×imgsz.
    """
    import cv2

    root = build_yolo_dataset(data_root, split).data_yaml.parent / "images" / "val"
    images = []
    for name in read_split(split)["val"][:n]:
        img = cv2.imread(str(root / name))
        if img is not None:
            images.append(letterbox_square(img, imgsz))
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

    for i in range(warmup):
        j = i % len(images)
        _predict(model, images[j : j + 1], imgsz, device, quantize)
    sample()

    sampler = ClockSampler()
    sampler.__enter__()
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
    sampler.__exit__(None, None, None)
    out["clock"] = sampler.stats()
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
            sm_clock_mhz_median=m["clock"]["median"] if m.get("clock") else "",
            sm_clock_mhz_max=m["clock"]["max"] if m.get("clock") else "",
            repeats=1,
            latency_b1_p50_min_ms=round(m["latency_b1"]["p50"], 3),
            latency_b1_p50_max_ms=round(m["latency_b1"]["p50"], 3),
            throughput_b16_min=round(m["throughput"][16], 2),
            throughput_b16_max=round(m["throughput"][16], 2),
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


def _run_isolated(
    spec: FormatSpec,
    model_dir: Path,
    split: str,
    device: str,
    imgsz: int,
    warmup: int,
    iters: int,
    map_part: str | None,
    final: bool,
    map_batch: int,
    data_root: str | None,
) -> dict:
    """Đo MỘT định dạng trong tiến trình riêng để VRAM và cache CUDA không lẫn từ định dạng khác."""
    with tempfile.TemporaryDirectory() as tmp:
        row_file = Path(tmp) / "row.json"
        cmd = [
            sys.executable,
            "-m",
            "vehicledet",
            "benchmark",
            "--model-dir",
            str(model_dir),
            "--split",
            split,
            "--device",
            device,
            "--imgsz",
            str(imgsz),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--map-part",
            map_part or "none",
            "--map-batch",
            str(map_batch),
            "--only",
            spec.key,
            "--row-out",
            str(row_file),
        ]
        if final:
            cmd.append("--final")
        if data_root:
            cmd += ["--data-root", str(data_root)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if row_file.exists():
            return json.loads(row_file.read_text(encoding="utf-8"))
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
        return {
            **empty_row(spec, device, ""),
            "status": "lỗi",
            "note": f"tiến trình con thoát mã {proc.returncode}: {' '.join(tail)}"[:300],
        }


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
    isolate: bool = True,
    row_out: Path | None = None,
    repeats: int = 3,
) -> list[dict]:
    """Chạy benchmark mọi định dạng trong model_dir; ghi reports/benchmark.csv + benchmark.json.

    isolate=True (mặc định): mỗi lần đo của mỗi định dạng chạy trong một tiến trình riêng, nên VRAM
    là của riêng định dạng đó. Đo lặp `repeats` vòng xen kẽ các định dạng (round-robin) để trôi xung
    nhịp GPU không dồn vào một định dạng; báo cáo trung vị và khoảng min–max. mAP chỉ đo ở vòng đầu.
    `row_out` (nội bộ, cho tiến trình con): chỉ ghi dòng kết quả ra file.
    """
    if map_part == "test" and not final:
        raise PermissionError(
            "mAP trên tập test chỉ dùng cho báo cáo cuối: thêm --final khi đã chốt model bằng val."
        )
    specs = [s for s in SPECS if not only or s.key in only]
    model_dir = Path(model_dir)
    rows = []
    raw_rounds: dict[str, list[dict]] = {}
    if isolate and row_out is None and len(specs) > 1:
        per_spec: dict[str, list[dict]] = {spec.key: [] for spec in specs}
        for rnd in range(max(1, repeats)):
            for spec in specs:
                log.info(
                    "Benchmark %s %s (vòng %d, tiến trình riêng) ...",
                    spec.label,
                    spec.precision,
                    rnd + 1,
                )
                per_spec[spec.key].append(
                    _run_isolated(
                        spec,
                        model_dir,
                        split,
                        device,
                        imgsz,
                        warmup,
                        iters,
                        map_part if rnd == 0 else None,
                        final,
                        map_batch,
                        data_root,
                    )
                )
        rows = [aggregate_rows(per_spec[spec.key]) for spec in specs]
        raw_rounds = per_spec
    else:
        root = resolve_data_root(data_root)
        images = load_sample_images(root, split, imgsz=imgsz)
        data_yaml = build_yolo_dataset(root, split).data_yaml
        for spec in specs:
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

    if row_out is not None:
        row_out.write_text(json.dumps(rows[0], ensure_ascii=False), encoding="utf-8")
        return rows

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
                "isolated_processes": isolate,
                "repeats": repeats,
                "rows": rows,
                "rounds": raw_rounds,
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
        "| Định dạng | Precision | Latency p50 (ms): trung vị [min–max] | Throughput b16 (img/s): "
        "trung vị [min–max] | VRAM (MB) | mAP50-95 | SM MHz (trung vị) | Ghi chú |\n"
        "|---|---|---|---|---|---|---|---|\n"
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
        lat = (
            f"{r['latency_b1_p50_ms']} [{r['latency_b1_p50_min_ms']}–{r['latency_b1_p50_max_ms']}]"
        )
        thr = f"{r['throughput_b16']} [{r['throughput_b16_min']}–{r['throughput_b16_max']}]"
        lines.append(
            f"| {r['format']} | {r['precision']} | {lat} | {thr} | {r['vram_mb'] or 'n/a'} | "
            f"{map_cell} | {r['sm_clock_mhz_median'] or 'n/a'} | n={r['repeats']} vòng |"
        )
    return head + "\n".join(lines) + "\n"
