"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

from pathlib import Path

import pytest

from vehicledet import benchmark as bm


def spec(key):
    return next(s for s in bm.SPECS if s.key == key)


def test_percentile_and_stats():
    assert bm.percentile([1, 2, 3, 4, 5], 50) == 3
    assert bm.percentile([10.0], 95) == 10.0
    stats = bm.latency_stats(list(range(1, 101)))
    assert stats["p50"] == pytest.approx(50.5) and stats["p95"] == pytest.approx(95.05)
    with pytest.raises(ValueError):
        bm.percentile([], 50)


def test_throughput_uses_median_wall_time():
    # batch 16, trung vị 100 ms -> 160 ảnh/s; giá trị ngoại lai 5000 ms không ảnh hưởng
    assert bm.throughput(16, [100, 100, 100, 5000, 100]) == pytest.approx(160)


def test_parse_gpu_memory():
    out = "1234, 512\n4321, 1024\n"
    assert bm.parse_gpu_memory(out, 4321) == 1024
    assert bm.parse_gpu_memory(out, 9) is None
    assert bm.parse_gpu_memory("", 1) is None
    assert bm.parse_gpu_memory("1234, [N/A]", 1234) is None


def test_check_available_missing_file(tmp_path):
    why = bm.check_available(spec("onnx_fp32"), tmp_path / "best.onnx", "cpu")
    assert "best.onnx" in why and "export" in why


def test_check_available_rules(tmp_path, monkeypatch):
    f = tmp_path / "x"
    f.write_bytes(b"0")
    assert "FP16" in bm.check_available(spec("pytorch_fp16"), f, "cpu")
    assert bm.check_available(spec("pytorch_fp16"), f, "0") is None
    assert bm.check_available(spec("pytorch_fp32"), f, "cpu") is None
    assert "GPU" in bm.check_available(spec("tensorrt_fp16"), f, "cpu")

    # ONNX Runtime báo có CUDA ở mức biên dịch nhưng session thật rơi về CPU -> phải "chưa đo"
    monkeypatch.setattr(bm, "_ort_session_providers", lambda p: ["CPUExecutionProvider"])
    assert "CUDAExecutionProvider" in bm.check_available(spec("onnx_fp32"), f, "0")
    monkeypatch.setattr(
        bm, "_ort_session_providers", lambda p: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert bm.check_available(spec("onnx_fp32"), f, "0") is None
    assert (
        bm.check_available(spec("onnx_fp32"), f, "cpu") is None
    )  # CPU: không cần kiểm tra provider

    def boom(p):
        raise RuntimeError("model hỏng")

    monkeypatch.setattr(bm, "_ort_session_providers", boom)
    assert "model hỏng" in bm.check_available(spec("onnx_fp32"), f, "0")


def test_check_available_tensorrt_not_installed(tmp_path, monkeypatch):
    f = tmp_path / "x"
    f.write_bytes(b"0")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "tensorrt":
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert "tensorrt" in bm.check_available(spec("tensorrt_int8"), f, "0")


def test_empty_row_has_all_csv_fields():
    row = bm.empty_row(spec("tensorrt_int8"), "0", "lý do")
    assert list(row) == bm.CSV_FIELDS
    assert row["status"] == "chưa đo" and row["note"] == "lý do" and row["latency_b1_p50_ms"] == ""


def measured(key, **kw):
    row = {
        **bm.empty_row(spec(key), "0", ""),
        "status": "đã đo",
        "repeats": 1,
        "latency_b1_p50_ms": 3.0,
        "latency_b1_p95_ms": 3.5,
        "latency_b1_p50_min_ms": 3.0,
        "latency_b1_p50_max_ms": 3.0,
        "e2e_b1_p50_ms": 4.0,
        "e2e_b1_p95_ms": 4.5,
        "throughput_b1": 200.0,
        "throughput_b4": 300.0,
        "throughput_b8": 350.0,
        "throughput_b16": 400.0,
        "throughput_b16_min": 400.0,
        "throughput_b16_max": 400.0,
        "vram_mb": 1000.0,
        "sm_clock_mhz_median": 780.0,
        "sm_clock_mhz_max": 780.0,
    }
    row.update(kw)
    return row


def test_aggregate_rows_median_range_and_map_from_first_round():
    r1 = measured(
        "onnx_fp32",
        latency_b1_p50_ms=3.2,
        throughput_b16=400.0,
        sm_clock_mhz_median=780.0,
        sm_clock_mhz_max=780.0,
        **{"mAP50-95": 0.53, "mAP50": 0.84, "map_part": "test"},
    )
    r2 = measured(
        "onnx_fp32",
        latency_b1_p50_ms=1.1,
        throughput_b16=650.0,
        sm_clock_mhz_median=2400.0,
        sm_clock_mhz_max=2490.0,
    )
    r3 = measured(
        "onnx_fp32",
        latency_b1_p50_ms=3.1,
        throughput_b16=410.0,
        sm_clock_mhz_median=780.0,
        sm_clock_mhz_max=900.0,
    )
    agg = bm.aggregate_rows([r1, r2, r3])
    assert agg["repeats"] == 3
    assert agg["latency_b1_p50_ms"] == 3.1  # trung vị, không bị giá trị 1.1 kéo
    assert (agg["latency_b1_p50_min_ms"], agg["latency_b1_p50_max_ms"]) == (1.1, 3.2)
    assert (agg["throughput_b16_min"], agg["throughput_b16_max"]) == (400.0, 650.0)
    assert agg["throughput_b16"] == 410.0
    assert agg["sm_clock_mhz_max"] == 2490.0
    assert agg["mAP50-95"] == 0.53 and agg["map_part"] == "test"  # chỉ vòng đầu có mAP


def test_aggregate_rows_skips_failed_rounds_and_handles_all_failed():
    ok = measured("onnx_fp32")
    bad = {**bm.empty_row(spec("onnx_fp32"), "0", "lỗi gì đó"), "status": "lỗi"}
    assert bm.aggregate_rows([bad, ok, bad])["repeats"] == 1
    assert bm.aggregate_rows([bad, bad])["status"] == "lỗi"
    skipped = bm.empty_row(spec("tensorrt_int8"), "0", "chưa cài tensorrt")
    assert bm.aggregate_rows([skipped, skipped])["note"] == "chưa cài tensorrt"


def test_render_table_marks_unmeasured_without_numbers():
    row = measured(
        "onnx_fp32",
        latency_b1_p50_ms=3.1,
        latency_b1_p50_min_ms=1.1,
        latency_b1_p50_max_ms=3.2,
        throughput_b16=410.0,
        throughput_b16_min=400.0,
        throughput_b16_max=650.0,
        repeats=3,
        **{"mAP50-95": 0.55, "map_part": "val"},
    )
    skipped = bm.empty_row(spec("tensorrt_fp16"), "0", "chưa cài tensorrt")
    table = bm.render_benchmark_table([row, skipped])
    assert (
        "| ONNX Runtime | FP32 | 3.1 [1.1–3.2] | 410.0 [400.0–650.0] | 1000.0 | 0.55 (val) "
        "| 780.0 |" in table
    )
    assert "n=3 vòng" in table
    assert "| TensorRT | FP16 | chưa đo | | | | | chưa cài tensorrt |" in table
    assert "chưa đo" in bm.render_benchmark_table([])


def test_clock_sampler_never_crashes_without_gpu(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_nvml(name, *a, **kw):
        if name == "pynvml":
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_nvml)
    with bm.ClockSampler(interval=0.01) as sampler:
        pass
    assert sampler.stats() is None


def test_test_split_map_requires_final_flag(tmp_path):
    with pytest.raises(PermissionError, match="--final"):
        bm.run_benchmark(Path(tmp_path), "v2", map_part="test", final=False)


def test_letterbox_square_shape_and_padding():
    import numpy as np

    img = np.full((100, 200, 3), 255, dtype=np.uint8)  # rộng gấp đôi cao
    out = bm.letterbox_square(img, 64)
    assert out.shape == (64, 64, 3)
    assert out[0, 0, 0] == 114 and out[32, 32, 0] == 255  # viền xám, giữa là ảnh


def test_isolated_runs_one_subprocess_per_format_and_round(monkeypatch, tmp_path):
    """Mỗi định dạng đo ở tiến trình con riêng, lặp `repeats` vòng xen kẽ; mAP chỉ ở vòng đầu."""
    import json
    import subprocess
    import types

    calls = []

    def fake_run(cmd, *args, **kwargs):
        if "--row-out" not in cmd:  # lệnh khác (nvidia-smi trong env_report): coi như không có
            raise OSError("not available")
        calls.append(cmd)
        key = cmd[cmd.index("--only") + 1]
        row = measured(key, vram_mb=111.0)
        open(cmd[cmd.index("--row-out") + 1], "w").write(json.dumps(row))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rows = bm.run_benchmark(tmp_path, "v2", map_part="val", out_dir=tmp_path / "out", repeats=2)
    assert len(calls) == 2 * len(bm.SPECS) and all("--row-out" in c for c in calls)
    # xen kẽ: vòng 1 qua mọi định dạng rồi mới sang vòng 2
    order = [c[c.index("--only") + 1] for c in calls]
    assert order == [s.key for s in bm.SPECS] * 2
    # mAP (map-part) chỉ truyền ở vòng đầu
    parts = [c[c.index("--map-part") + 1] for c in calls]
    assert parts == ["val"] * len(bm.SPECS) + ["none"] * len(bm.SPECS)
    assert [r["precision"] for r in rows] == [s.precision for s in bm.SPECS]
    assert all(r["repeats"] == 2 for r in rows)
    saved = json.loads((tmp_path / "out" / "benchmark.json").read_text())
    assert saved["isolated_processes"] is True and saved["repeats"] == 2
    assert set(saved["rounds"]) == {s.key for s in bm.SPECS}  # giữ dữ liệu thô từng vòng
    assert all(len(v) == 2 for v in saved["rounds"].values())
    assert (tmp_path / "out" / "benchmark.csv").exists()


def test_isolated_failure_becomes_row_with_error(monkeypatch, tmp_path):
    import subprocess
    import types

    def failing_run(cmd, *args, **kwargs):
        if "--row-out" not in cmd:
            raise OSError("not available")
        return types.SimpleNamespace(returncode=3, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", failing_run)
    rows = bm.run_benchmark(
        tmp_path, "v2", only=["onnx_fp32", "pytorch_fp32"], map_part=None, out_dir=tmp_path / "out"
    )
    assert [r["status"] for r in rows] == ["lỗi", "lỗi"] and "boom" in rows[0]["note"]
