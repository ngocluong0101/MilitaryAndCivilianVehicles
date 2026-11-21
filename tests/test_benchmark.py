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

    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert "CUDAExecutionProvider" in bm.check_available(spec("onnx_fp32"), f, "0")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    assert bm.check_available(spec("onnx_fp32"), f, "0") is None
    assert bm.check_available(spec("onnx_fp32"), f, "cpu") is None


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


def test_render_table_marks_unmeasured_without_numbers():
    measured = {
        **bm.empty_row(spec("onnx_fp32"), "0", ""),
        "status": "đã đo",
        "latency_b1_p50_ms": 3.1,
        "latency_b1_p95_ms": 3.9,
        "throughput_b16": 250.5,
        "vram_mb": 900.0,
        "mAP50-95": 0.55,
        "map_part": "val",
    }
    skipped = bm.empty_row(spec("tensorrt_fp16"), "0", "chưa cài tensorrt")
    table = bm.render_benchmark_table([measured, skipped])
    assert "| ONNX Runtime | FP32 | 3.1 | 3.9 | 250.5 | 900.0 | 0.55 (val) | |" in table
    assert "| TensorRT | FP16 | chưa đo | | | | | chưa cài tensorrt |" in table
    assert "chưa đo" in bm.render_benchmark_table([])


def test_test_split_map_requires_final_flag(tmp_path):
    with pytest.raises(PermissionError, match="--final"):
        bm.run_benchmark(Path(tmp_path), "v2", map_part="test", final=False)
