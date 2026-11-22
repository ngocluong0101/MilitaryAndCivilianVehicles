import numpy as np
from PIL import Image

from serve_helpers import StubDetector
from vehicledet.serve.demo import TABLE_HEADERS, build_demo, run_demo_inference


def test_demo_function_returns_image_table_and_timing():
    det = StubDetector()
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    out, rows, summary = run_demo_inference(det, img, conf=0.25)
    assert isinstance(out, Image.Image) and out.size == (64, 48)
    assert rows == [[1, "military tank", 0.91, 1.0, 2.0, 30.0, 40.0]] and len(rows[0]) == len(
        TABLE_HEADERS
    )
    assert "1 đối tượng" in summary and "1.5 ms" in summary and "stub" in summary
    assert det.calls == [((64, 48), 0.25, 0.7)]  # đi qua Detector.predict


def test_demo_handles_no_image_and_no_detections():
    det = StubDetector()
    assert run_demo_inference(det, None, 0.25) == (None, [], "Hãy tải một ảnh lên.")
    _, rows, summary = run_demo_inference(det, np.zeros((10, 10, 3), dtype=np.uint8), conf=0.95)
    assert rows == [] and "0 đối tượng" in summary


def test_build_demo_constructs_interface():
    demo = build_demo(StubDetector())
    assert demo.__class__.__name__ == "Interface"
