import numpy as np
import pytest
from PIL import Image

from serve_helpers import fake_output, write_fake_onnx
from vehicledet.serve.inference import (
    Detection,
    OnnxDetector,
    decode_output,
    draw_detections,
    letterbox,
    load_detector,
    nms,
)

NAMES = [
    "military truck",
    "civilian car",
    "military tank",
    "civilian aircraft",
    "military aircraft",
    "military helicopter",
]


def test_letterbox_shape_scale_and_padding():
    img = np.full((480, 640, 3), 200, dtype=np.uint8)  # rộng 640, cao 480
    tensor, r, (px, py) = letterbox(img, 640)
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert r == 1.0 and (px, py) == (0, 80)
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)  # vùng đệm
    assert tensor[0, 0, 320, 320] == pytest.approx(200 / 255)  # vùng ảnh


def test_letterbox_downscales_large_image():
    tensor, r, pad = letterbox(np.zeros((1000, 2000, 3), dtype=np.uint8), 640)
    assert r == pytest.approx(0.32) and pad == (0, 160) and tensor.shape == (1, 3, 640, 640)


def test_nms_keeps_best_and_non_overlapping():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]
    assert nms(boxes, np.array([0.8, 0.9, 0.7]), 0.5) == [1, 2]
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.99) == [0, 1, 2]  # IoU thấp hơn ngưỡng: giữ hết


def test_decode_maps_back_to_original_image_and_nms_is_per_class():
    # ảnh gốc 640x480 -> scale 1, pad (0, 80): box tâm (320,320) -> (320, 240)
    dets = decode_output(
        fake_output(), NAMES, conf=0.25, iou=0.7, scale=1.0, pad=(0, 80), image_size=(640, 480)
    )
    assert [(d.label, round(d.score, 2)) for d in dets] == [
        ("military tank", 0.9),
        ("civilian car", 0.7),
    ]
    assert dets[0].box == pytest.approx((270, 190, 370, 290))


def test_decode_respects_conf_and_clips_to_image():
    assert decode_output(fake_output(), NAMES, 0.95, 0.7, 1.0, (0, 0), (640, 640)) == []
    low = decode_output(fake_output(), NAMES, 0.05, 0.7, 1.0, (0, 0), (640, 640))
    assert {d.label for d in low} == {"military tank", "civilian car", "military truck"}
    # ảnh nhỏ hơn: toạ độ bị cắt trong [0, w] x [0, h]
    clipped = decode_output(fake_output(), NAMES, 0.25, 0.7, 1.0, (0, 0), (300, 300))
    assert all(0 <= v <= 300 for d in clipped for v in d.box)


def test_onnx_detector_end_to_end_with_fake_model(tmp_path):
    path = write_fake_onnx(tmp_path / "fake.onnx")
    det = OnnxDetector(path, device="cpu")
    assert det.class_names[2] == "military tank" and det.imgsz == 640
    assert det.backend == "onnxruntime (CPU)"
    res = det.predict(Image.new("RGB", (640, 480)), conf=0.25)
    assert [d.label for d in res.detections] == ["military tank", "civilian car"]
    assert res.detections[0].box == pytest.approx((270, 190, 370, 290), abs=0.5)
    assert (
        res.image_size == (640, 480) and res.inference_ms >= 0 and res.total_ms >= res.inference_ms
    )
    assert res.to_dict()["detections"][0]["bbox"]["x1"] == pytest.approx(270, abs=0.5)


def test_onnx_detector_rejects_end2end_models(tmp_path):
    with pytest.raises(ValueError, match="end2end"):
        OnnxDetector(write_fake_onnx(tmp_path / "e2e.onnx", end2end=True), device="cpu")


def test_load_detector_dispatch_and_errors(tmp_path):
    assert isinstance(
        load_detector(write_fake_onnx(tmp_path / "m.onnx"), device="cpu"), OnnxDetector
    )
    with pytest.raises(FileNotFoundError):
        load_detector(tmp_path / "missing.onnx")
    (tmp_path / "m.bin").write_bytes(b"0")
    with pytest.raises(ValueError, match="backend"):
        load_detector(tmp_path / "m.bin")
    with pytest.raises(ValueError, match="không hợp lệ"):
        load_detector(tmp_path / "m.bin", backend="caffe")


def test_draw_detections_changes_pixels_but_not_input():
    img = Image.new("RGB", (100, 80), "white")
    out = draw_detections(img, [Detection(2, "military tank", 0.9, (10, 10, 60, 50))])
    assert out.size == img.size and out.tobytes() != img.tobytes()
    assert img.tobytes() == Image.new("RGB", (100, 80), "white").tobytes()
