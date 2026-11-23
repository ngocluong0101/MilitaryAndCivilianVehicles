"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Tiện ích dựng model ONNX giả và detector stub cho test phục vụ model.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

from vehicledet.serve.inference import Detection, DetectionResult

NAMES = {
    0: "military truck",
    1: "civilian car",
    2: "military tank",
    3: "civilian aircraft",
    4: "military aircraft",
    5: "military helicopter",
}


def fake_output() -> np.ndarray:
    """Đầu ra YOLO giả (1, 4+6, 4), box trong không gian letterbox 640x640.

    a0 tank 0.9; a1 tank 0.8 (bị NMS triệt tiêu bởi a0); a2 truck 0.1 (bị lọc nếu conf > 0.1);
    a3 car 0.7 (khác lớp nên được giữ).
    """
    out = np.zeros((1, 10, 4), dtype=np.float32)
    boxes = [(320, 320, 100, 100), (322, 322, 100, 100), (100, 100, 40, 40), (320, 320, 100, 100)]
    cls_scores = [(2, 0.9), (2, 0.8), (0, 0.1), (1, 0.7)]
    for i, (box, (c, s)) in enumerate(zip(boxes, cls_scores, strict=True)):
        out[0, :4, i] = box
        out[0, 4 + c, i] = s
    return out


def write_fake_onnx(path, end2end: bool = False, imgsz: int = 640, with_names: bool = True):
    out = fake_output()
    const = helper.make_node(
        "Constant",
        [],
        ["output0"],
        value=helper.make_tensor("v", TensorProto.FLOAT, out.shape, out.flatten().tolist()),
    )
    graph = helper.make_graph(
        [const],
        "fake",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    meta = {"imgsz": f"[{imgsz}, {imgsz}]", "end2end": str(end2end)}
    if with_names:
        meta["names"] = str(NAMES)
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, str(path))
    return path


class StubDetector:
    """Detector giả: luôn trả về một phát hiện cố định, không cần model/weights."""

    backend = "stub"
    model_name = "stub.onnx"
    class_names = list(NAMES.values())

    def __init__(self):
        self.calls = []

    def predict(self, image, conf=0.25, iou=0.7):
        self.calls.append((image.size, conf, iou))
        det = Detection(2, "military tank", 0.91, (1.0, 2.0, 30.0, 40.0))
        return DetectionResult([det] if conf <= 0.91 else [], image.size, 1.5, 2.5, self.backend)
