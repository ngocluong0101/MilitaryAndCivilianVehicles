"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Demo Gradio: tải ảnh lên → ảnh có bbox + bảng kết quả + thời gian suy luận.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vehicledet.serve.inference import Detector, draw_detections, load_detector

TABLE_HEADERS = ["#", "Lớp", "Điểm", "x1", "y1", "x2", "y2"]


def run_demo_inference(
    detector: Detector, image: np.ndarray | Image.Image | None, conf: float, iou: float = 0.7
) -> tuple[Image.Image | None, list[list], str]:
    """Hàm thuần cho UI (kiểm thử được không cần mở Gradio)."""
    if image is None:
        return None, [], "Hãy tải một ảnh lên."
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    result = detector.predict(pil, conf=conf, iou=iou)
    rows = [
        [i, d.label, round(d.score, 3), *(round(v, 1) for v in d.box)]
        for i, d in enumerate(result.detections, 1)
    ]
    summary = (
        f"{len(result.detections)} đối tượng · model {result.inference_ms:.1f} ms · "
        f"tổng {result.total_ms:.1f} ms · backend: {result.backend}"
    )
    return draw_detections(pil, result.detections), rows, summary


def build_demo(detector: Detector):
    import gradio as gr

    def fn(image, conf):
        return run_demo_inference(detector, image, conf)

    return gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(type="numpy", label="Ảnh đầu vào"),
            gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Ngưỡng tin cậy"),
        ],
        outputs=[
            gr.Image(type="pil", label="Kết quả"),
            gr.Dataframe(headers=TABLE_HEADERS, label="Các đối tượng phát hiện"),
            gr.Textbox(label="Thời gian suy luận"),
        ],
        title="Military & Civilian Vehicle Detection",
        description=(
            "Phát hiện 6 lớp: military truck, civilian car, military tank, civilian aircraft, "
            "military aircraft, military helicopter. Chỉ phục vụ nghiên cứu/học tập; "
            "xem MODEL_CARD.md về giới hạn."
        ),
        flagging_mode="never",
    )


def launch(
    model_path: str, backend: str = "auto", device: str = "auto", host="127.0.0.1", port: int = 7860
) -> None:
    build_demo(load_detector(model_path, backend, device)).launch(
        server_name=host, server_port=port
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Demo Gradio (không cần cài các phụ thuộc huấn luyện)")
    ap.add_argument("--model", required=True, help="File .onnx hoặc .engine")
    ap.add_argument("--backend", choices=["auto", "onnx", "tensorrt"], default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    launch(a.model, a.backend, a.device, a.host, a.port)


if __name__ == "__main__":
    main()
