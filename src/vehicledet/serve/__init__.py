"""Phục vụ model: lõi inference dùng chung (inference.py), API FastAPI (app.py), demo (demo.py).

Các module ở đây chỉ phụ thuộc numpy, Pillow, onnxruntime (+ fastapi/gradio cho app/demo), để image
Docker CPU nhỏ và không cần torch/ultralytics (trừ backend TensorRT).
"""
