# (Tuỳ chọn) Đưa demo ONNX CPU lên Hugging Face Spaces miễn phí

Space dạng Gradio chạy được trên phần cứng CPU miễn phí, đủ cho model ONNX cỡ nhỏ (YOLO11n/s).
Tôi chưa triển khai thật Space nào; đây là các bước dự kiến, hãy kiểm tra lại giao diện Hugging Face
khi làm (tên mục có thể đổi).

## Chuẩn bị file (một thư mục riêng, ví dụ `hf_space/`)

```
hf_space/
├── README.md            # phần đầu (YAML) khai báo Space, xem bên dưới
├── app.py
├── requirements.txt
├── best.onnx            # exports/<tag>/best.onnx (cỡ ~10–40 MB; dùng git LFS nếu Space yêu cầu)
└── vehicledet/          # sao chép src/vehicledet/__init__.py và toàn bộ src/vehicledet/serve/
    ├── __init__.py
    └── serve/...
```

`README.md` (YAML ở đầu file):

```yaml
---
title: Military & Civilian Vehicle Detection
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
license: agpl-3.0
---
```

`requirements.txt` (chỉ cần phụ thuộc phục vụ, không cần torch/ultralytics):

```
onnxruntime==1.32.0
numpy==2.5.5
Pillow==13.0.0
```
(`gradio` do Space tự cài theo `sdk_version`.)

`app.py`:

```python
from vehicledet.serve.demo import build_demo
from vehicledet.serve.inference import load_detector

demo = build_demo(load_detector("best.onnx", backend="onnx", device="cpu"))
demo.launch()
```

## Triển khai

1. Tạo Space mới trên https://huggingface.co/new-space, chọn SDK **Gradio**, phần cứng **CPU basic**.
2. Clone repo Space về máy, chép các file ở trên vào, `git add`, `git commit`, `git push`.
3. Chờ Space build xong rồi mở link; thêm link vào README chính.

## Lưu ý về license và nội dung

- Ultralytics là **AGPL-3.0**: weights/mã dùng Ultralytics khi công khai phải tuân thủ AGPL-3.0
  (nên đặt `license: agpl-3.0` và công khai mã nguồn của Space).
- Dataset **CC BY 4.0** (ghi công tác giả: Gupta, Pareek, Singal, Rao). Ảnh gốc có thể có bản quyền riêng,
  nên **không** đưa ảnh dataset vào Space (kể cả ảnh ví dụ); dùng ảnh bạn có quyền sử dụng.
- Ghi rõ trong Space rằng demo chỉ phục vụ nghiên cứu/học tập và nêu giới hạn của model
  (xem MODEL_CARD.md khi đã có).
