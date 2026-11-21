# Cài TensorRT và onnxruntime-gpu để benchmark trên GPU

Mặc định `.venv` chỉ có `onnxruntime` (CPU) và **chưa có TensorRT**. Vì vậy các dòng TensorRT và
ONNX-GPU trong `reports/benchmark.csv` sẽ là **"chưa đo"** (kèm lý do) cho tới khi bạn cài các gói
dưới đây. Script không bao giờ ước đoán số cho định dạng không chạy được.

Máy đã kiểm tra: RTX 4060 Laptop 8 GB, driver 595.99.02, `torch 2.15.0+cu132` (CUDA runtime 13.2).

## 1. Kiểm tra phiên bản CUDA

```bash
nvidia-smi                                                    # driver, CUDA tối đa driver hỗ trợ
.venv/bin/python -c "import torch; print(torch.version.cuda)"  # CUDA mà torch đang dùng (hiện 13.2)
```

## 2. TensorRT (cho export engine FP16/INT8 và benchmark)

Cần gói **khớp họ CUDA** của torch (CUDA 13.x ↔ gói `-cu13`, CUDA 12.x ↔ gói `-cu12`). Tên gói và
phiên bản chính xác phụ thuộc bản TensorRT hiện hành, hãy tra tài liệu NVIDIA TensorRT / PyPI trước
khi cài (tôi không kiểm chứng được trên máy này). Ví dụ:

```bash
.venv/bin/python -m pip install tensorrt-cu13        # hoặc tensorrt-cu12 nếu torch dùng CUDA 12
.venv/bin/python -c "import tensorrt; print(tensorrt.__version__)"   # phải in ra phiên bản
```

Lưu ý: Ultralytics cũng tự thử cài `tensorrt` khi export engine, nhưng nên cài tay để biết chắc
phiên bản. Export engine bắt buộc chạy trên GPU (`--device 0`) và có thể mất vài phút.

## 3. onnxruntime-gpu (cho dòng ONNX Runtime trên GPU)

`onnxruntime` và `onnxruntime-gpu` không dùng chung trong một venv:

```bash
.venv/bin/python -m pip uninstall -y onnxruntime
.venv/bin/python -m pip install onnxruntime-gpu      # chọn bản khớp CUDA/cuDNN theo bảng tương thích của ONNX Runtime
.venv/bin/python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# kỳ vọng có 'CUDAExecutionProvider'
```

Nếu danh sách provider không có `CUDAExecutionProvider`, benchmark sẽ ghi "chưa đo" cho dòng ONNX
trên GPU thay vì đo nhầm trên CPU. (API/demo vẫn chạy được với `onnxruntime` CPU.)

## 4. Chạy export + benchmark sau khi đã chọn model (từ val)

```bash
export DATA_ROOT=$PWD/data
TAG=exp3b                                  # tên thí nghiệm của model được chọn
.venv/bin/python -m vehicledet export --weights runs/detect/$TAG/weights/best.pt --tag $TAG --split v2

# Chạy thử với mAP trên val (không đụng tập test):
.venv/bin/python -m vehicledet benchmark --model-dir exports/$TAG --split v2 --map-part val

# Báo cáo cuối: mAP trên tập test (chỉ khi đã chốt model; cần --final) + cập nhật bảng trong ROADMAP:
.venv/bin/python -m vehicledet benchmark --model-dir exports/$TAG --split v2 --map-part test --final --roadmap ROADMAP.md
```

Quy ước đo: warm-up 20 lần, 200 lần lặp cho latency batch 1 (p50/p95), throughput batch 1/4/8/16
(trung vị), VRAM của riêng tiến trình qua `nvidia-smi`, input 640×640. Hiệu chuẩn INT8 dùng 500 ảnh
lấy ngẫu nhiên (seed 0) từ tập **train**; danh sách ảnh nằm ở `exports/<tag>/calibration_images.txt`.
Khi đo trên laptop, hãy cắm điện, đóng các chương trình khác dùng GPU, và để máy nguội giữa các lần chạy
để số đo ổn định.
