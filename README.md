# Military & Civilian Vehicle Detection

Phát hiện và phân loại 6 lớp phương tiện (`military truck`, `civilian car`, `military tank`,
`civilian aircraft`, `military aircraft`, `military helicopter`) bằng
[Ultralytics](https://github.com/ultralytics/ultralytics) YOLO.

> **Trạng thái:** đang nâng cấp theo từng phase, xem [ROADMAP.md](ROADMAP.md). Chưa có kết quả đánh
> giá được công bố; README đầy đủ (kết quả, benchmark, demo) sẽ viết ở Phase 7.

## Cài đặt

Yêu cầu: Python ≥ 3.10, GPU NVIDIA (đã kiểm tra trên RTX 4060 Laptop 8 GB, driver 595).

```bash
bash scripts/setup_env.sh        # tạo .venv, cài requirements-dev.txt, cài package ở chế độ editable
```

Script này tương đương:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements-dev.txt   # torch 2.15.0, ultralytics 8.4.216, pytest, ruff
.venv/bin/python -m pip install -e . --no-deps
```

Phiên bản được cố định trong `requirements.txt`; bản đóng băng đầy đủ ở `requirements-lock.txt`.

## Dataset

Nguồn: [Mendeley Data](https://data.mendeley.com/datasets/njdjkbxdpn/1) (không nằm trong repo).
Tải về và giải nén sao cho có `Images/` và `Labels/{CSV Format,TXT format,...}`, rồi trỏ tới đó:

```bash
export DATA_ROOT=/duong/dan/toi/dataset   # mặc định: <repo>/data (đã được .gitignore)
```

## Sử dụng

```bash
# 1a. Kiểm tra dữ liệu, tìm ảnh trùng, chia train/val/test không rò rỉ (xem DATASET.md)
python -m vehicledet audit            # -> reports/data_audit.md
python -m vehicledet find-duplicates  # -> reports/duplicates.md
python -m vehicledet make-split       # -> splits/v2/{train,val,test}.txt, reports/split_stats.md

# 1b. Cách chia cũ (train_labels.csv trừ test_labels.csv) -> splits/legacy/, dùng để so sánh
python -m vehicledet make-legacy-split

# 2. (tuỳ chọn) dựng thư mục YOLO bằng symlink + data.yaml; lệnh train tự làm bước này
python -m vehicledet prepare-data --split v2

# 3. Train một thí nghiệm (tự đánh giá best.pt trên val, ghi MLflow + reports/experiments/<tên>/)
python -m vehicledet train --config configs/exp3b_yolo11s_v2_640.yaml
python -m vehicledet train --config configs/train_smoke.yaml     # chạy thử pipeline (1 epoch, 2% dữ liệu)

# 3b. Chạy tuần tự các thí nghiệm 1–3 (qua đêm; bỏ qua cái đã xong) rồi tổng hợp bảng
bash scripts/run_experiments.sh
python scripts/summarize_runs.py --roadmap ROADMAP.md

# 4. Đánh giá weights trên val (tập test cần --final và chỉ chạy một lần, xem bên dưới)
python -m vehicledet evaluate --weights runs/detect/<tên>/weights/best.pt --split v2 --part val
```

Mỗi run lưu ở `runs/` (không đưa lên git) kèm `vehicledet_run.json` (config, phiên bản thư viện, GPU,
git commit). Kết quả gọn (json/csv/confusion matrix png) nằm ở `reports/experiments/<tên>/` và được
đưa lên git. MLflow chạy local (sqlite `mlflow.db` + `mlruns/`, đều bị `.gitignore`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Quy tắc tập test:** mọi lựa chọn model/siêu tham số dựa trên **val**. `evaluate --part test` bị từ
chối nếu thiếu `--final`, và chỉ ghi vào `reports/test_results.json` một lần (không ghi đè trừ khi
`--allow-rerun`).

## Export, benchmark, API và demo

Sau khi chọn model bằng val (xem ROADMAP):

```bash
# Export ONNX + TensorRT FP16/INT8 (INT8 hiệu chuẩn bằng 500 ảnh TRAIN) -> exports/<tag>/
python -m vehicledet export --weights runs/detect/<tên>/weights/best.pt --tag <tên>

# Benchmark tốc độ/VRAM/mAP -> reports/benchmark.csv (định dạng không chạy được ghi "chưa đo")
python -m vehicledet benchmark --model-dir exports/<tên> --map-part val
# TensorRT và onnxruntime-gpu cần cài thêm: docs/TENSORRT_SETUP.md

# API FastAPI (ONNX Runtime, chạy được cả CPU): http://127.0.0.1:8000/docs
python -m vehicledet serve --model exports/<tên>/best.onnx
curl -F "file=@anh.jpg" "http://127.0.0.1:8000/predict?conf=0.25"

# Demo Gradio dùng chung lõi inference với API
python -m vehicledet demo --model exports/<tên>/best.onnx
```

API: `POST /predict` (ảnh, tham số `conf`, `iou`; giới hạn `MAX_UPLOAD_MB`, mặc định 10 MB; lỗi: 400 ảnh
không hợp lệ, 413 quá lớn, 422 tham số sai, 503 model chưa nạp) và `GET /health`.

Docker (CPU, ONNX Runtime; image ~760 MB, không chứa torch/ultralytics):

```bash
mkdir -p models && cp exports/<tên>/best.onnx models/best.onnx
docker compose up --build api                  # API  -> http://localhost:8000/docs
docker compose --profile demo up --build demo  # demo -> http://localhost:7860
```

Bản GPU (`docker/Dockerfile.gpu`, profile `gpu`) mới được viết, **chưa kiểm thử**. Hướng dẫn đưa demo
CPU lên Hugging Face Spaces: [docs/HF_SPACES.md](docs/HF_SPACES.md).

## Chất lượng dữ liệu

Tóm tắt (chi tiết: [DATASET.md](DATASET.md)): **71,7% ảnh nằm trong nhóm ảnh trùng/gần trùng** và
**64,9% ảnh val của cách chia cũ có bản gần trùng trong train** (rò rỉ). Cách chia mới `splits/v2`
gom cả nhóm vào cùng một tập nên không còn nhóm nào nằm ở hai tập.

## Kiểm thử

```bash
python -m pytest                 # không cần GPU/dataset
ruff check . && ruff format --check .
```

## License

- Mã dùng **Ultralytics (AGPL-3.0)**: nếu phân phối code hoặc weights huấn luyện bằng Ultralytics,
  cần tuân thủ AGPL-3.0 (hoặc có giấy phép doanh nghiệp của Ultralytics).
- Dataset thuộc điều khoản của Mendeley Data (xem trang dataset); không phân phối lại dữ liệu gốc.
- Chưa chọn license cho mã nguồn của riêng repo này.
