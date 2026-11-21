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
# 1. Tạo danh sách chia tập kiểu cũ (train_labels.csv trừ test_labels.csv) -> splits/legacy/
python -m vehicledet make-legacy-split

# 2. (tuỳ chọn) dựng thư mục YOLO bằng symlink + data.yaml; lệnh train tự làm bước này
python -m vehicledet prepare-data --split legacy

# 3. Train theo config
python -m vehicledet train --config configs/train_yolo11s.yaml
python -m vehicledet train --config configs/train_smoke.yaml     # chạy thử pipeline (1 epoch, 2% dữ liệu)
```

Mỗi run lưu ở `runs/` (không đưa lên git) kèm `vehicledet_run.json` (config, phiên bản thư viện, GPU,
git commit).

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
