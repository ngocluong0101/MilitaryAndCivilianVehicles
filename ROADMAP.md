# ROADMAP – Nâng cấp Military & Civilian Vehicle Detection

Trạng thái: Phase 0–6 đã xong có số liệu thật; Phase 7 còn GIF demo và xác nhận CI xanh sau khi push · Cập nhật: 2026-11-22

## Tiến độ
| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khảo sát & ROADMAP | ✅ |
| 1 | Dọn repo & cấu trúc lại | ✅ |
| 2 | Kiểm tra dữ liệu & chia lại tập | ✅ |
| 3 | Train & thí nghiệm | ✅ |
| 4 | Phân tích lỗi | ✅ |
| 5 | Export & benchmark RTX 4060 | ✅ |
| 6 | API, Docker & demo Gradio | ✅ |
| 7 | Hoàn thiện & bullet CV | 🟨 tài liệu/CI xong; còn GIF demo, push + CI xanh |

## Hiện trạng (khảo sát Phase 0)

### Cấu trúc repo (không tính nội dung `yolov5/`)
Git đang theo dõi 5 file ngoài `yolov5/` (+143 file trong `yolov5/`, 2,3 MB):

```
.
├── .gitignore
├── README.md                     # mô tả cách train bằng yolov5/train.py
├── custom_data.yaml              # path: "Military and Civilian Vehicles Classification/train_data"
├── requirements.txt              # chỉ có 1 dòng: pandas
├── Military and Civilian Vehicles Classification/
│   └── split_dataset.py          # chia train/val từ CSV
├── yolov5/                       # bản copy YOLOv5 (AGPL-3.0), yêu cầu ultralytics>=8.4.83
├── interview.md, myCV/           # đã .gitignore (không theo dõi)
```

Lịch sử: 6 commit (upload → sửa path cứng trong yaml → sửa rò rỉ train/val → ignore file cá nhân).

### Cách train hiện tại
1. `cd "Military and Civilian Vehicles Classification" && python split_dataset.py`
   - Đọc `Labels/CSV Format/{train,test}_labels.csv`.
   - `val = test_labels.csv` (496 ảnh); `train = train_labels.csv − test` (6.206 ảnh). Ảnh/nhãn TXT được **copy** sang `train_data/images|labels/{train,val}`.
2. `python yolov5/train.py --data custom_data.yaml --img 640 --batch 16 --epochs 50`
3. Không có evaluate, không có test split, không có log thí nghiệm, không có seed cố định.

### Môi trường máy này (đã kiểm tra)
- GPU: RTX 4060 Laptop, 8188 MiB, driver 595.99.02; `nvcc` 12.0 (hệ thống).
- Python 3.12.3 hệ thống **không có pip**. Env conda `~/anaconda3/envs/ai` đã có: torch 2.15.0 (CUDA khả dụng), torchvision 0.29.2, **ultralytics 8.4.106**, onnx 1.23.2, onnxruntime 1.32.0 (**bản CPU**), pytest 9.0.3, fastapi 0.128.5, opencv 4.13, numpy 2.5.5, pillow 13.
- **Chưa có**: mlflow, imagehash, ruff, gradio, pandas, scikit-learn, onnxruntime-gpu, tensorrt.
- **Dataset KHÔNG có trên máy** (thư mục `Military and Civilian Vehicles Classification/` chỉ chứa `split_dataset.py`; không tìm thấy ở Downloads/Documents/Desktop). Chưa có weights `.pt` nào của project.

## Rủi ro đã phát hiện

### A. Đường dẫn & cấu trúc
1. **Dấu cách trong tên thư mục** (`Military and Civilian Vehicles Classification/`, `Labels/TXT format/`, `Labels/CSV Format/`) – dễ lỗi khi dùng shell, Docker, script, YAML. → Phase 1: dùng `DATA_ROOT`, tham số hoá, `pathlib`.
2. `split_dataset.py` dùng đường dẫn **tương đối theo cwd** (`./Labels/...`), chỉ chạy được khi `cd` vào đúng thư mục.
3. `custom_data.yaml` có `path` tương đối: kết quả phụ thuộc cwd khi chạy `train.py` (đã từng sửa path cứng ở commit `8edab92`). Nên sinh yaml với đường dẫn tuyệt đối từ `DATA_ROOT`/`splits/*.txt`.
4. Split bằng **copy** toàn bộ ảnh (trùng dung lượng ~6.700 ảnh). Phase 2 nên chia bằng file danh sách `splits/*.txt`, không copy.
5. `.replace(".jpg", ".txt")` giả định mọi ảnh đều `.jpg` (viết thường). Ảnh `.png`/`.jpeg`/`.JPG` sẽ sinh tên nhãn sai → `FileNotFoundError` hoặc, tệ hơn, copy nhầm. Cần dùng `Path.with_suffix`.
6. `.gitignore` dùng pattern không neo (`Images/`, `Labels/`, `train_data/`) → sẽ vô tình ignore mọi thư mục cùng tên ở bất kỳ đâu (ví dụ `reports/.../Images/`). Chưa ignore `mlruns/`, `*.onnx`, `*.engine`, `*.cache`. → sửa ở Phase 1.

### B. Phụ thuộc vào `yolov5/`
1. Repo mang theo 143 file mã YOLOv5 + `LICENSE` AGPL-3.0; muốn dùng gói `ultralytics` (pip) thì thư mục này thừa. **Sẽ hỏi bạn trước khi xoá** (đúng quy định ở Phase 1).
2. Không có mã nguồn nào của project nằm ngoài `yolov5/` ngoài `split_dataset.py` → lõi project hiện là “gọi `train.py` của bên thứ ba”.
3. `requirements.txt` chỉ có `pandas`, còn README yêu cầu thêm `yolov5/requirements.txt` (`ultralytics>=8.4.83` – **không cố định phiên bản**) → không tái lập được.
4. Cần kiểm tra khi chuyển sang gói `ultralytics`: README nhắc “Do not change model training logic” – chuyển sang ultralytics tức là đổi pipeline (loss/augment mặc định của bản `yolov5su` ≠ YOLOv5 gốc anchor-based). Phải ghi rõ điều này khi so sánh.

### C. Rò rỉ dữ liệu / độ tin cậy của đánh giá
1. **Đã biết & đã xử lý một phần**: `test_labels.csv` bị lặp hoàn toàn trong `train_labels.csv`; script hiện loại các tên trùng khỏi train.
2. **Chưa có test độc lập**: tập “val” thực chất là `test_labels.csv` gốc. Nếu chọn model/ngưỡng trên nó thì không còn tập nào để báo cáo cuối → Phase 2 chia lại train/val/test.
3. **Mới loại trùng theo *tên file***, chưa loại theo *nội dung*: ảnh trùng/gần trùng (cùng ảnh khác tên, crop/resize, ảnh liên tiếp của một cảnh, ảnh web đăng lại) vẫn có thể nằm ở cả train và val → mAP có thể bị thổi phồng. Cần pHash (Phase 2).
4. Đếm tập dùng `unique()` theo `filename`; chưa kiểm tra CSV có khớp TXT hay không (nhãn có thể lệch giữa 3 định dạng CSV/TXT/XML).
5. Chưa kiểm tra: class id ngoài 0–5, toạ độ ngoài [0,1], bbox rỗng, ảnh không có nhãn, file nhãn thừa/thiếu, ảnh hỏng.
6. Cache nhãn của Ultralytics (`labels/*.cache`) có thể cũ nếu đổi split → phải xoá cache khi chia lại (ghi vào script).
7. Không có seed nào được ghi → mọi so sánh hiện tại không tái lập.
8. Cảnh báo công bằng khi so sánh Phase 3: baseline “cách chia cũ” có thể cao ảo; đó chính là điểm cần đo, không phải lỗi.

### D. Quyền riêng tư / git (ngoài phạm vi kỹ thuật nhưng nên biết)
- Commit `f75b473` ("update") đã thêm `myCV/TranNgocLuong_CV.pdf`, `myCV/interview_prep.md`, `interview.md`; commit `7decb15` chỉ gỡ khỏi *working tree*. Cả hai commit **đã có trên `origin/main`** (GitHub public hay không tôi không kiểm tra được) nên CV và ghi chú phỏng vấn vẫn còn trong lịch sử git. Nếu repo công khai, cân nhắc viết lại lịch sử (`git filter-repo`) hoặc coi nội dung đó là đã lộ. Tôi **không tự làm** việc này (thao tác phá huỷ, cần bạn quyết định).

### E. Môi trường / kỹ thuật cho các phase sau
- `onnxruntime` đang là bản CPU, chưa có TensorRT → Phase 5 cần cài `onnxruntime-gpu` và TensorRT (phiên bản phải khớp CUDA mà torch 2.15 đi kèm; `nvcc` hệ thống 12.0 khá cũ so với driver 595). Bước nào không chạy được sẽ ghi “chưa đo”.
- Chưa có `ruff`, `imagehash`, `mlflow`, `gradio` → cài ở các phase tương ứng; tôi sẽ **hỏi trước khi cài/sửa env `ai`** hoặc tạo venv riêng (đề xuất).
- RTX 4060 8 GB: YOLO11s imgsz 640 batch 16 thường vừa; imgsz 960 có thể phải giảm batch (8) – sẽ ghi rõ khi thực nghiệm.
- Ultralytics là AGPL-3.0: ghi chú license ở Phase 1; weights/code phát hành kèm phải tuân thủ.
- Ảnh trong dataset: chưa xác nhận kích thước/độ phân giải/giấy phép từng ảnh → xác nhận từ trang Mendeley khi viết `DATASET.md`.

## Việc cần bạn xử lý trước/trong Phase 1–2
1. **Tải dataset** từ Mendeley về một thư mục *ngoài repo* (ví dụ `~/data/military_civilian_vehicles`) và cho tôi biết đường dẫn → đây là `DATA_ROOT`. Phase 1 chưa cần dataset, **Phase 2 thì bắt buộc**.
2. Trả lời các câu hỏi “hỏi trước” của Phase 1: xoá `yolov5/` hay không; tạo venv mới hay dùng env conda `ai`.
3. Quyết định về lịch sử git chứa CV (mục D).

## Checklist chi tiết

### Phase 0 – Khảo sát & lập ROADMAP.md
- [x] Đọc toàn bộ repo (trừ nội dung `yolov5/`), ghi lại cấu trúc, cách train hiện tại
- [x] Tạo `ROADMAP.md`
- [x] Liệt kê rủi ro: đường dẫn có dấu cách, phụ thuộc `yolov5/`, chỗ dễ rò rỉ dữ liệu
- [x] Dừng và báo cáo

### Phase 1 – Dọn repo & cấu trúc lại
- [x] Cấu trúc mới (`configs/`, `src/vehicledet/`, `scripts/`, `tests/`, `reports/`, `splits/`; `docker/` để Phase 6)
- [x] Thay mã YOLOv5 copy bằng `ultralytics==8.4.216` (pip, cố định); đã xoá `yolov5/` theo yêu cầu
- [x] `DATA_ROOT` qua `--data-root` / biến môi trường (mặc định `<repo>/data`)
- [x] CLI thống nhất (`python -m vehicledet {make-legacy-split,prepare-data,train}`)
- [x] Ghi chú license (README + CLAUDE.md)
- [x] Cập nhật `.gitignore` (neo `/data/`, `/runs/`, `/mlruns/`, `/weights/`, `*.pt/*.onnx/*.engine/*.cache`)
- [x] Venv mới `.venv` + `scripts/setup_env.sh` + `requirements*.txt` (+ lock)
- [x] Xoá `myCV/` và `interview.md` khỏi lịch sử git (cục bộ; **chưa force-push**)
- [x] `CLAUDE.md`
- [x] Smoke test train thật trên RTX 4060 (1 epoch, 2% dữ liệu) chạy end-to-end
- [x] Dừng và báo cáo

### Phase 2 – Kiểm tra chất lượng dữ liệu & chia lại tập
- [x] `data audit` → `reports/data_audit.md` + json
  - [x] Số ảnh, bbox/lớp, bbox/ảnh, ảnh không nhãn
  - [x] Phân bố kích thước bbox (COCO small/medium/large) theo lớp
  - [x] Nhãn hỏng (class id, toạ độ, bbox 0, thiếu/thừa file, ảnh hỏng) – kết quả: 0 lỗi cứng
  - [x] So sánh CSV ↔ TXT
- [x] Ảnh trùng/gần trùng (pHash + Hamming), trong và giữa các tập; danh sách + ảnh minh hoạ ở `reports/duplicates*`
- [x] Chia lại train/val/test (80/10/10), stratify, nhóm gần trùng chung một tập, seed 0, `splits/v2/*.txt`
- [x] Giữ cách chia cũ (`splits/legacy`) để so sánh ở Phase 3
- [x] `DATASET.md`
- [x] Test (nhãn, audit, dedup, chia tập không rò rỉ nhóm trùng) bằng dữ liệu giả – 50 test pass, ruff sạch
- [x] Dừng và báo cáo

### Phase 3 – Train & thí nghiệm có so sánh
- [x] MLflow local (sqlite + `mlruns/`): log config, phiên bản thư viện, seed, metric từng epoch, AP từng lớp, artifact nhỏ (đã kiểm thử end-to-end trên CPU)
- [x] `evaluate.py` (mAP50, mAP50-95, P, R, AP từng lớp, confusion matrix, latency khi val) – chạy trên split/part bất kỳ; tập test có khoá `--final`, ghi một lần
- [x] Config thí nghiệm `configs/exp1…exp4` (mỗi file đổi một thứ) + `scripts/run_experiments.sh` (chạy tuần tự, bỏ qua cái đã xong)
- [x] `scripts/summarize_runs.py` → bảng vào ROADMAP (giữa cặp marker `EXPERIMENTS`)
- [x] Tải sẵn weights (`weights/`: yolov5su, yolo11n, yolo11s)
- [x] **Đã chạy** thí nghiệm 1, 2, 3a, 3b trên val (15 epoch, tổng ~69 phút gồm nghỉ; bảng ở mục Số liệu). **Thí nghiệm 4 (imgsz 960) chưa chạy** – tuỳ chọn, laptop
- [ ] (5) Xử lý mất cân bằng lớp – *chưa cần theo audit (xem Quyết định 2026-11-22); chỉ làm nếu AP từng lớp cho thấy*
- [x] Chốt số epoch/early stopping: **15 epoch, patience 8, close_mosaic 5** (laptop nên giảm tải nhiệt; 2026-11-22). Thời gian train thực tế vẫn cần ghi khi chạy
- [x] Thí nghiệm phụ exp3c (YOLO11n 30 epoch, chỉ đổi số epoch) vì mọi lượt 15 epoch chưa hội tụ; chọn model cuối bằng val = **exp3c** (0,480)
- [x] Chạy model được chọn **một lần** trên test → `reports/test_results.json`: mAP50 **0,844**, mAP50-95 **0,531** (679 ảnh)
- [x] Dừng và báo cáo

### Phase 4 – Phân tích lỗi
- [x] Gom lỗi theo nhóm trên **val** của model cuối: nhầm lớp, bỏ sót (nhỏ / bị che / ảnh nhiều vật thể), phát hiện thừa, trùng lặp, định vị sai, nhãn đáng ngờ (`python -m vehicledet analyze-errors`, có test)
- [x] Gallery + `reports/error_analysis/error_analysis.md` (+ json, 9 ảnh ghép nhỏ)
- [x] Nhãn đáng ngờ: 12 ca được ghi nhận, **không sửa nhãn nào**
- [x] Đã thử 1 cách khắc phục (exp6: `cls` 0,5 → 1,0) và đo lại trên val: **không cải thiện** (mAP50-95 0,469 vs 0,480; nhầm lớp 79 vs 76; xem `reports/error_analysis/remedy_comparison.md`)
- [x] Đề xuất cách khác chưa thử: imgsz 960; bổ sung/rà soát nhãn (cần bạn đồng ý)
- [x] Dừng và báo cáo

### Phase 5 – Export & benchmark trên RTX 4060
- [x] Export ONNX / TensorRT FP16 / TensorRT INT8 cho model cuối exp3c (`python -m vehicledet export`); INT8 hiệu chuẩn bằng 500 ảnh **train** (đã kiểm: 500/500 ảnh thuộc train)
- [x] `benchmark.py`: PyTorch FP32/FP16, ONNX Runtime GPU, TensorRT FP16/INT8 – latency p50/p95, throughput b1/4/8/16, VRAM của riêng tiến trình, mAP50-95 trên **test**; đo lặp 3 vòng xen kẽ, mỗi lần một tiến trình riêng, ghi xung SM (NVML)
- [x] Warm-up 20 lần, 200 lần lặp; ghi phiên bản driver/CUDA/TensorRT/ONNX Runtime và input vào `reports/benchmark.json`
- [x] Cài TensorRT 10.16.1.11 (cu13) và onnxruntime-gpu 1.32.0; hướng dẫn `docs/TENSORRT_SETUP.md`
- [x] **Đã chạy** export + benchmark thật; bảng cuối ở mục Số liệu (nguồn `reports/benchmark.csv`, `reports/benchmark_runs/`)
- [x] Dừng và báo cáo (bảng tốc độ vs độ chính xác)

### Phase 6 – Phục vụ model & demo
- [x] FastAPI: `POST /predict` (ngưỡng `conf`/`iou`, giới hạn kích thước, lỗi 400/413/422/500/503 rõ ràng), `GET /health`; backend ONNX Runtime (mặc định, CPU) hoặc TensorRT
- [x] Dockerfile CPU (đã build + chạy thử, 758 MB) + Dockerfile GPU (**chưa kiểm thử**) + `docker-compose.yml`
- [x] Demo Gradio gọi lại cùng `Detector.predict` với API (không viết hai lần)
- [x] (Tuỳ chọn) Hướng dẫn Hugging Face Spaces (`docs/HF_SPACES.md`, chưa triển khai thật)
- [x] Test API bằng `TestClient` với detector stub + model ONNX giả dựng bằng `onnx.helper` (không cần weights thật)
- [x] Chạy thử với model đã train của project (exp3c): API trên ONNX Runtime GPU và Docker CPU trả kết quả giống nhau; `scripts/check_serve_parity.py` → `reports/serve_parity.json` (bbox điểm ≥ 0,6: 100% khớp IoU > 0,5, 96% khớp IoU > 0,9 trên 99 bbox)
- [x] Dừng và báo cáo

### Phase 7 – Hoàn thiện & tổng kết
- [x] GitHub Actions (`.github/workflows/ci.yml`): ruff + pytest trên CPU, không cần dataset/GPU/torch (`requirements-ci.txt`). Đã kiểm chứng tương đương trong venv sạch (124 test pass, ruff sạch); **chưa chạy trên GitHub** nên chưa biết badge xanh hay chưa
- [x] `MODEL_CARD.md` (khung): mục đích, dữ liệu, giới hạn, sử dụng có trách nhiệm, license; các mục kết quả/AP theo lớp/lớp hay nhầm ghi "chưa đo"/"chưa phân tích"
- [x] README viết lại: bài toán, sơ đồ Mermaid, bảng dữ liệu (đã đo), cách tái lập từng bước; bảng test/benchmark/confusion matrix/GIF ghi "chưa đo/chưa có"
- [x] Hướng dẫn quay GIF demo (`docs/DEMO_RECORDING.md`)
- [x] Bullet CV (tiếng Anh) + câu hỏi phỏng vấn kèm đáp án gợi ý + thẻ học + bài luyện: `myCV/vehicle_detection_v2_study_guide.md` (file riêng tư, `myCV/` bị `.gitignore`); bullet 1–3 dùng số thật có nguồn, bullet 4 còn chỗ trống chờ số liệu
- [ ] Sau khi train/benchmark: điền bảng test, benchmark, confusion matrix, GIF vào README/MODEL_CARD; điền bullet 4 và các chỗ `[Điền ...]` trong study guide
- [ ] Push và xác nhận CI xanh
- [ ] Dừng và báo cáo

## Số liệu
> Chưa đo – chưa có số liệu thật nào. Các bảng dưới đây sẽ được điền từ file trong `reports/`.

### Dữ liệu
Nguồn: `reports/data_audit.json`, `reports/duplicates.json`, `reports/split_stats.json`.

| Split | Số ảnh | Số bbox | Ảnh gần trùng với tập khác | Ảnh gần trùng trong chính tập | Nhãn lỗi |
|---|---|---|---|---|---|
| Toàn bộ dataset | 6.772 | 10.718 | 4.853 ảnh (71,7%) thuộc 2.345 nhóm | – | 0 lỗi cứng; 36 dòng trùng hệt (4 ảnh), 3 bbox < 4 px |
| legacy train | 6.206 | 9.997 | 345 (có bản trong val) | 4.188 | 0 |
| legacy val | 496 | 718 | **322 (64,9%)** (có bản trong train) | 68 | 0 |
| v2 train | 5.412 | 8.535 | 0 | 3.870 | 0 |
| v2 val | 681 | 1.068 | 0 | 492 | 0 |
| v2 test | 679 | 1.115 | 0 | 491 | 0 |

Ảnh nền (không bbox): legacy 0; v2 train/val/test = 53/7/7. Kiểm tra chéo độc lập (dHash 8×8, Hamming ≤ 4): legacy 348 cặp khác tập, v2 14 cặp (đã xem tay: đều là ảnh khác nhau, cùng bố cục).

### Thí nghiệm (chọn trên val; dòng cuối = kết quả test của model được chọn)
Sinh tự động từ `reports/experiments/*/` bằng `python scripts/summarize_runs.py --roadmap ROADMAP.md`. Lưu ý: dòng `val` của `legacy` và `v2` là **hai tập val khác nhau** (legacy val bị rò rỉ), không so trực tiếp như cùng một phép đo.

<!-- EXPERIMENTS:START (sinh bởi scripts/summarize_runs.py, đừng sửa tay) -->
| # | Model | Split | imgsz | mAP50 | mAP50-95 | Lớp yếu nhất (AP50-95) | Thời gian train | Epoch | MLflow run |
|---|---|---|---|---|---|---|---|---|---|
| exp1_yolov5su_legacy_640 | yolov5su | legacy (val) | 640 | 0.792 | 0.540 | civilian aircraft (0.459) | 15 phút | 15 | `c2ad559e` |
| exp2_yolov5su_v2_640 | yolov5su | v2 (val) | 640 | 0.755 | 0.459 | civilian aircraft (0.380) | 14 phút | 15 | `adebc1d7` |
| exp3a_yolo11n_v2_640 | yolo11n | v2 (val) | 640 | 0.771 | 0.468 | civilian aircraft (0.402) | 9.0 phút | 15 | `ea2b6a7e` |
| exp3b_yolo11s_v2_640 | yolo11s | v2 (val) | 640 | 0.770 | 0.465 | civilian aircraft (0.383) | 15 phút | 15 | `96484233` |
| exp3c_yolo11n_v2_640_30ep | yolo11n | v2 (val) | 640 | 0.788 | 0.480 | civilian aircraft (0.422) | 15 phút | 30 | `dd0d8145` |
| exp6_yolo11n_v2_640_30ep_cls1 | yolo11n | v2 (val) | 640 | 0.781 | 0.469 | civilian aircraft (0.401) | 15 phút | 30 | `369051d5` |
| TEST: exp3c_yolo11n_v2_640_30ep | yolo11n | v2 (test) | 640 | 0.844 | 0.531 | military aircraft (0.430) | 15 phút | 30 | `dd0d8145` |
<!-- EXPERIMENTS:END -->

### Benchmark (RTX 4060 Laptop 8 GB, input 1×3×640×640 letterbox)
Sinh từ `reports/benchmark.csv` bằng `python -m vehicledet benchmark --roadmap ROADMAP.md` (trung vị 3 vòng đo, [min–max]; mỗi lần đo một tiến trình riêng). Latency = thời gian model (`speed['inference']` của Ultralytics) cho 1 ảnh, batch 1; throughput = ảnh/giây của cả lời gọi predict (gồm tiền/hậu xử lý). Định dạng không chạy được trên máy ghi "chưa đo" kèm lý do.

<!-- BENCHMARK:START (sinh bởi `python -m vehicledet benchmark --roadmap`) -->
| Định dạng | Precision | Latency p50 (ms): trung vị [min–max] | Throughput b16 (img/s): trung vị [min–max] | VRAM (MB) | mAP50-95 | SM MHz (trung vị) | Ghi chú |
|---|---|---|---|---|---|---|---|
| PyTorch | FP32 | 4.822 [4.774–6.561] | 308.75 [178.46–309.13] | 672.0 | 0.5323 (test) | 2325.0 | n=3 vòng |
| PyTorch | FP16 | 5.002 [4.927–5.409] | 471.46 [254.93–477.64] | 532.0 | 0.5319 (test) | 2250.0 | n=3 vòng |
| ONNX Runtime | FP32 | 3.684 [3.683–8.422] | 158.77 [112.76–159.0] | 1308.0 | 0.5323 (test) | 2460.0 | n=3 vòng |
| TensorRT | FP16 | 1.11 [1.104–1.207] | 650.83 [633.12–657.58] | 1196.0 | 0.5321 (test) | 2550.0 | n=3 vòng |
| TensorRT | INT8 | 1.141 [1.141–1.142] | 766.52 [766.34–766.59] | 1056.0 | 0.5228 (test) | 2550.0 | n=3 vòng |
<!-- BENCHMARK:END -->

## Quyết định & lý do
- 2026-11-20: Giữ nguyên `yolov5/` và mọi file hiện có trong Phase 0 (chỉ thêm `ROADMAP.md`).
- 2026-11-21: Người dùng chốt: dataset ở `<repo>/data` (ngoài git); tạo venv mới `.venv` (cách tạo: `scripts/setup_env.sh`, README); xoá `myCV/` khỏi git; xoá `yolov5/`.
- 2026-11-21: Dựng dataset cho Ultralytics bằng **symlink** (`<DATA_ROOT>/yolo/<split>/`) thay vì copy ảnh, vì Ultralytics tìm nhãn bằng cách thay `/images/`→`/labels/` còn dataset gốc dùng `Images/` + `Labels/TXT format/`.
- 2026-11-21: Cách chia cũ được tái tạo thành `splits/legacy/{train,val}.txt` (6.206 / 496, khớp số trong `custom_data.yaml`) để so sánh ở Phase 3. Cách chia mới (Phase 2) sẽ vào `splits/v2/`.
- 2026-11-21: Đã xoá `custom_data.yaml` và `split_dataset.py` (thay bằng `configs/data.yaml` + `vehicledet.data.splits`); bản cũ còn trong lịch sử git.

- 2026-11-22: Phát hiện ảnh trùng bằng pHash 16×16 ≤ 56 **hoặc** pHash 8×8 ≤ 4; ngưỡng chọn bằng cách xem ảnh theo từng mức khoảng cách; số nhóm phẳng trong 8–56, bị nối dây chuyền từ 64. Ưu tiên chống rò rỉ nên chấp nhận gộp thêm vài ảnh chỉ "cùng mẫu". Kiểm tra chéo bằng dHash độc lập.
- 2026-11-22: Nhãn chính = TXT (YOLO); CSV chỉ để đối chiếu. Không sửa nhãn nào. 4 file TXT lặp khối dòng để nguyên (Ultralytics tự loại dòng trùng).
- 2026-11-22: 67 ảnh không bbox được giữ làm ảnh nền trong `v2` (gần như đều là ảnh nền thật sau khi xem tay). Hệ quả: so sánh `legacy` vs `v2` ở Phase 3 khác nhau ở cả cách chia lẫn thành phần (v2 có thêm ảnh nền + 3 ảnh bị đổi tên) – phải ghi rõ khi báo cáo.
- 2026-11-22: Phase 3 mục 5 (xử lý mất cân bằng lớp) **chưa cần**: lớp nhiều/ít nhất chỉ ≈ 1,8 lần theo bbox (2.428 vs 1.373) và mỗi tập có đủ cả 6 lớp. Chỉ làm nếu kết quả AP theo lớp cho thấy lớp yếu liên quan tới số lượng.

- 2026-11-22: Đo thử 1 epoch yolo11s, imgsz 640, batch 16, train v2 (5.412 ảnh): ~65 giây/epoch (gồm val), VRAM đỉnh 3,97 GB. Ngoại suy (chưa đo riêng yolov5su/yolo11n/960): với **15 epoch**, mỗi thí nghiệm 640 ≈ 15–20 phút, 4 thí nghiệm 640 ≈ 1–1,3 giờ (+ ~15 phút nghỉ giữa các lượt); exp4 (960) ≈ 40 phút và là tuỳ chọn. Đây là ước tính, không phải kết quả.
- 2026-11-22: Dùng Tracker MLflow riêng, gỡ callback MLflow tích hợp của Ultralytics (nó log cả weights và đặt tên experiment theo đường dẫn) – chỉ ở mức run, không đổi cấu hình toàn cục. `reports/experiments/*/run_info.json` dùng đường dẫn `<repo>/…` để không lộ đường dẫn máy.
- 2026-11-22: Khoá tập test ngay trong code: `evaluate --part test` cần `--final`, chỉ ghi `reports/test_results.json` một lần.

- 2026-11-22: Làm trước code Phase 5 và 6 (theo yêu cầu, vì chưa train được). Export/benchmark đi qua cùng đường `predict` của Ultralytics cho mọi định dạng (tiền/hậu xử lý giống nhau); API/demo dùng pipeline ONNX tự viết bằng numpy/Pillow để image CPU không cần torch. Ultralytics 8.4.216 đã thay `half`/`int8` bằng `quantize` (16/8).
- 2026-11-22: Kiểm tra pipeline ONNX tự viết với yolo11n COCO (không phải model của project, chỉ để kiểm tra cơ chế): với bbox điểm ≥ 0,6 khớp Ultralytics IoU > 0,9 ở ≥ 95% (20–40 ảnh val; kiểm tra thủ công, không lưu trong `reports/`); phát hiện điểm thấp lệch nhiều hơn do tiền xử lý khác nhau (Pillow vs OpenCV resize). Chưa phải kết quả của model project; `scripts/check_serve_parity.py` sẽ đo lại với model thật.

- 2026-11-22: CI dùng `requirements-ci.txt` riêng (không torch/ultralytics) vì mọi test chạy được không cần chúng (import nặng đều để trễ); giữ job nhanh và rẻ.
- 2026-11-22: Bullet CV và câu hỏi phỏng vấn đặt trong `myCV/` (riêng tư) thay vì trong repo công khai; không đưa bullet CV vào README/ROADMAP.

- 2026-11-22: Người dùng chốt **20 epoch** (thay cho 60); sau đó giảm tiếp còn 15 epoch (mục kế tiếp). Cập nhật `configs/exp*.yaml` (epochs 20, patience 10) và MODEL_CARD. Lưu ý: với 20 epoch, warm-up mặc định 3 epoch chiếm 15% và `close_mosaic` mặc định 10 nghĩa là nửa sau không dùng mosaic; giữ mặc định cho mọi run nên so sánh vẫn công bằng, nhưng mAP có thể thấp hơn so với train dài hơn.

- 2026-11-22: Người dùng chỉ có laptop, lo máy nóng khi train nhiều → đề xuất và áp dụng: **15 epoch** (patience 8, `close_mosaic` 5, mọi lượt như nhau), chỉ chạy 4 lượt 640 (exp4 960 là tuỳ chọn), `scripts/run_experiments.sh` nghỉ 300 giây giữa các lượt và in nhiệt độ GPU. Lý do giảm epoch: project phục vụ phỏng vấn, giá trị nằm ở quy trình đối chứng chứ không ở mAP tuyệt đối; khởi tạo từ COCO nên 15 epoch đủ để so sánh. Chưa đo được nhiệt độ máy người dùng.

- 2026-11-22: Kết quả 4 thí nghiệm (val, 15 epoch, seed 0; nguồn `reports/experiments/*/val.json`): trên val sạch v2, yolov5su 0,459 · YOLO11n **0,468** · YOLO11s 0,465 mAP50-95. Theo quy tắc (chọn theo mAP50-95 trên val) model được chọn là **YOLO11n**; nhưng ba model chỉ cách nhau 0,009 (một seed, val 681 ảnh/1.068 bbox) nên **không phân biệt được về mặt thống kê**; YOLO11n được ưu tiên thêm vì nhỏ nhất và nhanh nhất (train ~9 phút; thời gian inference trong lúc val 2,1 ms/ảnh so với 8,4 ms của YOLO11s). Chưa chạy test.
- 2026-11-22: Mọi lượt đều có epoch tốt nhất là epoch 15/15 và mAP50-95 còn tăng ở các epoch cuối (vd YOLO11n: 0,409 → 0,465 trong 5 epoch cuối) ⇒ **chưa hội tụ**, mAP tuyệt đối chắc chắn thấp hơn mức tối đa; so sánh giữa các model vẫn công bằng (cùng ngân sách) nhưng thứ hạng có thể đổi nếu train dài hơn.
- 2026-11-22: Tác động rò rỉ (ước lượng chiều hướng, **không phải phép đo sạch** vì hai val và hai train khác nhau): exp1 (val cũ, rò rỉ) 0,540 vs exp2 (val v2, sạch) 0,459 mAP50-95, chênh ≈ 0,08; mAP50 0,792 vs 0,755. Lớp giảm nhiều nhất: civilian car (0,687 → 0,450).

- 2026-11-22: Chạy thí nghiệm phụ exp3c (YOLO11n 30 epoch; chỉ đổi số epoch so với exp3a): val mAP50-95 **0,480** (exp3a 0,468; epoch tốt nhất 30/30, 4 epoch cuối 0,464 → 0,478 nên đã khá phẳng). Chọn exp3c làm model cuối theo val rồi **chạy test một lần**: mAP50 0,844, mAP50-95 0,531 (nguồn `reports/test_results.json`). Test cao hơn val ≈ 0,05; coi là dao động giữa hai tập nhỏ (một seed), chưa đo khoảng tin cậy; lớp yếu nhất đổi từ civilian aircraft (val) sang military aircraft (test).
- 2026-11-22: Phân tích lỗi (val, conf 0,25, IoU 0,5; `reports/error_analysis/`): 830 đúng · 76 nhầm lớp · 87 định vị sai · 27 trùng · 121 thừa · 122 bỏ sót. Nhầm lớp lớn nhất civilian aircraft → military aircraft (28/76). Phần lớn "thừa" là vật thể thật chưa gán nhãn ⇒ nhãn không đầy đủ làm precision bị đánh giá thấp; không tự sửa nhãn.
- 2026-11-22: Cách khắc phục thử ở exp6: tăng trọng số loss phân loại `cls` 0,5 → 1,0 (chỉ đổi một tham số so với exp3c). Ghi kết quả kể cả khi không cải thiện. Vì test đã chạy một lần cho exp3c, exp6 nếu tốt hơn **chỉ được báo cáo trên val** (không đo lại trên test, tránh vi phạm quy tắc test một lần).
- 2026-11-22: **Kết quả exp6: không cải thiện** (val mAP50-95 0,469 so với 0,480 của exp3c; precision 0,805 vs 0,760, recall 0,707 vs 0,755; nhầm lớp 79 vs 76; civilian→military aircraft 29 vs 28). Model cuối giữ nguyên exp3c. Chi tiết: `reports/error_analysis/remedy_comparison.md`.
- 2026-11-22: Cài `tensorrt-cu13` 10.16.1.11 (chọn nhánh 10.x, tránh 11.x chưa chắc tương thích Ultralytics 8.4.216/INT8) và `onnxruntime-gpu` 1.32.0 (thay `onnxruntime` CPU trong `.venv`) cho Phase 5.

- 2026-11-22: **Benchmark thật** (model exp3c, test mAP): TensorRT FP16 p50 batch 1 = 1,11 ms (PyTorch FP32 4,82 ms) với mAP50-95 0,5321 vs 0,5323; INT8 1,14 ms, throughput b16 767 vs 651 img/s (+18%), mAP50-95 **0,5228 (−0,0095)**. Kết luận: FP16 là điểm cân bằng tốt nhất cho model này; INT8 không đáng kể với latency batch 1.
- 2026-11-22: Ba lỗi đo được sửa trong quá trình benchmark (đều do kiểm tra, không phải do chọn số): (1) ONNX Runtime báo có `CUDAExecutionProvider` nhưng session thật rơi về CPU vì thiếu thư viện CUDA 13 trong đường dẫn → kiểm tra bằng session thật và dùng `ort.preload_dlls()`; (2) ảnh mẫu nhiều kích thước làm model đầu vào động tối ưu lại liên tục → letterbox về đúng 640×640 trước khi đo; (3) VRAM bị lẫn giữa các định dạng trong một tiến trình → mỗi lần đo một tiến trình riêng.
- 2026-11-22: GPU laptop bị hãm xung (~780 MHz, cờ `0x24`) khi tải nhẹ và lên 2250–2550 MHz khi tải nặng; làm latency TensorRT dao động ~3 lần giữa các lần chạy (FP16 3,2 vs 1,1 ms). Xử lý: đo lặp 3 vòng xen kẽ, ghi xung SM, báo trung vị + [min–max], lưu các lần đo trước ở `reports/benchmark_runs/` (không giấu số).

- 2026-11-22: Phase 6 chạy với model thật exp3c: `GET /health` trả đủ 6 tên lớp lấy từ metadata ONNX; `POST /predict` trên GPU (ONNX Runtime CUDA) và CPU (Docker, image 758 MB, `healthy`) cho cùng kết quả (điểm lệch ≤ 0,001). Độ khớp pipeline tự viết với Ultralytics (`reports/serve_parity.json`, 100 ảnh val): bbox điểm ≥ 0,6 khớp IoU > 0,9 ở 96%, điểm 0,25–0,4 chỉ 67% (lệch do Pillow vs OpenCV resize). **Lần gọi đầu trên GPU chậm (~805 ms)** do khởi tạo/autotune → khi triển khai GPU cần warm-up.

## Vấn đề còn mở
- Model cuối (exp3c, 30 epoch) có epoch tốt nhất là epoch cuối nhưng đường cong đã khá phẳng; chưa đo khoảng tin cậy, một seed duy nhất.
- Nhãn dataset không đầy đủ (nhiều vật thể phụ cùng ảnh không có nhãn) → precision/AP đo được thấp hơn chất lượng thật. Sửa nhãn cần bạn đồng ý (chưa làm).
- Lớp yếu nhất ở cả 4 lượt là **civilian aircraft** (AP50-95 0,38–0,46); ma trận nhầm lẫn gợi ý civilian aircraft hay bị dự đoán thành military aircraft (cần phân tích ở ngưỡng conf thực tế trong bước phân tích lỗi, chưa kết luận).
- **Toàn bộ kết quả mô hình vẫn "chưa đo"** (train, test, phân tích lỗi, benchmark GPU). README/MODEL_CARD/bullet CV số 4 đang chờ các số này.
- Workflow CI chưa chạy trên GitHub; phiên bản action (`checkout@v4`, `setup-python@v5`) chưa kiểm chứng bằng một lần chạy thật.
- `docker/Dockerfile.gpu` chưa build/chạy thử (cần NVIDIA Container Toolkit; Docker trên máy có runtime `nvidia`). Cần chọn base image/onnxruntime-gpu khớp CUDA.
- Số latency phụ thuộc trạng thái xung nhịp GPU laptop (xem Quyết định 2026-11-22); chưa thử khoá xung (`nvidia-smi -lgc` cần quyền root, laptop có thể không hỗ trợ). Bảng chính thức đại diện cho trung vị 3 vòng, không phải "số tuyệt đối".
- Pipeline ONNX của API tiền xử lý hơi khác Ultralytics (Pillow vs OpenCV): đã đo độ khớp bbox (`reports/serve_parity.json`) nhưng **chưa đo ảnh hưởng tới mAP** của pipeline API.
- `Dockerfile.gpu` vẫn chưa kiểm thử (Docker có runtime nvidia nhưng chưa build/chạy ảnh GPU).
- **Cần bạn chạy để hoàn tất việc xoá `myCV/` và `interview.md` khỏi GitHub:** `git push --force origin main` (lịch sử đã viết lại cục bộ; hash các commit từ `update` trở đi đã đổi, hiện tại `main` = `3e82533`). Bản sao lưu trước khi viết lại: `../MilitaryAndCivilianVehicles-pre-filter.bundle` (chứa cả CV và interview.md; xoá sau khi xác nhận ổn). GitHub vẫn có thể giữ commit cũ trong cache/fork/PR; nếu từng lộ thì coi như đã lộ.
- Cập nhật 2026-11-21: đã xoá `interview.md` khỏi lịch sử (0 tham chiếu `interview`/`myCV` trong mọi commit); cây thư mục của HEAD không đổi.
- Trang Mendeley ghi 6.642 ảnh dương / 130 ảnh âm / 11.528 đối tượng, trong khi đếm từ file thật là 6.705 / 67 / 10.718 bbox hợp lệ. Chưa rõ lý do (xem `DATASET.md`); project dùng số đếm từ file.
- Train v2 vẫn chứa nhiều bản sao gần trùng trong chính nó (3.870/5.412 ảnh): không gây rò rỉ nhưng làm số ảnh "hiệu dụng" nhỏ hơn (~4.264 bức ảnh khác nhau toàn dataset). Có thể cân nhắc khi giải thích kết quả ở Phase 3.
- pHash không bắt ảnh lật ngang/crop mạnh; ảnh gần trùng loại này (nếu có) vẫn có thể nằm ở hai tập khác nhau.
- Bản quyền ảnh gốc của dataset chưa xác minh (trang Mendeley chỉ nêu license CC BY 4.0 cho dataset).
- Cảnh báo vô hại khi train: `Python.h: No such file or directory` (triton không compile được vì thiếu `python3-dev`); Ultralytics vẫn chạy bình thường. Cài `python3.12-dev` nếu muốn mất cảnh báo.
- Ultralytics ghi/cập nhật cấu hình toàn cục ở `~/.config/Ultralytics/settings.json` (dùng chung với project khác trên máy); không ảnh hưởng đường dẫn vì config dùng đường dẫn tuyệt đối.
- Chưa chọn license cho mã nguồn repo (README ghi rõ).
- Trang Mendeley không mô tả cách chia train/test của CSV.
