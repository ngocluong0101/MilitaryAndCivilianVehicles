# ROADMAP – Nâng cấp Military & Civilian Vehicle Detection

Trạng thái: code/tài liệu Phase 3, 5, 6, 7 đã xong; **chờ chạy train** để có số liệu thật · Cập nhật: 2026-11-22

## Tiến độ
| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khảo sát & ROADMAP | ✅ |
| 1 | Dọn repo & cấu trúc lại | ✅ |
| 2 | Kiểm tra dữ liệu & chia lại tập | ✅ |
| 3 | Train & thí nghiệm | 🟨 code xong, chưa chạy train |
| 4 | Phân tích lỗi | ⬜ |
| 5 | Export & benchmark RTX 4060 | 🟨 code xong, chưa đo |
| 6 | API, Docker & demo Gradio | 🟨 code xong, chưa thử với model đã train |
| 7 | Hoàn thiện & bullet CV | 🟨 tài liệu/CI xong, chờ số liệu thật |

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
- [ ] **Chạy** thí nghiệm trên val: (1) yolov5su chia cũ, (2) yolov5su chia mới, (3) YOLO11n/s, (4) imgsz 640 vs 960 – *chưa chạy, cần GPU*
- [ ] (5) Xử lý mất cân bằng lớp – *chưa cần theo audit (xem Quyết định 2026-11-22); chỉ làm nếu AP từng lớp cho thấy*
- [ ] Chốt số epoch/early stopping (đang đặt tạm 60 / patience 15, chờ bạn xác nhận) và ghi thời gian train
- [ ] Chạy model được chọn **một lần** trên test → `reports/test_results.json`
- [ ] Dừng và báo cáo

### Phase 4 – Phân tích lỗi
- [ ] Gom lỗi theo nhóm (nhầm lớp, bỏ sót nhỏ/che/chồng, phát hiện thừa, nhãn sai)
- [ ] Gallery + `reports/error_analysis/error_analysis.md`
- [ ] Đề xuất & thử 1 cách khắc phục, đo lại trên val
- [ ] Dừng và báo cáo

### Phase 5 – Export & benchmark trên RTX 4060
- [x] Code export ONNX / TensorRT FP16 / TensorRT INT8 (`python -m vehicledet export`); INT8 hiệu chuẩn bằng 500 ảnh **train** (seed 0, có test chứng minh không lẫn val/test)
- [x] `benchmark.py`: PyTorch FP32/FP16, ONNX Runtime, TensorRT FP16/INT8 – latency p50/p95 batch 1, throughput batch 1/4/8/16, VRAM của riêng tiến trình (nvidia-smi), mAP50-95 (val; test cần `--final`)
- [x] Warm-up 20 lần, 200 lần lặp; ghi phiên bản driver/CUDA/TensorRT/ONNX Runtime và kích thước input vào `reports/benchmark.json`
- [x] Hướng dẫn cài TensorRT/onnxruntime-gpu (`docs/TENSORRT_SETUP.md`); định dạng không chạy được ghi "chưa đo" kèm lý do
- [ ] **Chạy** export + benchmark với model đã chọn – *chưa chạy: chưa có model train xong; TensorRT và onnxruntime-gpu chưa cài*
- [ ] Dừng và báo cáo (kèm bảng tốc độ vs độ chính xác)

### Phase 6 – Phục vụ model & demo
- [x] FastAPI: `POST /predict` (ngưỡng `conf`/`iou`, giới hạn kích thước, lỗi 400/413/422/500/503 rõ ràng), `GET /health`; backend ONNX Runtime (mặc định, CPU) hoặc TensorRT
- [x] Dockerfile CPU (đã build + chạy thử, 758 MB) + Dockerfile GPU (**chưa kiểm thử**) + `docker-compose.yml`
- [x] Demo Gradio gọi lại cùng `Detector.predict` với API (không viết hai lần)
- [x] (Tuỳ chọn) Hướng dẫn Hugging Face Spaces (`docs/HF_SPACES.md`, chưa triển khai thật)
- [x] Test API bằng `TestClient` với detector stub + model ONNX giả dựng bằng `onnx.helper` (không cần weights thật)
- [ ] Chạy thử với model đã train của project (hiện mới thử với yolo11n COCO để kiểm tra cơ chế) và đo `scripts/check_serve_parity.py`
- [ ] Dừng và báo cáo

### Phase 7 – Hoàn thiện & tổng kết
- [x] GitHub Actions (`.github/workflows/ci.yml`): ruff + pytest trên CPU, không cần dataset/GPU/torch (`requirements-ci.txt`). Đã kiểm chứng tương đương trong venv sạch (107 test pass, ruff sạch); **chưa chạy trên GitHub** nên chưa biết badge xanh hay chưa
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
| – | chưa chạy | | | | | | | | |
<!-- EXPERIMENTS:END -->

### Benchmark (RTX 4060 Laptop 8 GB, input 1×3×640×640 letterbox)
Sinh từ `reports/benchmark.csv` bằng `python -m vehicledet benchmark --roadmap ROADMAP.md`. Latency = thời gian model (`speed['inference']` của Ultralytics) cho 1 ảnh, batch 1; throughput = ảnh/giây của cả lời gọi predict (gồm tiền/hậu xử lý). Định dạng không chạy được trên máy ghi "chưa đo" kèm lý do.

<!-- BENCHMARK:START (sinh bởi `python -m vehicledet benchmark --roadmap`) -->
| Định dạng | Precision | Latency p50 (ms) | p95 (ms) | Throughput b16 (img/s) | VRAM (MB) | mAP50-95 | Ghi chú |
|---|---|---|---|---|---|---|---|
| chưa đo | | | | | | | |
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

- 2026-11-22: Đo thử 1 epoch yolo11s, imgsz 640, batch 16, train v2 (5.412 ảnh): ~65 giây/epoch (gồm val), VRAM đỉnh 3,97 GB. Ngoại suy 4 thí nghiệm 640 ≈ 6–7 giờ với 60 epoch (chưa đo riêng yolov5su/yolo11n/960). Đây là ước tính, không phải kết quả.
- 2026-11-22: Dùng Tracker MLflow riêng, gỡ callback MLflow tích hợp của Ultralytics (nó log cả weights và đặt tên experiment theo đường dẫn) – chỉ ở mức run, không đổi cấu hình toàn cục. `reports/experiments/*/run_info.json` dùng đường dẫn `<repo>/…` để không lộ đường dẫn máy.
- 2026-11-22: Khoá tập test ngay trong code: `evaluate --part test` cần `--final`, chỉ ghi `reports/test_results.json` một lần.

- 2026-11-22: Làm trước code Phase 5 và 6 (theo yêu cầu, vì chưa train được). Export/benchmark đi qua cùng đường `predict` của Ultralytics cho mọi định dạng (tiền/hậu xử lý giống nhau); API/demo dùng pipeline ONNX tự viết bằng numpy/Pillow để image CPU không cần torch. Ultralytics 8.4.216 đã thay `half`/`int8` bằng `quantize` (16/8).
- 2026-11-22: Kiểm tra pipeline ONNX tự viết với yolo11n COCO (không phải model của project, chỉ để kiểm tra cơ chế): với bbox điểm ≥ 0,6 khớp Ultralytics IoU > 0,9 ở ≥ 95% (20–40 ảnh val; kiểm tra thủ công, không lưu trong `reports/`); phát hiện điểm thấp lệch nhiều hơn do tiền xử lý khác nhau (Pillow vs OpenCV resize). Chưa phải kết quả của model project; `scripts/check_serve_parity.py` sẽ đo lại với model thật.

- 2026-11-22: CI dùng `requirements-ci.txt` riêng (không torch/ultralytics) vì mọi test chạy được không cần chúng (import nặng đều để trễ); giữ job nhanh và rẻ.
- 2026-11-22: Bullet CV và câu hỏi phỏng vấn đặt trong `myCV/` (riêng tư) thay vì trong repo công khai; không đưa bullet CV vào README/ROADMAP.

## Vấn đề còn mở
- **Toàn bộ kết quả mô hình vẫn "chưa đo"** (train, test, phân tích lỗi, benchmark GPU). README/MODEL_CARD/bullet CV số 4 đang chờ các số này.
- Workflow CI chưa chạy trên GitHub; phiên bản action (`checkout@v4`, `setup-python@v5`) chưa kiểm chứng bằng một lần chạy thật.
- `.venv` chưa có TensorRT và onnxruntime-gpu (cần cài tay, xem `docs/TENSORRT_SETUP.md`); tới lúc đó các dòng TensorRT/ONNX-GPU trong benchmark là "chưa đo".
- `docker/Dockerfile.gpu` chưa build/chạy thử (cần NVIDIA Container Toolkit; Docker trên máy có runtime `nvidia`). Cần chọn base image/onnxruntime-gpu khớp CUDA.
- Pipeline ONNX của API tiền xử lý hơi khác Ultralytics (Pillow vs OpenCV), điểm tin cậy có thể lệch ~0,04 ở phát hiện chắc chắn; cần đo ảnh hưởng tới mAP với model thật.
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
