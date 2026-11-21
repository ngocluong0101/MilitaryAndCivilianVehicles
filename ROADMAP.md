# ROADMAP – Nâng cấp Military & Civilian Vehicle Detection

Trạng thái: Phase 1 / 7 (chờ duyệt) · Cập nhật: 2026-11-21

## Tiến độ
| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khảo sát & ROADMAP | ✅ |
| 1 | Dọn repo & cấu trúc lại | ✅ (chờ bạn duyệt) |
| 2 | Kiểm tra dữ liệu & chia lại tập | ⬜ |
| 3 | Train & thí nghiệm | ⬜ |
| 4 | Phân tích lỗi | ⬜ |
| 5 | Export & benchmark RTX 4060 | ⬜ |
| 6 | API, Docker & demo Gradio | ⬜ |
| 7 | Hoàn thiện & bullet CV | ⬜ |

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
- [ ] `data audit` → `reports/data_audit.md` + json
  - [ ] Số ảnh, bbox/lớp, bbox/ảnh, ảnh không nhãn
  - [ ] Phân bố kích thước bbox (COCO small/medium/large) theo lớp
  - [ ] Nhãn hỏng (class id, toạ độ, bbox 0, thiếu/thừa file, ảnh hỏng)
  - [ ] So sánh CSV ↔ TXT
- [ ] Ảnh trùng/gần trùng (pHash + Hamming), trong và giữa các tập; xuất danh sách + ảnh minh hoạ
- [ ] Chia lại train/val/test (80/10/10), stratify, nhóm gần trùng chung một tập, seed cố định, `splits/*.txt`
- [ ] Giữ cách chia cũ để so sánh ở Phase 3
- [ ] `DATASET.md`
- [ ] Test (nhãn, chia tập không rò rỉ nhóm trùng) bằng dữ liệu giả
- [ ] Dừng và báo cáo

### Phase 3 – Train & thí nghiệm có so sánh
- [ ] MLflow local
- [ ] `evaluate.py` (mAP50, mAP50-95, P, R, AP từng lớp, confusion matrix, latency)
- [ ] Thí nghiệm trên val: (1) yolov5su cách chia cũ, (2) yolov5su chia mới, (3) YOLO11n/s, (4) imgsz 640 vs 960, (5) xử lý mất cân bằng (nếu cần)
- [ ] Epoch hợp lý + early stopping; ghi thời gian train
- [ ] Chạy model được chọn **một lần** trên test → `reports/test_results.json`
- [ ] `scripts/summarize_runs.py` → bảng vào ROADMAP
- [ ] Dừng và báo cáo

### Phase 4 – Phân tích lỗi
- [ ] Gom lỗi theo nhóm (nhầm lớp, bỏ sót nhỏ/che/chồng, phát hiện thừa, nhãn sai)
- [ ] Gallery + `reports/error_analysis/error_analysis.md`
- [ ] Đề xuất & thử 1 cách khắc phục, đo lại trên val
- [ ] Dừng và báo cáo

### Phase 5 – Export & benchmark trên RTX 4060
- [ ] Export ONNX, TensorRT FP16, TensorRT INT8 (calib ~500 ảnh train)
- [ ] `benchmark.py` (latency p50/p95, throughput b1/4/8/16, VRAM, mAP50-95 test)
- [ ] Warm-up, lặp nhiều lần, ghi phiên bản driver/CUDA/TensorRT → `reports/benchmark.csv`
- [ ] Hướng dẫn cài + “chưa đo” cho bước không chạy được
- [ ] Dừng và báo cáo

### Phase 6 – Phục vụ model & demo
- [ ] FastAPI (`/predict`, `/health`, giới hạn file, xử lý lỗi; backend ONNX Runtime/TensorRT)
- [ ] Dockerfile (CPU + GPU tuỳ chọn) + `docker-compose.yml`
- [ ] Demo Gradio dùng chung code inference
- [ ] (Tuỳ chọn) Hướng dẫn Hugging Face Spaces
- [ ] Test API bằng `TestClient` + model stub
- [ ] Dừng và báo cáo

### Phase 7 – Hoàn thiện & tổng kết
- [ ] GitHub Actions: ruff + pytest
- [ ] `MODEL_CARD.md`
- [ ] README mới (Mermaid, bảng kết quả, benchmark, confusion matrix, GIF demo)
- [ ] 3–4 bullet CV tiếng Anh (số liệu thật, ghi nguồn file)
- [ ] Câu hỏi phỏng vấn + ý trả lời ngắn
- [ ] Dừng và báo cáo

## Số liệu
> Chưa đo – chưa có số liệu thật nào. Các bảng dưới đây sẽ được điền từ file trong `reports/`.

### Dữ liệu
| Split | Số ảnh | Số bbox | Ảnh trùng/gần trùng phát hiện | Nhãn lỗi |
|---|---|---|---|---|
| train (cách chia cũ) | 6.206 (theo `custom_data.yaml`, chưa kiểm chứng) | chưa đo | chưa đo | chưa đo |
| val (cách chia cũ) | 496 (theo `custom_data.yaml`, chưa kiểm chứng) | chưa đo | chưa đo | chưa đo |

### Thí nghiệm (chọn trên val; dòng cuối = kết quả test của model được chọn)
| # | Model | Split | imgsz | mAP50 | mAP50-95 | Lớp yếu nhất (AP) | Thời gian train | MLflow run |
|---|---|---|---|---|---|---|---|---|
| – | chưa chạy | | | | | | | |

### Benchmark (RTX 4060 Laptop 8 GB, input chưa xác định)
| Định dạng | Precision | Latency p50 (ms) | p95 (ms) | Throughput b16 (img/s) | VRAM (MB) | mAP50-95 test |
|---|---|---|---|---|---|---|
| chưa đo | | | | | | |

## Quyết định & lý do
- 2026-11-20: Giữ nguyên `yolov5/` và mọi file hiện có trong Phase 0 (chỉ thêm `ROADMAP.md`).
- 2026-11-21: Người dùng chốt: dataset ở `<repo>/data` (ngoài git); tạo venv mới `.venv` (cách tạo: `scripts/setup_env.sh`, README); xoá `myCV/` khỏi git; xoá `yolov5/`.
- 2026-11-21: Dựng dataset cho Ultralytics bằng **symlink** (`<DATA_ROOT>/yolo/<split>/`) thay vì copy ảnh, vì Ultralytics tìm nhãn bằng cách thay `/images/`→`/labels/` còn dataset gốc dùng `Images/` + `Labels/TXT format/`.
- 2026-11-21: Cách chia cũ được tái tạo thành `splits/legacy/{train,val}.txt` (6.206 / 496, khớp số trong `custom_data.yaml`) để so sánh ở Phase 3. Cách chia mới (Phase 2) sẽ vào `splits/v2/`.
- 2026-11-21: Đã xoá `custom_data.yaml` và `split_dataset.py` (thay bằng `configs/data.yaml` + `vehicledet.data.splits`); bản cũ còn trong lịch sử git.

## Vấn đề còn mở
- **Cần bạn chạy để hoàn tất việc xoá `myCV/` và `interview.md` khỏi GitHub:** `git push --force origin main` (lịch sử đã viết lại cục bộ; hash các commit từ `update` trở đi đã đổi, hiện tại `main` = `3e82533`). Bản sao lưu trước khi viết lại: `../MilitaryAndCivilianVehicles-pre-filter.bundle` (chứa cả CV và interview.md; xoá sau khi xác nhận ổn). GitHub vẫn có thể giữ commit cũ trong cache/fork/PR; nếu từng lộ thì coi như đã lộ.
- Cập nhật 2026-11-21: đã xoá `interview.md` khỏi lịch sử (0 tham chiếu `interview`/`myCV` trong mọi commit); cây thư mục của HEAD không đổi.
- Dữ liệu thực tế (phát hiện lúc dựng split cũ, chưa phải kết luận audit): 3 tên file trong `train_labels.csv` không có ảnh trên đĩa (`1convoy.jpg`, `h1.jpg`, `ripsaw.jpg`); 70 ảnh không xuất hiện trong CSV nào (67 ảnh có file nhãn TXT rỗng, 3 ảnh có nhãn TXT nhưng không có trong CSV); `test_labels.csv` (496) nằm trọn trong `train_labels.csv`. Tổng 6.772 ảnh, 6.772 file TXT/XML. Phase 2 sẽ audit đầy đủ.
- Cảnh báo vô hại khi train: `Python.h: No such file or directory` (triton không compile được vì thiếu `python3-dev`); Ultralytics vẫn chạy bình thường. Cài `python3.12-dev` nếu muốn mất cảnh báo.
- Ultralytics ghi/cập nhật cấu hình toàn cục ở `~/.config/Ultralytics/settings.json` (dùng chung với project khác trên máy); không ảnh hưởng đường dẫn vì config dùng đường dẫn tuyệt đối.
- Chưa chọn license cho mã nguồn repo (README ghi rõ).
- Cách chia gốc của Mendeley (`train/test` CSV) chưa rõ có ngụ ý quy ước nào về tập test; sẽ xác nhận từ trang dataset ở Phase 2.
