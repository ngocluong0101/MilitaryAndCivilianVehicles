# Model card – Military & Civilian Vehicle Detection

> **Trạng thái:** model đã train và đánh giá trên tập test **một lần** (số liệu bên dưới lấy từ
> `reports/test_results.json`). Không có con số nào trong file này
> được ước đoán.

## Tổng quan

| | |
|---|---|
| Nhiệm vụ | Phát hiện vật thể (bbox + lớp), 6 lớp |
| Lớp | `military truck`, `civilian car`, `military tank`, `civilian aircraft`, `military aircraft`, `military helicopter` |
| Họ model | Ultralytics **YOLO11n** (2,6 triệu tham số), chọn bằng mAP50-95 trên val trong số yolov5su, YOLO11n, YOLO11s (ba model cách nhau ≤ 0,009 ở 15 epoch; xem `reports/experiments_summary.md`) |
| Khởi tạo | Trọng số COCO pretrained của Ultralytics |
| Input | Ảnh RGB, letterbox về 640×640 (có thể 960 nếu thí nghiệm imgsz cho kết quả tốt hơn) |
| Phiên bản | thí nghiệm `exp3c_yolo11n_v2_640_30ep` (seed 0); chưa phát hành bản chính thức |

## Mục đích sử dụng

- **Dự kiến:** học tập, nghiên cứu, minh hoạ quy trình phát triển model computer vision; ví dụ trên ảnh
  công khai.
- **Ngoài phạm vi / không khuyến nghị:** giám sát, nhận dạng mục tiêu, hỗ trợ quyết định quân sự hay
  an ninh trong thực tế; xử lý ảnh vệ tinh, UAV, hồng ngoại, ban đêm; bất kỳ ứng dụng nào mà nhầm giữa
  "quân sự" và "dân sự" có thể gây hại.

## Dữ liệu

Nguồn, thống kê và các vấn đề đã phát hiện: xem [DATASET.md](DATASET.md). Tóm tắt:

- 6.772 ảnh, 10.718 bbox hợp lệ; 67 ảnh nền không có vật thể.
- 71,7% ảnh nằm trong nhóm ảnh trùng/gần trùng (≈ 4.264 bức ảnh khác nhau); cách chia gốc bị rò rỉ
  (64,9% ảnh val có bản gần trùng trong train).
- Chia `v2` (seed 0): train 5.412 / val 681 / test 679 ảnh; nhóm gần trùng không bị tách giữa các tập.
- Nhãn không bị sửa; 4 file nhãn lặp dòng (Ultralytics tự loại), 74 ảnh có bbox CSV/TXT lệch nhẹ.

## Huấn luyện

Cấu hình model cuối trong `configs/exp3c_yolo11n_v2_640_30ep.yaml` (các config so sánh cũ trong git history): seed 0, `deterministic`, khởi tạo từ COCO. Bốn thí nghiệm so sánh dùng
15 epoch (patience 8, `close_mosaic` 5, batch 16); model cuối là lượt YOLO11n **30 epoch** (chỉ đổi số epoch,
vì các lượt 15 epoch đều chưa hội tụ). Phần cứng: RTX 4060 Laptop 8 GB; lượt cuối mất ≈ 15 phút. Mỗi run lưu
config, phiên bản thư viện và git commit (`vehicledet_run.json`). Model được chọn **chỉ dựa trên val**.

## Đánh giá

Quy ước: mAP tính bằng Ultralytics (IoU 0,5 và 0,5–0,95), `conf=0.001`, `iou=0.7`, imgsz 640.

### Kết quả trên tập test (model được chọn, chạy một lần)

679 ảnh, nguồn `reports/test_results.json`.

| Chỉ số | Test | Val (để so sánh) |
|---|---|---|
| mAP50 | **0,844** | 0,788 |
| mAP50-95 | **0,531** | 0,480 |
| Precision | 0,823 | 0,760 |
| Recall | 0,778 | 0,755 |

Test cao hơn val khoảng 0,05 mAP50-95. Hai tập chỉ có ~680 ảnh và 1.000–1.100 bbox, một seed, nên chênh
lệch kiểu này nằm trong mức dao động do cách chia/độ khó từng tập; **không** nên đọc test là "tốt hơn" val
hay coi 0,531 là con số chính xác. Chưa đo khoảng tin cậy.

### AP theo lớp

| Lớp | AP50 (test) | AP50-95 (test) | AP50-95 (val) |
|---|---|---|---|
| military truck | 0,847 | 0,553 | 0,494 |
| civilian car | 0,917 | 0,638 | 0,467 |
| military tank | 0,914 | 0,606 | 0,560 |
| civilian aircraft | 0,810 | 0,497 | 0,422 |
| military aircraft | 0,716 | 0,430 | 0,487 |
| military helicopter | 0,857 | 0,462 | 0,451 |

Lớp yếu nhất đổi chỗ giữa val (civilian aircraft) và test (military aircraft); mỗi lớp chỉ có ~130–250
bbox mỗi tập nên chênh lệch từng lớp cỡ 0,05–0,1 chưa đủ cơ sở để kết luận lớp nào "thật sự" yếu hơn.

## Giới hạn đã biết

- **Ảnh web, ít đa dạng:** nhiều ảnh có watermark kho ảnh stock, minh hoạ/icon/render nhìn từ trên xuống;
  ảnh trùng khiến số ảnh hiệu dụng nhỏ hơn nhiều so với 6.772.
- **Vật thể nhỏ hiếm:** chỉ 3–11% bbox thuộc cỡ small (COCO), nên không có cơ sở khẳng định model tốt với
  vật thể nhỏ/xa.
- **Chưa kiểm tra** ảnh vệ tinh, UAV, hồng ngoại, ban đêm, thời tiết xấu, che khuất nặng.
- **Định nghĩa lớp mang tính hình thức:** "quân sự/dân sự" được gán theo vẻ ngoài; xe dân sự sơn ngụy
  trang hay xe quân sự sơn dân sự có thể bị phân loại sai.
- **Nhãn không đầy đủ:** nhiều ảnh chỉ gán nhãn vật thể chính (không gán các vật thể phụ cùng ảnh), nên
  precision bị đánh giá thấp và model có thể bị "phạt" khi phát hiện đúng vật thể chưa có nhãn. Không có đánh
  giá độc lập về độ chính xác của nhãn.
- **Độ không chắc chắn của kết quả:** một seed, val/test chỉ ~680 ảnh; chưa đo khoảng tin cậy; model chưa
  hội tụ hoàn toàn (xem `reports/experiments/*/results.csv`).

## Sử dụng có trách nhiệm

Model chỉ là công cụ học tập. Mọi kết quả cần được con người kiểm tra; không dùng đầu ra làm căn cứ
duy nhất cho quyết định có hậu quả. Không dùng để xâm phạm quyền riêng tư hoặc nhắm mục tiêu vào người
hay phương tiện thật.

## License và ghi công

- Mã/weights dùng Ultralytics: **AGPL-3.0**.
- Dataset: CC BY 4.0, Priyanka Gupta, Bhavya Pareek, Gaurav Singal, D Vijay Rao (DRDO Institute for
  System Studies and Analyses), DOI 10.17632/njdjkbxdpn.1. Ảnh gốc có thể có bản quyền riêng.
