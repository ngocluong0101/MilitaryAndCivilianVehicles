# Model card – Military & Civilian Vehicle Detection

> **Trạng thái:** khung model card. Phần kết quả huấn luyện/đánh giá **chưa có** vì các lượt train chưa
> chạy; những chỗ ghi "chưa đo" sẽ được điền từ `reports/` sau khi có số liệu thật. Không có con số nào
> trong file này được ước đoán.

## Tổng quan

| | |
|---|---|
| Nhiệm vụ | Phát hiện vật thể (bbox + lớp), 6 lớp |
| Lớp | `military truck`, `civilian car`, `military tank`, `civilian aircraft`, `military aircraft`, `military helicopter` |
| Họ model | Ultralytics YOLO (ứng viên: yolov5su, YOLO11n, YOLO11s; model cuối: **chưa chọn**) |
| Khởi tạo | Trọng số COCO pretrained của Ultralytics |
| Input | Ảnh RGB, letterbox về 640×640 (có thể 960 nếu thí nghiệm imgsz cho kết quả tốt hơn) |
| Phiên bản | chưa phát hành |

## Mục đích sử dụng

- **Dự kiến:** học tập, nghiên cứu, minh hoạ quy trình phát triển model computer vision; demo trên ảnh
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

Cấu hình trong `configs/exp*.yaml`: seed 0, `deterministic`, tối đa 20 epoch với early stopping
(patience 10), batch 16 (8 cho imgsz 960), AdamW/SGD tự chọn bởi Ultralytics (`optimizer=auto`), phần cứng
RTX 4060 Laptop 8 GB. Mỗi run lưu config, phiên bản thư viện và git commit
(`vehicledet_run.json`, MLflow). Tên model được chọn **chỉ dựa trên tập val**.

## Đánh giá

Quy ước: mAP tính bằng Ultralytics (IoU 0,5 và 0,5–0,95), `conf=0.001`, `iou=0.7` khi đo mAP.

### Kết quả trên tập test (model được chọn)

| Chỉ số | Giá trị |
|---|---|
| mAP50 | chưa đo |
| mAP50-95 | chưa đo |
| Precision / Recall | chưa đo |

### AP theo lớp

| Lớp | AP50 | AP50-95 |
|---|---|---|
| military truck | chưa đo | chưa đo |
| civilian car | chưa đo | chưa đo |
| military tank | chưa đo | chưa đo |
| civilian aircraft | chưa đo | chưa đo |
| military aircraft | chưa đo | chưa đo |
| military helicopter | chưa đo | chưa đo |

### Các lớp hay nhầm

**Chưa phân tích** (chưa có kết quả để phân tích lỗi). Giả thuyết cần kiểm chứng bằng confusion matrix:
các cặp cùng loại khác mục đích (military truck ↔ civilian car, military aircraft ↔ civilian aircraft,
tank ↔ truck) vì khác biệt chủ yếu ở màu sơn/ngụy trang và chi tiết nhỏ. Đây là giả thuyết, chưa phải
kết luận.

## Giới hạn đã biết

- **Ảnh web, ít đa dạng:** nhiều ảnh có watermark kho ảnh stock, minh hoạ/icon/render nhìn từ trên xuống;
  ảnh trùng khiến số ảnh hiệu dụng nhỏ hơn nhiều so với 6.772.
- **Vật thể nhỏ hiếm:** chỉ 3–11% bbox thuộc cỡ small (COCO), nên không có cơ sở khẳng định model tốt với
  vật thể nhỏ/xa.
- **Chưa kiểm tra** ảnh vệ tinh, UAV, hồng ngoại, ban đêm, thời tiết xấu, che khuất nặng.
- **Định nghĩa lớp mang tính hình thức:** "quân sự/dân sự" được gán theo vẻ ngoài; xe dân sự sơn ngụy
  trang hay xe quân sự sơn dân sự có thể bị phân loại sai.
- **Nhiễu nhãn nhỏ:** xem mục Dữ liệu. Không có đánh giá độc lập về độ chính xác của nhãn.
- **Tiền xử lý của API khác Ultralytics một chút** (Pillow vs OpenCV resize); ảnh hưởng tới mAP chưa đo
  với model thật.

## Sử dụng có trách nhiệm

Model chỉ là công cụ học tập. Mọi kết quả cần được con người kiểm tra; không dùng đầu ra làm căn cứ
duy nhất cho quyết định có hậu quả. Không dùng để xâm phạm quyền riêng tư hoặc nhắm mục tiêu vào người
hay phương tiện thật.

## License và ghi công

- Mã/weights dùng Ultralytics: **AGPL-3.0**.
- Dataset: CC BY 4.0, Priyanka Gupta, Bhavya Pareek, Gaurav Singal, D Vijay Rao (DRDO Institute for
  System Studies and Analyses), DOI 10.17632/njdjkbxdpn.1. Ảnh gốc có thể có bản quyền riêng.
