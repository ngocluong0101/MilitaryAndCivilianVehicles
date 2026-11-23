# Dataset card – Military and Civilian Vehicles Classification

Mọi số liệu dưới đây lấy từ `reports/` (sinh bởi `python -m vehicledet audit | find-duplicates |
make-split`), trừ các mục ghi rõ "theo trang Mendeley".

## Nguồn & license

| | |
|---|---|
| Tên | Military and Civilian Vehicles Classification |
| Tác giả (theo trang Mendeley) | Priyanka Gupta, Bhavya Pareek, Gaurav Singal, D Vijay Rao (DRDO Institute for System Studies and Analyses) |
| Phát hành | 1/6/2021, version 1, DOI `10.17632/njdjkbxdpn.1` |
| URL | https://data.mendeley.com/datasets/njdjkbxdpn/1 |
| License (theo trang Mendeley) | **CC BY 4.0** – được chia sẻ/chỉnh sửa, phải ghi công tác giả |
| Trang không nêu | cách thu thập ảnh, bản quyền từng ảnh gốc, độ phân giải, công cụ gán nhãn, cách chia train/test, trích dẫn mẫu |

Ảnh trong dataset trông như thu thập từ web (nhiều ảnh có watermark của kho ảnh stock, minh hoạ/icon
nhìn từ trên xuống). Trang Mendeley không nói rõ nguồn, nên **bản quyền ảnh gốc chưa được xác minh**:
chỉ dùng cho nghiên cứu/học tập, không phân phối lại ảnh. Dữ liệu không nằm trong repo.

## Lớp

`0 military truck`, `1 civilian car`, `2 military tank`, `3 civilian aircraft`,
`4 military aircraft`, `5 military helicopter` (định nghĩa duy nhất ở `configs/data.yaml`).

## Cấu trúc & định dạng

`Images/` (6.772 ảnh `.jpg`), `Labels/{CSV Format, TXT format, XML Format, TF Records Format}`.
Pipeline dùng **TXT (YOLO)** làm nhãn chính và CSV để đối chiếu; XML/TFRecord không được kiểm tra.
CSV có 2 file: `train_labels.csv` (6.705 ảnh) và `test_labels.csv` (496 ảnh) – 496 ảnh này **nằm trọn
trong** `train_labels.csv`.

## Thống kê (nhãn TXT, `reports/data_audit.json`)

| Chỉ số | Giá trị |
|---|---|
| Ảnh | 6.772 (0 ảnh hỏng) |
| Ảnh có ≥1 bbox | 6.705 |
| Ảnh không có bbox (file TXT rỗng) | 67 |
| Bbox hợp lệ | 10.718 (trung bình 1,6/ảnh, median 1, p95 4, max 52) |

| Lớp | Bbox | % | Ảnh chứa lớp | small | medium | large |
|---|---|---|---|---|---|---|
| military truck | 1.426 | 13,3% | 1.003 | 5% | 22% | 73% |
| civilian car | 2.164 | 20,2% | 1.066 | 9% | 37% | 53% |
| military tank | 2.428 | 22,7% | 1.646 | 4% | 19% | 77% |
| civilian aircraft | 1.373 | 12,8% | 971 | 3% | 22% | 74% |
| military aircraft | 1.642 | 15,3% | 1.004 | 11% | 28% | 60% |
| military helicopter | 1.685 | 15,7% | 1.201 | 4% | 23% | 73% |

(small/medium/large theo chuẩn COCO, diện tích bbox trên ảnh gốc: < 32², < 96², còn lại.)

- **Mất cân bằng lớp nhẹ**: lớp nhiều nhất / ít nhất ≈ 1,8 lần theo bbox (tank 2.428 vs civilian
  aircraft 1.373). Ít vật thể nhỏ (3–11%), phần lớn bbox lớn; ảnh thường chỉ có 1 vật thể.
- **Số liệu theo trang Mendeley khác số đếm từ file**: trang ghi 6.642 ảnh dương, 130 ảnh âm, 11.528
  đối tượng; đếm từ nhãn TXT thật cho 6.705 ảnh có bbox, 67 ảnh không bbox, 10.718 bbox hợp lệ. Chưa rõ
  nguyên nhân (chưa đối chiếu được); mọi số liệu của project dùng số đếm từ file.

## Vấn đề phát hiện & cách xử lý

| # | Vấn đề | Mức độ | Xử lý |
|---|---|---|---|
| 1 | **Ảnh trùng/gần trùng rất nhiều**: 4.853/6.772 ảnh (71,7%) thuộc 2.345 nhóm gần trùng (nhóm lớn nhất 5 ảnh); chỉ ≈4.264 "bức ảnh khác nhau". 171 nhóm trùng byte (346 ảnh). Biến thể gồm khác watermark, màu, độ phân giải, crop/zoom | **Nghiêm trọng** | Chia tập theo nhóm (mục dưới) |
| 2 | **Rò rỉ trong cách chia cũ**: 311 nhóm trải qua cả train và val; 322/496 ảnh val (64,9%) có bản gần trùng trong train. Đánh giá trên val cũ sẽ thổi phồng mAP | **Nghiêm trọng** | Chia lại (`splits/v2`); cách chia cũ đã gỡ khỏi repo (số liệu rò rỉ còn trong `reports/duplicates.md`) |
| 3 | Chưa có test độc lập (val cũ = `test_labels.csv`) | Cao | `splits/v2` có train/val/test |
| 4 | 4 file TXT lặp nguyên khối dòng (36 dòng trùng hệt: `images348`, `images358`, `images58`, `Kona_Pic_1-770x433`); CSV đúng | Thấp | Không sửa file; Ultralytics tự loại dòng trùng khi nạp nhãn |
| 5 | CSV vs TXT: 6.624 ảnh khớp; 74 ảnh có bbox lệch nhẹ (IoU 0,5–0,9; mẫu quan sát thấp nhất 0,62); 4 ảnh lệch số bbox (chính là 4 ảnh ở mục 4) | Thấp | Dùng TXT; chỉ ghi nhận |
| 6 | 3 tên file trong CSV không có ảnh trên đĩa (`1convoy.jpg`, `h1.jpg`, `ripsaw.jpg`). Bbox khớp 1-1 với 3 ảnh có TXT nhưng không có trong CSV ⇒ rất có thể đã bị đổi tên. Script chia cũ bỏ 3 ảnh này | Thấp | `v2` dùng nhãn TXT nên các ảnh này được giữ |
| 7 | 67 ảnh không có bbox (áo, quảng cáo, bản vẽ kỹ thuật, linh kiện…). Xem tay cả 67 ảnh: gần như đều là ảnh nền đúng nghĩa; vài ảnh nhỏ/khó (ảnh bản vẽ, xe kéo) có thể gây tranh cãi | Thông tin | Giữ làm ảnh nền trong `v2` (stratum riêng: 53/7/7 ảnh train/val/test). Chưa dùng trong cách chia cũ |
| 8 | 3 bbox rất nhỏ (cạnh < 4 px: `images125(8)`, `images340(6)`, `images352(3)`) | Thấp | Giữ nguyên, ghi nhận |
| 9 | Không có nhãn lỗi cứng: 0 class id ngoài 0–5, 0 toạ độ ngoài [0,1], 0 bbox kích thước 0, 0 file nhãn thiếu/thừa, 0 ảnh hỏng, 0 ảnh sai kích thước so với CSV | – | – |

Chưa sửa nhãn nào. Chi tiết: `reports/data_audit.md`, `reports/duplicates.md`.

## Phương pháp phát hiện ảnh trùng

pHash 16×16 (256 bit): hai ảnh cùng nhóm nếu Hamming ≤ 56, **hoặc** pHash 8×8 cách nhau ≤ 4 (bắt thêm
ảnh bị zoom/crop); nhóm là bao đóng bắc cầu. Ngưỡng chọn bằng cách xem ảnh theo từng mức khoảng cách;
số nhóm gần như phẳng trong khoảng 8–56 và bắt đầu bị nối dây chuyền từ 64 (bảng độ nhạy trong
`reports/duplicates.md`). Có ví dụ ảnh minh hoạ ở `reports/duplicates/`.

Hạn chế: pHash không bắt được ảnh lật ngang hoặc crop mạnh; một số nhóm gộp ảnh chỉ "cùng mẫu"
(vd ô tô nhìn từ trên xuống khác màu) – hậu quả duy nhất là chúng phải nằm chung một tập.

## Chia tập `splits/v2` (seed 0, tỉ lệ 80/10/10)

- Stratify theo **lớp chính của ảnh** (lớp có nhiều bbox nhất; hoà → diện tích lớn hơn → id nhỏ hơn;
  ảnh không bbox = "nền"). Mỗi nhóm gần trùng được gán nguyên khối cho một tập (tham lam theo mức
  thiếu hụt từng lớp).
- Không loại ảnh nào (0 ảnh hỏng/lỗi nhãn).

| Tập | Ảnh | Bbox | Ảnh nền |
|---|---|---|---|
| train | 5.412 | 8.535 | 53 |
| val | 681 | 1.068 | 7 |
| test | 679 | 1.115 | 7 |

Phân bố lớp từng tập: `reports/split_stats.md`.

**Kiểm tra rò rỉ** (`reports/split_stats.json`):

| Cách chia | Nhóm gần trùng nằm ở >1 tập | Cặp gần trùng giữa các tập theo dHash 8×8 (kiểm tra độc lập) |
|---|---|---|
| `legacy` | 311 | 348 |
| `v2` | **0** | 14 |

14 cặp còn lại ở `v2` đã được xem tay: đều là ảnh khác nhau (ảnh sprite/icon xe nhìn từ trên xuống
có bố cục giống nhau, dHash dễ báo nhầm), không phải cùng một bức ảnh. Lưu ý: với ảnh gần trùng
trong cùng một tập (vd 3.870 ảnh train có bản gần trùng trong train) thì train vẫn chứa nhiều bản
sao; điều này làm train "ít đa dạng" hơn số ảnh nhưng không gây rò rỉ.

Tái tạo: `python -m vehicledet audit && python -m vehicledet find-duplicates && python -m vehicledet make-split`
(đã kiểm tra: chạy lại cho file `splits/v2/*.txt` giống hệt – md5 trùng).

## Lưu ý sử dụng

- Dataset cho nghiên cứu/học tập; ảnh có thể có bản quyền riêng. Ghi công tác giả (CC BY 4.0).
- Ảnh web, nhiều ảnh minh hoạ/icon/render, nên mô hình có thể không tổng quát hoá sang ảnh vệ tinh,
  UAV, ban đêm, hay phương tiện bị che khuất nặng.
