# Phân tích lỗi (tập val)

Sinh bởi `python -m vehicledet analyze-errors`. Tham số: {"weights": "runs/detect/exp3c_yolo11n_v2_640_30ep/weights/best.pt", "split": "v2", "part": "val", "conf": 0.25, "iou_match": 0.5, "imgsz": 640, "seed": 0}

- Ảnh phân tích: **681**; số dự đoán: **1152**; GT (xấp xỉ): **1115**.
- Chỉ dùng tập **val** (không dùng test). Không sửa nhãn nào; mục "nhãn đáng ngờ" chỉ ghi nhận.

## 1. Số lượng theo loại lỗi

| Loại | Số lượng | Giải thích |
|---|---|---|
| tp | 830 | đúng lớp, IoU ≥ 0,5 |
| class_confusion | 76 | IoU ≥ 0,5 nhưng sai lớp |
| localization | 87 | đúng lớp nhưng IoU 0,1–0,5 |
| duplicate | 27 | trùng với GT đã được ghép |
| background_fp | 121 | dự đoán không chạm GT nào (nền) |
| other_fp | 11 | chạm GT khác lớp ở mức IoU thấp |
| miss | 122 | GT không được phát hiện (bỏ sót) |

## 2. Nhầm lớp (GT → dự đoán)

| GT | Dự đoán | Số lần |
|---|---|---|
| civilian aircraft | military aircraft | 28 |
| military aircraft | civilian aircraft | 7 |
| military tank | military truck | 7 |
| civilian car | military truck | 7 |
| military aircraft | military helicopter | 6 |
| military truck | civilian car | 6 |
| military helicopter | military aircraft | 5 |
| civilian aircraft | civilian car | 2 |
| military truck | military tank | 2 |
| civilian aircraft | military helicopter | 1 |

## 3. Bỏ sót (theo thẻ; một GT có thể có nhiều thẻ)

| Thẻ | Số GT |
|---|---|
| other | 54 |
| crowded | 49 |
| small | 31 |
| occluded | 28 |

## 4. Theo từng lớp

| Lớp | tp | bị nhầm sang lớp khác | bỏ sót | dự đoán nhầm vào lớp này | thừa (nền) |
|---|---|---|---|---|---|
| military truck | 136 | 9 | 22 | 14 | 20 |
| civilian car | 178 | 7 | 39 | 8 | 39 |
| military tank | 192 | 9 | 16 | 4 | 20 |
| civilian aircraft | 83 | 31 | 14 | 8 | 0 |
| military aircraft | 107 | 14 | 11 | 34 | 29 |
| military helicopter | 134 | 6 | 20 | 8 | 13 |

## 5. Nhãn đáng ngờ (chỉ ghi nhận, KHÔNG sửa)

Số ca: **12** (dự đoán điểm ≥ 0.7 không có nhãn, hoặc ≥ 0.8 nhưng khác lớp nhãn). Cần người kiểm tra tay trước khi kết luận nhãn sai.
- `boeing-787-800x415.jpg`: dự đoán `civilian car` (0.868); nhãn: `civilian aircraft` — high-confidence class disagreement
- `images308(10).jpg`: dự đoán `civilian car` (0.867); nhãn: `civilian aircraft` — high-confidence class disagreement
- `id_ltt-hmt_trailer_m1101_04_700.jpg`: dự đoán `military truck` (0.836); nhãn: `None` — high-confidence detection without label
- `images51(6).jpg`: dự đoán `military helicopter` (0.825); nhãn: `military aircraft` — high-confidence class disagreement
- `images318(2).jpg`: dự đoán `military truck` (0.819); nhãn: `military tank` — high-confidence class disagreement
- `images292(9).jpg`: dự đoán `civilian car` (0.807); nhãn: `None` — high-confidence detection without label
- `images240.jpg`: dự đoán `military truck` (0.805); nhãn: `military tank` — high-confidence class disagreement
- `taxi-police-cars-top-view-vector-set-auto-car-icon-illustration-vector-illustration-flat-design-top-view-police-taxi-image.jpg`: dự đoán `civilian car` (0.801); nhãn: `None` — high-confidence detection without label
- `images318(7).jpg`: dự đoán `military helicopter` (0.772); nhãn: `None` — high-confidence detection without label
- `aletta-wenas-helispread2-2forportfoliobox.jpg`: dự đoán `military helicopter` (0.745); nhãn: `None` — high-confidence detection without label
- `Cold-War-poster-equipment-Soviet-army.jpg`: dự đoán `military tank` (0.722); nhãn: `None` — high-confidence detection without label
- `images36(8).jpg`: dự đoán `civilian car` (0.721); nhãn: `None` — high-confidence detection without label

## 6. Gallery (xanh = nhãn thật, đỏ = dự đoán)

### Nhầm lớp


Ảnh: `boeing-787-800x415.jpg`, `images308(10).jpg`, `images51(6).jpg`, `images318(2).jpg`, `images240.jpg`, `images232(5).jpg`

### Bỏ sót: vật thể nhỏ


Ảnh: `images136(3).jpg`, `images195(6).jpg`, `images290(10).jpg`, `images288(4).jpg`, `airplanes-aircrafts-helicopters-set-air-260nw-1503096545.jpg`, `images79(4).jpg`

### Bỏ sót: bị che/chồng


Ảnh: `LaFontaine-Global-Vehicles-Home-Image-1200x450_c.jpg`, `m1a2-abrams-battle-tank-01.jpg`, `india-arunachal-pradesh-tenga-cantonment-army-recovery-truck-recovering-CC74CB.jpg`, `gbp-jp-2016769.jpg`, `images607.jpg`, `images247(8).jpg`

### Bỏ sót: ảnh nhiều vật thể


Ảnh: `images45(8).jpg`, `india-arunachal-pradesh-tenga-cantonment-army-recovery-truck-recovering-CC74CB.jpg`, `images58(9).jpg`, `Cold-War-poster-equipment-Soviet-army.jpg`, `gbp-jp-2016769.jpg`, `1200px-ALCOB_Ashok_Leyland_Corporate_Building_in_Guindy2C_Chennai.jpg`

### Bỏ sót: không thuộc nhóm trên


Ảnh: `1(2).jpg`, `Vehicles_850x460_acf_cropped.jpg`, `images10(7).jpg`, `images48(5).jpg`, `BG-Museums-1200x343.jpg`, `Types-of-Vehicle..jpg`

### Phát hiện thừa (nền)


Ảnh: `id_ltt-hmt_trailer_m1101_04_700.jpg`, `images292(9).jpg`, `taxi-police-cars-top-view-vector-set-auto-car-icon-illustration-vector-illustration-flat-design-top-view-police-taxi-image.jpg`, `images318(7).jpg`, `aletta-wenas-helispread2-2forportfoliobox.jpg`, `Cold-War-poster-equipment-Soviet-army.jpg`

### Dự đoán trùng lặp


Ảnh: `images263(1).jpg`, `Airbus-Commercial-Aircraft-formation-flight-3.jpg`, `images100(5).jpg`, `images242(1).jpg`, `images304(1).jpg`, `images239(8).jpg`

### Định vị sai (đúng lớp, IoU 0,1–0,5)


Ảnh: `images360(8).jpg`, `images347(9).jpg`, `gbp-jp-2016769.jpg`, `images245(7).jpg`, `2014chilec130aircraft-1575947699.jpg`, `images21(4).jpg`

### Nhãn đáng ngờ (dự đoán điểm cao không khớp nhãn)


Ảnh: `boeing-787-800x415.jpg`, `images308(10).jpg`, `id_ltt-hmt_trailer_m1101_04_700.jpg`, `images51(6).jpg`, `images318(2).jpg`, `images292(9).jpg`

