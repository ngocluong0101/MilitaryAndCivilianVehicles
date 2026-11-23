# Phân tích lỗi (tập val)

Sinh bởi `python -m vehicledet analyze-errors`. Tham số: {"weights": "runs/detect/exp6_yolo11n_v2_640_30ep_cls1/weights/best.pt", "split": "v2", "part": "val", "conf": 0.25, "iou_match": 0.5, "imgsz": 640, "seed": 0}

- Ảnh phân tích: **681**; số dự đoán: **1153**; GT (xấp xỉ): **1128**.
- Chỉ dùng tập **val** (không dùng test). Không sửa nhãn nào; mục "nhãn đáng ngờ" chỉ ghi nhận.

## 1. Số lượng theo loại lỗi

| Loại | Số lượng | Giải thích |
|---|---|---|
| tp | 822 | đúng lớp, IoU ≥ 0,5 |
| class_confusion | 79 | IoU ≥ 0,5 nhưng sai lớp |
| localization | 84 | đúng lớp nhưng IoU 0,1–0,5 |
| duplicate | 43 | trùng với GT đã được ghép |
| background_fp | 111 | dự đoán không chạm GT nào (nền) |
| other_fp | 14 | chạm GT khác lớp ở mức IoU thấp |
| miss | 143 | GT không được phát hiện (bỏ sót) |

## 2. Nhầm lớp (GT → dự đoán)

| GT | Dự đoán | Số lần |
|---|---|---|
| civilian aircraft | military aircraft | 29 |
| military tank | military truck | 8 |
| military helicopter | military aircraft | 8 |
| military truck | civilian car | 6 |
| military aircraft | civilian aircraft | 6 |
| military truck | military tank | 5 |
| military aircraft | military helicopter | 5 |
| civilian car | military truck | 3 |
| civilian aircraft | civilian car | 2 |
| military tank | military helicopter | 1 |

## 3. Bỏ sót (theo thẻ; một GT có thể có nhiều thẻ)

| Thẻ | Số GT |
|---|---|
| other | 74 |
| crowded | 45 |
| occluded | 33 |
| small | 33 |

## 4. Theo từng lớp

| Lớp | tp | bị nhầm sang lớp khác | bỏ sót | dự đoán nhầm vào lớp này | thừa (nền) |
|---|---|---|---|---|---|
| military truck | 135 | 12 | 26 | 11 | 21 |
| civilian car | 175 | 3 | 46 | 9 | 33 |
| military tank | 191 | 10 | 23 | 7 | 20 |
| civilian aircraft | 83 | 31 | 18 | 8 | 2 |
| military aircraft | 107 | 12 | 13 | 38 | 25 |
| military helicopter | 131 | 11 | 17 | 6 | 10 |

## 5. Nhãn đáng ngờ (chỉ ghi nhận, KHÔNG sửa)

Số ca: **7** (dự đoán điểm ≥ 0.7 không có nhãn, hoặc ≥ 0.8 nhưng khác lớp nhãn). Cần người kiểm tra tay trước khi kết luận nhãn sai.
- `images292(9).jpg`: dự đoán `civilian car` (0.833); nhãn: `None` — high-confidence detection without label
- `taxi-police-cars-top-view-vector-set-auto-car-icon-illustration-vector-illustration-flat-design-top-view-police-taxi-image.jpg`: dự đoán `civilian car` (0.82); nhãn: `None` — high-confidence detection without label
- `id_ltt-hmt_trailer_m1101_04_700.jpg`: dự đoán `military truck` (0.806); nhãn: `None` — high-confidence detection without label
- `bus-ferry-water-taxi-car-top-view.jpg`: dự đoán `civilian car` (0.771); nhãn: `None` — high-confidence detection without label
- `images261(2).jpg`: dự đoán `military tank` (0.709); nhãn: `None` — high-confidence detection without label
- `images300.jpg`: dự đoán `military tank` (0.709); nhãn: `None` — high-confidence detection without label
- `Cold-War-poster-equipment-Soviet-army.jpg`: dự đoán `military tank` (0.705); nhãn: `None` — high-confidence detection without label

## 6. Gallery (xanh = nhãn thật, đỏ = dự đoán)

### Nhầm lớp


Ảnh: `boeing-787-800x415.jpg`, `landscape-1488485223-gigahorse.jpg`, `clouds-minimalism-the-plane-fighter-wallpaper-preview.jpg`, `images294(4).jpg`, `images51(6).jpg`, `images174(11).jpg`

### Bỏ sót: vật thể nhỏ


Ảnh: `images195(6).jpg`, `images290(10).jpg`, `airplanes-aircrafts-helicopters-set-air-260nw-1503096545.jpg`, `images190(9).jpg`, `images607.jpg`, `images239(8).jpg`

### Bỏ sót: bị che/chồng


Ảnh: `LaFontaine-Global-Vehicles-Home-Image-1200x450_c.jpg`, `images299(1).jpg`, `india-arunachal-pradesh-tenga-cantonment-army-recovery-truck-recovering-CC74CB.jpg`, `gbp-jp-2016769.jpg`, `images239(8).jpg`, `images236(10).jpg`

### Bỏ sót: ảnh nhiều vật thể


Ảnh: `india-arunachal-pradesh-tenga-cantonment-army-recovery-truck-recovering-CC74CB.jpg`, `1200px-ALCOB_Ashok_Leyland_Corporate_Building_in_Guindy2C_Chennai.jpg`, `gbp-jp-2016769.jpg`, `images58(9).jpg`, `images36(8).jpg`, `images45(8).jpg`

### Bỏ sót: không thuộc nhóm trên


Ảnh: `images255(10).jpg`, `images300(6).jpg`, `images10(7).jpg`, `images131(4).jpg`, `images251(5).jpg`, `image(5).jpg`

### Phát hiện thừa (nền)


Ảnh: `images292(9).jpg`, `taxi-police-cars-top-view-vector-set-auto-car-icon-illustration-vector-illustration-flat-design-top-view-police-taxi-image.jpg`, `id_ltt-hmt_trailer_m1101_04_700.jpg`, `bus-ferry-water-taxi-car-top-view.jpg`, `images261(2).jpg`, `images300.jpg`

### Dự đoán trùng lặp


Ảnh: `0f0c81a7303883289bd9d0e9b468b834.jpg`, `images304(1).jpg`, `images230(1).jpg`, `51By3Q12BIDL._SX355_.jpg`, `images208(6).jpg`, `depositphotos_174265582-stock-illustration-blueprint-of-realistic-tank-top.jpg`

### Định vị sai (đúng lớp, IoU 0,1–0,5)


Ảnh: `images347(9).jpg`, `gbp-jp-2016769.jpg`, `images360(8).jpg`, `images70(11).jpg`, `images405(1).jpg`, `L242028129.jpg`

### Nhãn đáng ngờ (dự đoán điểm cao không khớp nhãn)


Ảnh: `images292(9).jpg`, `taxi-police-cars-top-view-vector-set-auto-car-icon-illustration-vector-illustration-flat-design-top-view-police-taxi-image.jpg`, `id_ltt-hmt_trailer_m1101_04_700.jpg`, `bus-ferry-water-taxi-car-top-view.jpg`, `images261(2).jpg`, `images300.jpg`

