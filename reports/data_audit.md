# Báo cáo kiểm tra dữ liệu

Sinh tự động bởi `python -m vehicledet audit`; danh sách đầy đủ trong `data_audit.json`.

## 1. Tổng quan ảnh

| Chỉ số | Số lượng |
|---|---|
| Tổng số ảnh | 6,772 |
| Ảnh hỏng (không giải mã được) | 0 |
| Ảnh có ≥1 bbox hợp lệ | 6,705 |
| Ảnh không có file nhãn TXT | 0 |
| Ảnh có file TXT nhưng không có bbox hợp lệ | 67 |
| File TXT không có ảnh tương ứng | 0 |
| Ảnh có ≥1 dòng nhãn lỗi | 0 |

## 2. Bbox theo lớp

Tổng số bbox hợp lệ: **10,718**

| Lớp | Số bbox | % bbox | Số ảnh chứa lớp | small | medium | large |
|---|---|---|---|---|---|---|
| military truck | 1,426 | 13.3% | 1,003 | 74 (5%) | 312 (22%) | 1,040 (73%) |
| civilian car | 2,164 | 20.2% | 1,066 | 202 (9%) | 807 (37%) | 1,155 (53%) |
| military tank | 2,428 | 22.7% | 1,646 | 85 (4%) | 470 (19%) | 1,873 (77%) |
| civilian aircraft | 1,373 | 12.8% | 971 | 48 (3%) | 306 (22%) | 1,019 (74%) |
| military aircraft | 1,642 | 15.3% | 1,004 | 188 (11%) | 463 (28%) | 991 (60%) |
| military helicopter | 1,685 | 15.7% | 1,201 | 69 (4%) | 383 (23%) | 1,233 (73%) |

small/medium/large theo COCO: diện tích bbox < 32², < 96², còn lại (pixel², ảnh gốc).

Số bbox mỗi ảnh (ảnh có nhãn): min=1, mean=1.6, median=1, p95=4, max=52

| Số bbox/ảnh | 1 | 2 | 3-5 | 6-10 | >10 |
|---|---|---|---|---|---|
| Số ảnh | 5297 | 624 | 575 | 146 | 63 |

## 3. Nhãn hỏng / đáng ngờ

| Loại | Mức | Số dòng |
|---|---|---|
| `bad_format` | lỗi | 0 |
| `class_out_of_range` | lỗi | 0 |
| `coord_out_of_range` | lỗi | 0 |
| `zero_size` | lỗi | 0 |
| `box_exceeds_image` | cảnh báo | 0 |
| `duplicate_box` | cảnh báo | 36 |
| `tiny_box` | cảnh báo | 3 |

Ví dụ `duplicate_box`:

- `Kona_Pic_1-770x433.txt:2 1 0.457031 0.514583 0.882813 0.940278`
- `images348.txt:2 1 0.491406 0.477027 0.976563 0.916216`
- `images348.txt:3 1 0.491406 0.477027 0.976563 0.916216`
- `images348.txt:4 1 0.491406 0.477027 0.976563 0.916216`
- `images348.txt:5 1 0.491406 0.477027 0.976563 0.916216`

Ví dụ `tiny_box`:

- `images125(8).txt:3 1 0.801562 0.924324 0.010937 0.024324`
- `images340(6).txt:1 5 0.025781 0.038889 0.012500 0.019444`
- `images352(3).txt:3 3 0.405469 0.748649 0.104688 0.010811`

## 4. So sánh CSV với TXT

Số ảnh có trong CSV: **6,705**

| Trạng thái | Số ảnh |
|---|---|
| `count_mismatch` | 4 |
| `csv_image_missing_on_disk` | 3 |
| `match` | 6,624 |
| `minor_box_diff` | 74 |
| `txt_only_not_in_csv_empty` | 67 |
| `txt_only_not_in_csv_labeled` | 3 |

Ảnh có kích thước thật khác width/height trong CSV: **0**

Ví dụ `count_mismatch`:

- `Kona_Pic_1-770x433.jpg`
- `images348.jpg`
- `images358.jpg`
- `images58.jpg`

Ví dụ `csv_image_missing_on_disk`:

- `1convoy.jpg`
- `h1.jpg`
- `ripsaw.jpg`

Ví dụ `minor_box_diff`:

- `10(2).jpg`
- `9780124199538.jpg`
- `four-indian-army-trucks-blown-up-90-jawans-onboard-same-train-1525984925-5543.jpg`
- `gbp-jp-2016769.jpg`
- `hqdefault5.jpg`

Ví dụ `txt_only_not_in_csv_empty`:

- `77374810001_1741676573001_ari-origin06-arc-132-1342642334111.jpg`
- `957146777596170b2f6220ca74af9bc4x.jpg`
- `9999015903-l.jpg`
- `A-chip-carrier-is-shown-The-central-part-of-the-holder-is-raised-using_Q320.jpg`
- `ARMY_MOM__92199.1524853080.jpg`

Ví dụ `txt_only_not_in_csv_labeled`:

- `-H2FRPD.jpg`
- `-convoy-in-pulwama-on-thursday-february-14-which-killed-more-than-50-indian-army-men-credit-idrees-abbassopa-imageszuma-wirealamy-live-news-T3JP27.jpg`
- `ripsaw-ev3-f4-4-seater-ripsaw-ev2-price-cost-sherp-cost-luxury-super-tank-sherp-off-road-4x4-tank-army-for-sale-ssr-mini-ev2-engine-street-legal-howe-and-howe-ripsaw-top-speed-specs-interior_orig.jpg`

Ảnh có trong CSV nhưng thiếu trên đĩa, khả năng cao đã bị đổi tên (bbox + kích thước khớp 1-1 với một ảnh có TXT nhưng không có trong CSV):

- `1convoy.jpg` → `-convoy-in-pulwama-on-thursday-february-14-which-killed-more-than-50-indian-army-men-credit-idrees-abbassopa-imageszuma-wirealamy-live-news-T3JP27.jpg`
- `h1.jpg` → `-H2FRPD.jpg`
- `ripsaw.jpg` → `ripsaw-ev3-f4-4-seater-ripsaw-ev2-price-cost-sherp-cost-luxury-super-tank-sherp-off-road-4x4-tank-army-for-sale-ssr-mini-ev2-engine-street-legal-howe-and-howe-ripsaw-top-speed-specs-interior_orig.jpg`
