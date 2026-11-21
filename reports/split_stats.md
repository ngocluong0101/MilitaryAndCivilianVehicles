# Thống kê chia tập & rò rỉ

Sinh bởi `python -m vehicledet make-split`. Tham số: {"name": "v2", "seed": 0, "ratios": [0.8, 0.1, 0.1], "dup_threshold": 56, "dup_hash": "phash"}

## Cách chia `legacy`

| Tập | Ảnh | Bbox | Ảnh nền (không bbox) |
|---|---|---|---|
| train | 6,206 | 9,997 | 0 |
| val | 496 | 718 | 0 |

Số ảnh theo lớp chính / số bbox theo lớp:

| Lớp | train | val |
|---|---|---|
| military truck | 884 / 1298 | 77 / 127 |
| civilian car | 971 / 2061 | 61 / 103 |
| military tank | 1505 / 2261 | 110 / 166 |
| civilian aircraft | 879 / 1295 | 69 / 78 |
| military aircraft | 909 / 1536 | 71 / 105 |
| military helicopter | 1058 / 1546 | 108 / 139 |

- Nhóm gần trùng (pHash 16x16) nằm ở >1 tập: **311**
- Ảnh tập `val` có bản gần trùng trong `train`: **322**
- Ảnh tập `train` có bản gần trùng trong `val`: **345**
- Kiểm tra chéo độc lập bằng dHash 8x8 (Hamming ≤ 4): **348** cặp nằm ở hai tập khác nhau

## Cách chia `v2`

| Tập | Ảnh | Bbox | Ảnh nền (không bbox) |
|---|---|---|---|
| train | 5,412 | 8,535 | 53 |
| val | 681 | 1,068 | 7 |
| test | 679 | 1,115 | 7 |

Số ảnh theo lớp chính / số bbox theo lớp:

| Lớp | train | val | test |
|---|---|---|---|
| military truck | 769 / 1128 | 97 / 169 | 96 / 129 |
| civilian car | 824 / 1678 | 104 / 236 | 104 / 250 |
| military tank | 1292 / 1950 | 162 / 226 | 162 / 252 |
| civilian aircraft | 758 / 1099 | 95 / 136 | 95 / 138 |
| military aircraft | 784 / 1330 | 99 / 136 | 98 / 176 |
| military helicopter | 932 / 1350 | 117 / 165 | 117 / 170 |

- Nhóm gần trùng (pHash 16x16) nằm ở >1 tập: **0**
- Kiểm tra chéo độc lập bằng dHash 8x8 (Hamming ≤ 4): **14** cặp nằm ở hai tập khác nhau

