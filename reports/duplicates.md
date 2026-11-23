# Ảnh trùng / gần trùng

Sinh bởi `python -m vehicledet find-duplicates` (danh sách nhóm đầy đủ: `duplicates.json`).

## Phương pháp

- pHash 16x16 (256 bit) trên ảnh RGB→xám; hai ảnh cùng nhóm nếu khoảng cách Hamming ≤ **56**, **hoặc** pHash 8x8 (64 bit) cách nhau ≤ **4** (bắt thêm ảnh bị zoom/crop); nhóm lấy bao đóng bắc cầu (union-find).
- Ngưỡng chọn bằng cách xem trực quan các cặp theo từng mức khoảng cách (mẫu 8–12 cặp mỗi mức): đến ~44 (/256) vẫn là cùng một bức ảnh (khác watermark, màu, độ phân giải, crop); từ ~45 bắt đầu lẫn ảnh cùng mẫu nhưng khác nội dung (vd ô tô nhìn từ trên xuống khác màu).
- Chọn **56**: số nhóm gần như phẳng trong khoảng 8–56 (xem bảng độ nhạy); từ 64 trở lên nhóm bắt đầu bị nối dây chuyền (nhóm lớn nhất tăng). Ưu tiên chống rò rỉ nên chấp nhận gộp thêm vài ảnh chỉ giống mẫu: hậu quả duy nhất là chúng buộc phải nằm chung một tập.

## Kết quả

| Chỉ số | Giá trị |
|---|---|
| Ảnh đã hash | 6,772 |
| Nhóm gần trùng (≥2 ảnh) | 2,345 |
| Ảnh nằm trong nhóm gần trùng | 4,853 (71.7%) |
| Ảnh không có bản gần trùng | 1,919 |
| Số "bức ảnh khác nhau" ước lượng | 4,264 |
| Nhóm lớn nhất | 5 ảnh |
| Ảnh chỉ được gom nhờ tiêu chí pHash 8x8 | 5 |
| Nhóm trùng byte (md5) | 171 nhóm (346 ảnh) |

### Độ nhạy theo ngưỡng (chỉ pHash 16x16)

| Ngưỡng Hamming (/256) | Nhóm | Ảnh trong nhóm | Nhóm lớn nhất |
|---|---|---|---|
| 0 | 737 | 1,506 | 4 |
| 8 | 2,295 | 4,717 | 5 |
| 16 | 2,315 | 4,760 | 5 |
| 24 | 2,323 | 4,784 | 5 |
| 32 | 2,329 | 4,802 | 5 |
| 40 | 2,330 | 4,809 | 5 |
| 48 | 2,340 | 4,833 | 5 |
| 56 | 2,343 | 4,848 | 5 |
| 64 | 2,358 | 4,893 | 7 |
| 72 | 2,359 | 4,930 | 11 |

## Rò rỉ trong cách chia cũ (`splits/legacy`)

- Nhóm gần trùng nằm ở cả train và val: **311**
- Ảnh `val` có bản gần trùng trong `train`: **322**
- Ảnh `train` có bản gần trùng trong `val`: **345**
- Ảnh `train` có bản gần trùng khác trong chính `train`: **4188**
- Ảnh `val` có bản gần trùng khác trong chính `val`: **68**

## Ví dụ (chữ đỏ = tập theo cách chia cũ, nếu có)

