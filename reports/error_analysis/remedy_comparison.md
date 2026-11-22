# Thí nghiệm khắc phục: tăng trọng số loss phân loại (exp6) – đo trên VAL

Giả thuyết: nhóm lỗi model lớn nhất là nhầm lớp (trội nhất civilian aircraft → military aircraft), nên tăng
trọng số loss phân loại `cls` từ 0,5 (mặc định) lên 1,0 có thể giảm nhầm lớp. Chỉ đổi một tham số so với
exp3c (cùng 30 epoch, seed 0). **Kết quả: không cải thiện.**

Nguồn: `reports/experiments/exp3c_*/val.json`, `reports/experiments/exp6_*/val.json`,
`reports/error_analysis/error_analysis.json` (exp3c), `reports/error_analysis/exp6_cls1/error_analysis.json`.

| Chỉ số (val) | exp3c (cls 0,5) | exp6 (cls 1,0) |
|---|---|---|
| mAP50 | 0,788 | 0,781 |
| mAP50-95 | **0,480** | 0,469 |
| Precision | 0,760 | 0,805 |
| Recall | 0,755 | 0,707 |
| Tổng ca nhầm lớp (conf 0,25) | 76 | 79 |
| civilian aircraft → military aircraft | 28 | 29 |
| military aircraft → civilian aircraft | 7 | 6 |
| Bỏ sót | 122 | 143 |
| Phát hiện thừa (nền) | 121 | 111 |

Cách đọc: tăng `cls` làm model "thận trọng" hơn (precision tăng, recall giảm, bỏ sót tăng) nhưng **không giảm
nhầm lớp** và mAP50-95 giảm 0,011. Với một seed và val ~680 ảnh, mức chênh này nằm trong khoảng dao động, nên
kết luận chính xác là "không có bằng chứng cải thiện", không phải "chắc chắn tệ hơn". Model cuối vẫn là exp3c.
Exp6 chỉ được đo trên val (không chạm tập test, vì test đã được chạy một lần cho exp3c).

Cách khắc phục khác chưa thử (đề xuất): (1) imgsz 960 để giữ chi tiết phân biệt quân sự/dân sự và giảm bỏ sót
vật thể nhỏ; (2) bổ sung nhãn cho các vật thể phụ chưa gán nhãn và rà soát các ca nhãn mơ hồ (cần bạn đồng ý
trước khi sửa nhãn).
