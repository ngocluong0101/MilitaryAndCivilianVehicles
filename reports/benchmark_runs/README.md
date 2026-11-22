# Các lần đo benchmark và độ dao động của GPU laptop

Model: `exp3c` (YOLO11n, 30 epoch). Input 1×3×640×640 (letterbox), RTX 4060 Laptop 8 GB, driver 595.99.02,
torch 2.15.0+cu132, TensorRT 10.16.1.11, ONNX Runtime GPU 1.32.0. Latency = thời gian model cho 1 ảnh
(`speed['inference']` của Ultralytics, batch 1); throughput = ảnh/giây của cả lời gọi predict (gồm tiền/hậu
xử lý). **Bảng chính thức là `../benchmark.csv`** (3 vòng, mỗi lần một tiến trình riêng); thư mục này giữ các lần
đo trước và quan sát giải thích vì sao phải đo như vậy.

## Phát hiện: xung nhịp GPU laptop làm latency dao động 2–3 lần

Trong lúc đo, xung SM của GPU được ghi lại (NVML / `nvidia-smi`). Có hai trạng thái lặp lại được:

| Trạng thái | Xung SM | Cờ giới hạn | TensorRT FP16 (b1 p50) | TensorRT INT8 (b1 p50) |
|---|---|---|---|---|
| **Bị hãm** (tải nhẹ, ngắn) | ~780 MHz (trung vị; tối đa 2490 khi có đợt tải nặng) | `0x24` = giới hạn công suất phần mềm + nhiệt phần mềm, dù chỉ ~42 °C và ~13 W | 3,20–3,21 ms | 2,53–2,54 ms |
| **Xung cao** (tải nặng kéo dài) | 2250–2550 MHz | – | 1,10–1,11 ms | 1,14 ms |

Hệ quả: một lần đo duy nhất có thể rơi vào bất kỳ trạng thái nào, nên **không nên so sánh con số của các
định dạng đo ở hai trạng thái khác nhau**. Bảng chính thức vì vậy đo lặp 3 vòng xen kẽ (round-robin) và báo
cáo trung vị kèm khoảng min–max cùng xung SM trung vị. Điều kết luận được trong cả hai trạng thái: TensorRT
FP16 nhanh hơn PyTorch FP32 rõ rệt; độ chính xác (mAP) không phụ thuộc xung nhịp.

## Các lần đo (nguồn)

| Lần | File | Cách đo | Ghi chú |
|---|---|---|---|
| 1 | `run1_with_test_map_single_process.csv/json` | 5 định dạng trong **một** tiến trình, có mAP test | VRAM bị thổi phồng do dùng chung tiến trình (TensorRT FP16 1740 MB so với 1196 MB khi chạy riêng) |
| 2 | `run2_with_test_map_single_process.csv/json` | như lần 1 (lặp lại) | tái lập lần 1: INT8 0,87–1,0 ms; TensorRT FP16 b1 vẫn 3,2 ms |
| 3 | `run3_isolated_one_round.csv/json` | mỗi định dạng một tiến trình, 1 vòng | VRAM đúng (672 / 532 / 1308 / 1196 / 1056 MB); TensorRT FP16 1,096 ms (xung cao) |
| 4 | `../benchmark.csv/json` | mỗi định dạng một tiến trình, **3 vòng xen kẽ**, ghi xung SM | bảng chính thức; từ lần chạy sau `benchmark.json` còn lưu dữ liệu thô từng vòng (`rounds`) |

Các lần đo nhẹ sau đây chỉ được ghi từ log chạy, **không lưu file gốc** (chạy `benchmark --map-part none`, hoặc
`--only`): TensorRT FP16 3,21 ms (b16 ≈ 402–405 img/s), INT8 2,535–2,54 ms (b16 ≈ 494–501 img/s), xung SM trung
vị 780 MHz. Hai lần ghi xung (INT8 riêng; PyTorch FP32 + INT8 có đo mAP val) đều cho INT8 = 2,54 ms.

## Ghi chú về mAP

- mAP50-95 test của benchmark (batch 8, PyTorch FP32) là **0,5323**, còn `evaluate` (`reports/test_results.json`)
  là **0,5307** (batch 16): chênh 0,0016 do kích thước batch/đệm ảnh khi đo, không phải do model khác nhau.
  Dùng 0,5323 để so các định dạng với nhau (cùng quy trình đo), dùng 0,5307 khi nói "kết quả test của model".
- INT8: Ultralytics giữ 26 lớp ở đầu mạng ngoài INT8 để bảo toàn độ chính xác (có trong log export).
