# Quay GIF demo cho README

Mục tiêu: một GIF ngắn (≈ 10–15 giây, ≤ ~5 MB) cho thấy tải ảnh lên → ảnh có bbox → bảng kết quả → thời
gian suy luận. **Chỉ dùng ảnh bạn có quyền sử dụng** (ảnh của dataset có thể có bản quyền riêng, tránh đưa
vào README công khai).

1. Chạy demo với model đã export: `python -m vehicledet demo --model exports/<tên>/best.onnx`
   rồi mở http://127.0.0.1:7860.
2. Quay màn hình vùng trình duyệt (OBS Studio, hoặc `peek`/`byzanz` trên Linux, hoặc công cụ quay màn hình
   của hệ điều hành) ở độ phân giải khoảng 1000×600.
3. Kịch bản gợi ý: tải 2–3 ảnh khác nhau (xe tải, máy bay, ô tô) → kéo thanh ngưỡng tin cậy một lần →
   dừng 2 giây ở kết quả cuối.
4. Chuyển sang GIF nhỏ (nếu quay ra mp4/webm):
   ```bash
   ffmpeg -i demo.mp4 -vf "fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" demo.gif
   ```
5. Lưu thành `docs/demo.gif` và thêm vào README mục Demo: `![demo](docs/demo.gif)`.
