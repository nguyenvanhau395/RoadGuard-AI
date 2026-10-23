# Mô hình tốt dùng cho kiểm thử

Bộ checkpoint được chốt để chạy thử trong giao diện local. Các file trong `models/runtime` được quản lý bằng Git LFS để clone được trên máy khác mà không đưa toàn bộ thư mục huấn luyện lên GitHub.

| Vai trò | Model | Checkpoint sử dụng |
|---|---|---|
| DET | YOLOv8m | `models/runtime/roadguard_det.pt` |
| SEG Crack | YOLOv8s-seg | `models/runtime/crack_seg.pt` |
| SEG Pothole | YOLOv8m-seg | `models/runtime/pothole_seg.pt` |

## Sau khi clone

```powershell
git lfs install
git lfs pull
```

## Mapping của pipeline

DET dùng bốn mã lớp RDD2022: `D00` nứt dọc, `D10` nứt ngang, `D20` nứt da cá sấu và `D40` ổ gà. SEG Crack được gọi cho `D00`/`D10` để đo chiều dài skeleton theo pixel; SEG Pothole được gọi cho `D40` để đo diện tích mask theo px².

Giao diện đọc các đường dẫn tương đối này, nên vẫn chạy được khi clone project sang máy khác nếu Git LFS đã tải đủ ba checkpoint và môi trường đã cài `ultralytics`.
