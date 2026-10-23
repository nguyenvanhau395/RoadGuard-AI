# Mô hình tốt dùng cho kiểm thử

Bộ checkpoint được chốt để chạy thử trong giao diện local. Các file runtime được quản lý bằng Git LFS.

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

Giao diện test đọc đúng các đường dẫn tương đối này, nên vẫn chạy được khi clone project sang máy khác nếu Git LFS đã tải đủ ba checkpoint và đã cài `ultralytics`.
