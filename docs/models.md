# Model runtime được chọn

Ba checkpoint dưới đây là artifact model duy nhất được theo dõi qua Git LFS. Giao diện local dùng các đường dẫn tương đối nên chạy được sau khi clone và chạy `git lfs pull`.

| Vai trò | Model | Checkpoint |
|---|---|---|
| DET | YOLOv8m | `models/runtime/roadguard_det.pt` |
| SEG Crack | YOLOv8s-seg | `models/runtime/crack_seg.pt` |
| SEG Pothole | YOLOv8m-seg | `models/runtime/pothole_seg.pt` |

DET dùng bốn mã lớp RDD2022: `D00` nứt dọc, `D10` nứt ngang, `D20` nứt da cá sấu và `D40` ổ gà. SEG Crack phục vụ đo chiều dài theo skeleton; SEG Pothole phục vụ đo diện tích mask.

Checkpoint của các lần huấn luyện (`models/trained/`) và model pretrained được lưu local, không đưa vào Git. Điều này tránh commit các artifact thí nghiệm lặp lại và giữ repository có thể clone nhanh.