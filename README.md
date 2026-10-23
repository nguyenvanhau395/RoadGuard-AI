# RoadGuard AI

RoadGuard AI là ứng dụng thị giác máy tính phát hiện, phân đoạn và đo lường hư hỏng mặt đường từ ảnh hoặc video.

## Chức năng

- Phát hiện bốn nhóm hư hỏng RDD2022: `D00`, `D10`, `D20`, `D40`.
- Phân đoạn vết nứt và ổ gà bằng các model SEG chuyên biệt.
- Đo chiều dài nứt và diện tích ổ gà theo pixel hoặc mét/m² khi có hiệu chuẩn phối cảnh.
- Theo dõi hư hỏng trong video và tránh đếm lặp theo Track ID.
- Xuất báo cáo, gửi email, hỏi đáp theo kết quả và xem lại lịch sử kiểm tra.
- Tăng tốc suy luận bằng CUDA khi có GPU NVIDIA tương thích.

> Kết quả mô hình chỉ hỗ trợ khảo sát. Quyết định sửa chữa cần được xác nhận ngoài hiện trường.

## Cấu trúc chính

```text
configs/          Cấu hình SEG, kiến trúc thí nghiệm và mẫu môi trường
models/runtime/   Ba checkpoint dùng bởi giao diện
notebooks/        Notebook thí nghiệm/tái lập kết quả cục bộ
scripts/          Script khởi động, kiểm thử và xử lý dữ liệu
src/road_damage/  Mã nguồn pipeline dùng chung
tests/            Bộ kiểm thử tự động
web/              Giao diện kiểm thử local
docs/             Tài liệu thiết kế và model đã chọn
```

Dataset, môi trường Python, video đầu vào, kết quả phân tích và run huấn luyện là artifact local, bị `.gitignore` loại khỏi Git.

Chỉ ba checkpoint runtime trong `models/runtime/*.pt` được theo dõi bằng Git LFS. Dataset, checkpoint huấn luyện, kết quả chạy và archive vẫn ở local để repository gọn và clone nhanh.

| Vai trò | Checkpoint | Mục đích |
|---|---|---|
| DET | `models/runtime/roadguard_det.pt` | Phát hiện `D00`, `D10`, `D20`, `D40` |
| SEG Crack | `models/runtime/crack_seg.pt` | Tạo mask để đo chiều dài vết nứt |
| SEG Pothole | `models/runtime/pothole_seg.pt` | Tạo mask để đo diện tích ổ gà |

Xem thêm [`docs/models.md`](docs/models.md).

## Yêu cầu

- Windows/PowerShell cho các launcher có sẵn; Python $3.11$; Git và Git LFS.
- GPU NVIDIA/CUDA là tùy chọn khi suy luận giao diện, nhưng bắt buộc cho training benchmark chuẩn.
- `imageio-ffmpeg` (được liệt kê trong requirements) mã hoá kết quả video thành H.264 để phát trên trình duyệt.
- `requirements.lock.txt` phản ánh môi trường CUDA/Conda tham chiếu. Nếu `pip` không cài được dependency `file:///...`, hãy dùng môi trường Conda tương thích hoặc tái tạo requirements portable.

## Clone

Các checkpoint runtime được quản lý bằng Git LFS.

```powershell
git lfs install
git clone <repository-url> RoadGuard-AI
Set-Location RoadGuard-AI
git lfs pull
```

## Cài đặt

Yêu cầu Python 3.11. Các phiên bản thư viện tham chiếu nằm trong `requirements.lock.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
pip install -e ".[dev]"
```

Nếu dùng GPU, hãy cài bản PyTorch CUDA phù hợp với driver trước khi cài các thư viện còn lại.

## Chạy giao diện local

Launcher ưu tiên `.venv`, sau đó mới dùng `.conda` nếu có.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_test_interface.ps1
```

Sau đó mở <http://127.0.0.1:8765>.

Trên Windows cũng có thể chạy `KHOI_DONG_ROADGUARD.bat`. Có thể chạy trực tiếp sau khi kích hoạt môi trường:

```powershell
python scripts/test_interface.py --host 127.0.0.1 --port 8765
```

Giao diện bind mặc định vào `127.0.0.1`, giới hạn upload 200 MB và lưu media, JSON report cùng lịch sử cục bộ ở `.interface_runs/`. Đây không phải dịch vụ production/multi-user hoặc camera RTSP thời gian thực.

## Cấu hình tùy chọn

Sao chép `.roadguard.env.example` thành `.roadguard.env`, sau đó điền cấu hình chatbot hoặc SMTP:

```powershell
Copy-Item .roadguard.env.example .roadguard.env
```

`CHATBOT_API_KEY` chỉ cần khi dùng chatbot qua API; cấu hình `REPORT_SMTP_*` chỉ cần khi gửi email. File `.roadguard.env` thật đã được loại khỏi Git.

Không ghi API key, mật khẩu SMTP hoặc dữ liệu cá nhân vào source code.

## Kiểm thử

```powershell
python -m pytest -q
```

Lệnh test yêu cầu đã cài extra `.[dev]` và đã tải đủ ba checkpoint bằng `git lfs pull`.

## Dữ liệu và huấn luyện

Thư mục `data/`, `runs/`, `reports/`, `archive/` và `models/trained/` là dữ liệu local, không commit. Raw dataset phải có `data.yaml` cùng ba split `train`, `valid` (hoặc `val`) và `test`; mỗi split gồm `images/` và `labels/`.

```text
data/raw/crack-4/      # dataset crack
data/raw/pothole-v2/   # dataset pothole
```

Chuẩn bị dataset không sửa dữ liệu raw: ảnh có label thiếu/lỗi bị loại khỏi manifest, label rỗng được chấp nhận là background. Lệnh dưới đây tạo manifest, audit JSON, preview và YAML có đường dẫn tuyệt đối cho máy hiện tại:

```powershell
python -c "from road_damage.dataset import prepare_dataset; print(prepare_dataset('crack'))"
python -c "from road_damage.dataset import prepare_dataset; print(prepare_dataset('pothole'))"
```

Các YAML trong `configs/seg/` chỉ là mẫu portable trước khi tạo manifest local.

## Pipeline và giới hạn đo lường

- `D00`/`D10` chỉ được đưa vào kết quả khi SEG Crack tạo mask hợp lệ; `D40` cần mask từ SEG Pothole. Detection không có mask không được thay bằng giá trị đo bằng 0. `D20` được báo cáo theo detection và chưa có phép đo SEG tương ứng.
- Video chạy DET và SEG trên mọi frame, sau đó dùng track ID để tránh đếm lặp. Đây không phải luồng RTSP/camera trực tiếp.
- Không hiệu chuẩn, chiều dài và diện tích chỉ có đơn vị `px`/`px²`. Khi hiệu chuẩn phối cảnh, cần bốn góc vùng mặt đường theo thứ tự gần trái → gần phải → xa phải → xa trái cùng kích thước thực tế. Với video, quy đổi chỉ đáng tin cậy khi tương quan camera–mặt đường ổn định.

## Huấn luyện SEG

Sau khi chuẩn bị dataset và có pretrained weights local trong `models/pretrained/`, chạy pilot an toàn:

```powershell
python -m road_damage.runner --stage pilot30
```

Queue chỉ cho phép một tiến trình training dùng GPU tại một thời điểm. `configs/architectures/` là cấu hình thử nghiệm/lịch sử, không phải cấu hình của ba checkpoint runtime. Kết quả training và report sinh ra không được commit.
