# RoadGuard AI

RoadGuard AI là ứng dụng thị giác máy tính phát hiện, phân đoạn và đo lường hư hỏng mặt đường từ ảnh hoặc video.

## Chức năng

- Phát hiện bốn nhóm hư hỏng RDD2022: `D00`, `D10`, `D20`, `D40`.
- Phân đoạn vết nứt và ổ gà bằng các model SEG chuyên biệt.
- Đo chiều dài nứt và diện tích ổ gà theo pixel hoặc mét/m² khi có hiệu chuẩn phối cảnh.
- Theo dõi hư hỏng trong video và tránh đếm lặp theo Track ID.
- Xuất báo cáo, gửi email, hỏi đáp theo kết quả và xem lại lịch sử kiểm tra.
- Tăng tốc suy luận bằng CUDA khi có GPU NVIDIA tương thích.

## Cấu trúc chính

```text
configs/          Cấu hình model và môi trường mẫu
models/runtime/   Ba checkpoint dùng bởi giao diện
notebooks/        Notebook huấn luyện và đánh giá
scripts/          Script khởi động, kiểm thử và xử lý dữ liệu
src/road_damage/  Mã nguồn pipeline dùng chung
tests/            Bộ kiểm thử tự động
web/              Giao diện kiểm thử local
```

Dataset, môi trường Python, video đầu vào, kết quả phân tích và run huấn luyện không được lưu trong repository.

## Clone

Các checkpoint runtime được quản lý bằng Git LFS.

```powershell
git lfs install
git clone <repository-url>
Set-Location RoadGuard-AI
git lfs pull
```

## Cài đặt

Yêu cầu Python 3.11. Các phiên bản thư viện tham chiếu nằm trong `requirements.lock.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
```

Nếu dùng GPU, hãy cài bản PyTorch CUDA phù hợp với driver trước khi cài các thư viện còn lại.

## Chạy giao diện

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_test_interface.ps1
```

Sau đó mở <http://127.0.0.1:8765>.

Trên Windows cũng có thể chạy `KHOI_DONG_ROADGUARD.bat`.

## Cấu hình tùy chọn

Sao chép `.roadguard.env.example` thành `.roadguard.env`, sau đó điền cấu hình chatbot hoặc SMTP. File `.roadguard.env` thật đã được loại khỏi Git.

Không ghi API key, mật khẩu SMTP hoặc dữ liệu cá nhân vào source code.

## Kiểm thử

```powershell
python -m pytest -q
```

Giao diện xử lý đầy đủ file video tải lên theo job nền. Đây không phải luồng camera/RTSP real-time; camera trực tiếp cần một pipeline streaming riêng.
