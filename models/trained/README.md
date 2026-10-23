# Trained checkpoints

Thư mục này lưu các checkpoint cuối của những lượt huấn luyện lịch sử:

- `best*.pt`: checkpoint có kết quả validation tốt nhất của lượt chạy.
- `last*.pt`: checkpoint cuối, dùng khi cần tiếp tục huấn luyện.
- Cấu trúc con `runs/` và `archive/runs/` giữ lại nguồn gốc tương đối của từng lượt.

Các file được quản lý bằng Git LFS. Chạy `git lfs pull` sau khi clone.

Snapshot `epoch*.pt` không được đưa lên repository vì có tổng dung lượng trên 14 GB và phần lớn chỉ là mốc trung gian trùng lặp.
