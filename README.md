# 🚦 Hệ Thống Giám Sát Giao Thông – Traffic Monitoring System

Hệ thống phát hiện xe vượt đèn đỏ sử dụng YOLOv10, OpenCV và thuật toán tracking tùy chỉnh.

---

## 📁 Cấu Trúc Dự Án

```
project/
│
├── train_xe.ipynb       # Notebook huấn luyện mô hình (chạy trên Google Colab)
├── main.py              # File chạy chính – phát hiện & ghi nhận vi phạm
├── light.py             # Module nhận diện màu đèn giao thông (HSV)
├── tracker.py           # Module theo dõi đối tượng (Euclidean Tracker)
├── coco.txt             # Danh sách 80 nhãn lớp COCO
│
├── youtube.mp4          # ← Video đầu vào (bạn tự chuẩn bị)
├── yolov10s.pt          # ← Trọng số YOLOv10 (tải về hoặc dùng model tự train)
│
└── output/
    ├── ket_qua.avi      # Video kết quả xuất ra
    └── vi_pham/         # Ảnh chụp màn hình xe vi phạm
        └── ID_<id>_<timestamp>.jpg
```

---

## 🖥️ Yêu Cầu Môi Trường

| Thư viện       | Phiên bản khuyến nghị |
|----------------|-----------------------|
| Python         | 3.9 – 3.11            |
| ultralytics    | ≥ 8.0                 |
| opencv-python  | ≥ 4.8                 |
| cvzone         | ≥ 1.6                 |
| pandas         | ≥ 1.5                 |
| numpy          | ≥ 1.24                |

---

## 🔧 BƯỚC 1 – Cài Đặt Môi Trường

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Cài đặt các thư viện
pip install ultralytics opencv-python cvzone pandas numpy
```

---

## 🏋️ BƯỚC 2 – Huấn Luyện Mô Hình (Google Colab)

> **Lưu ý:** Bước này chỉ cần thực hiện nếu bạn muốn tự huấn luyện mô hình với dataset riêng.  
> Bạn có thể **bỏ qua** và dùng thẳng `yolov10s.pt` từ [Ultralytics](https://github.com/THU-MIG/yolov10).

### 2.1 Chuẩn Bị Dataset

Cấu trúc dataset cần có dạng:

```
train_xe.zip
├── train/
│   ├── images/   # Ảnh huấn luyện (.jpg, .png)
│   └── labels/   # Nhãn YOLO format (.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Mỗi file nhãn `.txt` theo định dạng YOLO:
```
<class_id> <x_center> <y_center> <width> <height>
```

### 2.2 Upload Dataset Lên Google Drive

Nén toàn bộ dataset thành file `train_xe.zip` và upload vào:
```
Google Drive > My Drive > Colab Notebooks > train_xe.zip
```

### 2.3 Chạy Notebook Trên Google Colab

Mở file `train_xe.ipynb` trên Google Colab và chạy lần lượt từng cell:

| Cell | Mô tả |
|------|-------|
| **Cell 1** | Cài đặt thư viện `ultralytics` |
| **Cell 2** | Mount Google Drive |
| **Cell 3** | Giải nén dataset vào `/content/dataset` |
| **Cell 4** | Tạo file `data.yaml` cấu hình dataset |
| **Cell 5** | **Bắt đầu huấn luyện** – YOLOv10n, 50 epochs, imgsz=640 |
| **Cell 6** | Chuẩn hóa class ID về `0` trong tất cả file nhãn |
| **Cell 7** | Export model sang định dạng ONNX *(tùy chọn)* |
| **Cell 8** | Tải file `best.pt` về máy |

### 2.4 Lệnh Huấn Luyện (Chi Tiết)

```bash
# Lệnh được chạy tự động trong Cell 5
yolo detect train \
  data=/content/dataset/data.yaml \
  model=yolov10n.pt \
  epochs=50 \
  imgsz=640
```

Kết quả lưu tại: `runs/detect/train/weights/best.pt`

### 2.5 Tải Model Về Máy

```python
# Cell cuối trong notebook
from google.colab import files
files.download('/content/runs/detect/train2/weights/best.pt')
```

Sau khi tải về, **đổi tên** thành `yolov10s.pt` (hoặc cập nhật đường dẫn trong `main.py`).

---

## ▶️ BƯỚC 3 – Chạy Hệ Thống

### 3.1 Chuẩn Bị File Đầu Vào

Đảm bảo các file sau nằm cùng thư mục với `main.py`:

```
✅ youtube.mp4      – Video giao thông cần phân tích
✅ yolov10s.pt      – File trọng số mô hình YOLO
✅ coco.txt         – Danh sách class (đã có sẵn)
✅ light.py         – Module đèn (đã có sẵn)
✅ tracker.py       – Module tracker (đã có sẵn)
```

Tạo thư mục output (nếu chưa có):

```bash
mkdir -p output/vi_pham
```

### 3.2 Chạy File Chính

```bash
python main.py
```

### 3.3 Điều Khiển Khi Chạy

| Phím | Chức năng |
|------|-----------|
| `Q`  | Thoát chương trình |

---

## ⚙️ Cấu Hình & Tuỳ Chỉnh

### Thay Đổi Video Đầu Vào

Mở `main.py`, sửa dòng:

```python
video_path = 'youtube.mp4'   # ← Thay bằng đường dẫn video của bạn
```

### Thay Đổi Mô Hình YOLO

```python
model = YOLO("yolov10s.pt")  # ← Thay bằng tên file .pt của bạn
```

### Điều Chỉnh Vùng Kiểm Soát (Khu Vực Giao Lộ)

Trong `main.py`, thay đổi tọa độ polygon `area` cho phù hợp với video:

```python
# Mỗi tuple là (x, y) một đỉnh của vùng kiểm soát
area = [(324, 313), (283, 374), (854, 392), (864, 322)]
```

> **Mẹo:** Dùng công cụ [CVAT](https://app.cvat.ai) hoặc vẽ thử trực tiếp bằng OpenCV để xác định tọa độ chính xác.

### Điều Chỉnh Ngưỡng Màu Đèn

Mở `light.py` để chỉnh khoảng HSV nếu màu đèn trong video khác:

```python
# Màu XANH (Green)
lower_range = np.array([58, 97, 222])
upper_range = np.array([179, 255, 255])

# Màu ĐỎ (Red)
lower_range1 = np.array([0, 43, 184])
upper_range1 = np.array([56, 132, 255])
```

> **Mẹo:** Dùng script HSV picker của OpenCV để lấy đúng ngưỡng cho video của bạn.

### Điều Chỉnh Ngưỡng Tracking

Trong `tracker.py`, thay đổi khoảng cách Euclidean để nhận diện lại đối tượng:

```python
if dist < 35:   # ← Tăng nếu xe di chuyển nhanh, giảm nếu bị nhầm lẫn
```

---

## 📤 Kết Quả Đầu Ra

Sau khi chạy xong, kết quả được lưu tại:

```
output/
├── ket_qua.avi              # Toàn bộ video đã xử lý
└── vi_pham/
    ├── ID_3_14-22-05.jpg    # Ảnh xe vi phạm, ID=3, lúc 14:22:05
    ├── ID_7_14-22-18.jpg
    └── ...
```

Trên màn hình hiển thị:
- **Khung xanh**: Xe đang di chuyển bình thường
- **Khung đỏ + nhãn `VIOLATION #ID`**: Xe vi phạm đèn đỏ
- **Góc trái**: Tổng số vi phạm (`TOTAL VIOLATIONS`)
- **Góc phải**: Trạng thái đèn hiện tại (`LIGHT: RED / GREEN`)

---

## 🐛 Xử Lý Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
|-----|-------------|----------------|
| `ModuleNotFoundError: cvzone` | Thiếu thư viện | `pip install cvzone` |
| `Error: video not found` | Sai đường dẫn video | Kiểm tra lại `video_path` trong `main.py` |
| Không phát hiện đèn | Ngưỡng HSV không khớp | Điều chỉnh `lower_range` / `upper_range` trong `light.py` |
| Xe bị mất tracking | Ngưỡng khoảng cách quá nhỏ | Tăng giá trị `dist < 35` trong `tracker.py` |
| `CUDA out of memory` | GPU không đủ VRAM | Giảm `conf` hoặc dùng CPU: `model = YOLO(..., device='cpu')` |

---

## 📌 Sơ Đồ Luồng Hoạt Động

```
Video Frame
    │
    ├──► light.py         → Nhận diện màu đèn (RED / GREEN)
    │
    ├──► YOLOv10          → Phát hiện xe (car, truck, bus, motorcycle)
    │
    ├──► tracker.py       → Gán ID và theo dõi từng xe
    │
    └──► Kiểm tra vi phạm:
             Xe trong vùng area  +  Đèn RED
                     │
                     ▼
             Lưu ảnh vi phạm → output/vi_pham/
             Ghi video kết quả → output/ket_qua.avi
```

---

## 📄 Giấy Phép

Dự án sử dụng cho mục đích học tập và nghiên cứu.  
Model YOLOv10 thuộc bản quyền của [THU-MIG](https://github.com/THU-MIG/yolov10) – AGPL-3.0 License.
