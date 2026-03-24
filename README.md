# CNN Nhận Diện Mèo - Chó (Cats vs Dogs)

## Mục Tiêu
Xây dựng mô hình CNN (Convolutional Neural Network) để phân loại ảnh mèo và chó.

## Cấu Trúc Dự Án
```
CNN_Cats_Dogs/
├── cats/              # Thư mục chứa ảnh mèo
├── dogs/              # Thư mục chứa ảnh chó
├── 1_data.py          # Bước 1: Load và lưu dữ liệu
├── 2_train.py         # Bước 2: Huấn luyện model
├── 3_predict.py       # Bước 3: Dự đoán ảnh mới
├── data.npz           # File dữ liệu đã xử lý (tự tạo)
└── model.keras        # File model đã train (tự tạo)
```

## Dataset
- **70 ảnh mèo** trong thư mục `cats/`
- **70 ảnh chó** trong thư mục `dogs/`
- Kích thước ảnh: 64x64 pixels

---

## Bước 1: Load Dữ Liệu

### File: `1_data.py`

```python
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = 64

# Load ảnh từ thư mục cats và dogs
# Resize về 64x64, chuẩn hóa pixel về 0-1
# Label: 0 = Mèo, 1 = Chó

# Chia train/test: 80%/20%
X_train, X_test, y_train, y_test = train_test_split(...)

# Lưu vào file data.npz
np.savez_compressed('data.npz', ...)
```

### Chạy:
```bash
python 1_data.py
```

### Kết quả:
```
Da load 140 anh
  - Meo: 70
  - Cho: 70
Kich thuoc anh: (64, 64, 3)

Train: 112 | Test: 28
Da luu data vao data.npz
```

---

## Bước 2: Huấn Luyện Model

### File: `2_train.py`

### Kiến Trúc CNN:

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT (64, 64, 3)                    │
│                     Ảnh màu 64x64                      │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              CONV2D(32 filters, 3x3)                    │
│              Activation: ReLU                            │
│              Output: (62, 62, 32)                        │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              MAXPOOLING2D (2x2)                         │
│              Output: (31, 31, 32)                        │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              CONV2D(64 filters, 3x3)                     │
│              Activation: ReLU                            │
│              Output: (29, 29, 64)                        │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              MAXPOOLING2D (2x2)                         │
│              Output: (14, 14, 64)                        │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              CONV2D(128 filters, 3x3)                   │
│              Activation: ReLU                           │
│              Output: (12, 12, 128)                      │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              MAXPOOLING2D (2x2)                         │
│              Output: (6, 6, 128)                        │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    FLATTEN                              │
│              Output: 6×6×128 = 4608 features            │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              DENSE(128 neurons)                         │
│              Activation: ReLU                           │
│              Dropout: 0.5 (giảm overfitting)            │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              DENSE(1 neuron)                            │
│              Activation: Sigmoid                         │
│              Output: 0 đến 1                             │
└─────────────────────────┴───────────────────────────────┘
                          ▼
              ┌───────────────────────┐
              │   OUTPUT: 0 = Mèo     │
              │          1 = Chó      │
              └───────────────────────┘
```

### Giải Thích Các Layer:

| Layer | Chức năng |
|-------|-----------|
| **Conv2D** | Trích xuất đặc trưng từ ảnh (edges, textures, shapes) |
| **MaxPooling2D** | Giảm kích thước, giữ features quan trọng |
| **Flatten** | Chuyển ma trận thành vector 1D |
| **Dense** | Classifier để phân loại |
| **Dropout** | Tắt ngẫu nhiên neurons để tránh overfitting |
| **Sigmoid** | Output xác suất (0 đến 1) |

### Chạy:
```bash
python 2_train.py
```

### Kết quả mong đợi:
```
Epoch 1/50
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.693 - accuracy: 0.50 - val_loss: 0.693 - val_accuracy: 0.50
Epoch 2/50
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - loss: 0.692 - accuracy: 0.52 - val_loss: 0.691 - val_accuracy: 0.55
...
Epoch 25/50
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - loss: 0.123 - accuracy: 0.95 - val_loss: 0.15 - val_accuracy: 0.94

=== KET QUA ===
Test Loss: 0.12
Test Accuracy: 0.96

Da luu model vao model.keras
```

---

## Bước 3: Dự Đoán

### File: `3_predict.py`

### Cách Hoạt Động:

```
1. Load ảnh mới
       │
       ▼
2. Resize về 64x64, chuẩn hóa
       │
       ▼
3. Đưa vào model CNN
       │
       ▼
4. Model trả về xác suất (0 đến 1)
       │
       ▼
5. Nếu > 0.5 → CHÓ
   Nếu ≤ 0.5 → MÈO
```

### Chạy:
```bash
python 3_predict.py
```

### Kết quả mong đợi:
```
=== TEST ANH MEO ===
cat_1.jpg: MEO (Cat) (0.95) - ✓
cat_2.jpg: MEO (Cat) (0.92) - ✓
...

=== TEST ANH CHO ===
dog_1.jpg: CHO (Dog) (0.88) - ✓
dog_2.jpg: CHO (Dog) (0.91) - ✓
...

Accuracy: 95.0% (133/140)
```

---

## Tổng Kết

| Bước | File | Output |
|------|------|--------|
| 1 | `1_data.py` | `data.npz` |
| 2 | `2_train.py` | `model.keras` |
| 3 | `3_predict.py` | Kết quả dự đoán |

## Lệnh Chạy Nhanh

```bash
# Bước 1: Load dữ liệu
python 1_data.py

# Bước 2: Train model
python 2_train.py

# Bước 3: Test model
python 3_predict.py
```

---

## Nâng Cao (Tuỳ Chọn)

1. **Tăng dữ liệu (Data Augmentation)**:
   - Xoay ảnh, lật ảnh, zoom
   - Giúp model học tốt hơn

2. **Transfer Learning**:
   - Dùng model đã train sẵn (VGG16, ResNet)
   - Kết quả tốt hơn với dữ liệu ít

3. **Tăng epochs** nếu accuracy còn thấp

4. **Đổi IMG_SIZE** = 128 hoặc 224 để ảnh rõ hơn

---

## Troubleshooting

| Lỗi | Cách sửa |
|------|----------|
| `Module not found: tensorflow` | `pip install tensorflow` |
| `Module not found: sklearn` | `pip install scikit-learn` |
| `Module not found: PIL` | `pip install Pillow` |
| Accuracy = 50% (không học) | Tăng epochs, kiểm tra data |
| Out of memory | Giảm batch_size trong train |

---

## Yêu Cầu Cài Đặt

```bash
pip install tensorflow numpy pillow scikit-learn
```

**Lưu ý**: Nên dùng GPU để train nhanh hơn (TensorFlow sẽ tự detect).
