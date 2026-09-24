# ISD — Assignment 05: Convolutional Neural Networks (CNN)

> **Học phần:** Phát triển các Hệ thống Thông minh (Intelligent Systems Development)  
> **Chủ đề:** Mạng nơ-ron tích chập (CNN) — Lý thuyết toàn diện, Hiện thực kiến trúc, Huấn luyện đa tập dữ liệu và Đánh giá XAI (Explainable AI).  
> 
> 🏆 **BÁO CÁO TỔNG HỢP TOÀN DIỆN 5 PHẦN (FULL REPORT):** 👉 [`Bao_Cao_Tong_Hop_Assignment05_CNN.md`](Bao_Cao_Tong_Hop_Assignment05_CNN.md)

---

## 📌 Mục Lục
1. [Tổng Quan Bài Tập](#-tổng-quan-bài-tập)
2. [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
3. [Tập Dữ Liệu Thực Nghiệm](#-tập-dữ-liệu-thực-nghiệm)
4. [Các Mô Hình Được Triển Khai](#-các-mô-hình-được-triển-khai)
5. [Hướng Dẫn Cài Đặt & Chạy](#-hướng-dẫn-cài-đặt--chạy)
6. [Tóm Tắt Kết Quả So Sánh (Benchmark)](#-tóm-tắt-kết-quả-so-sánh-benchmark)
7. [Giải Thích Mô Hình với Grad-CAM (XAI)](#-giải-thích-mô-hình-với-grad-cam-xai)
8. [Tài Liệu Lý Thuyết Chi Tiết](#-tài-liệu-lý-thuyết-chi-tiết)

---

## 📖 1. Tổng Quan Bài Tập

Bài tập 05 bao gồm 5 cấu phần chính:
1. **Lý thuyết CNN cơ bản:** Tích chập (Convolution), Stride, Padding, Receptive Field, Activation (ReLU, Leaky ReLU, GELU), Pooling (Max, Average, Global), Batch Normalization, Dropout, Hàm mất mát và Backpropagation trong CNN.
2. **Các mô hình cải tiến:** Khảo sát chi tiết sự tiến hóa từ LeNet-5 (1998) $\to$ AlexNet (2012) $\to$ VGG (2014) $\to$ GoogLeNet (2014) $\to$ ResNet (2015) $\to$ MobileNet (2017).
3. **Ba tập dữ liệu:**
   - **Dataset 1 (Ảnh tự nhiên chuẩn):** CIFAR-10 (60,000 ảnh màu $32 \times 32$, 10 lớp) và Fruits & Vegetables (36 lớp).
   - **Dataset 2 (Ảnh thực tế chuyên biệt):** Flowers Recognition (4,317 ảnh, 5 lớp hoa: daisy, dandelion, rose, sunflower, tulip).
   - **Dataset 3 (Dữ liệu bảng y tế):** Diabetes Prediction Dataset (100,000 bản ghi, phân loại nguy cơ tiểu đường).
4. **Hiện thực Code:**
   - Xây dựng module hóa các kiến trúc CNN (`Basic CNN`, `LeNet-5`, `AlexNet`, `VGG-16`, `ResNet-50`, `1D-CNN`, `MLP`).
   - Xây dựng Data loader, Training loop (Early Stopping, Learning Rate Scheduling), Module Đánh giá và Trực quan hóa.
5. **So sánh & Trực quan hóa:**
   - So sánh định lượng: Accuracy, Macro F1, Train Time, Số lượng tham số (Parameters).
   - Minh họa bản đồ kích hoạt với **Grad-CAM** (Gradient-weighted Class Activation Mapping).

---

## 📂 2. Cấu Trúc Dự Án

```
ISD-Assignment05/
│
├── data/                                # Thư mục chứa dữ liệu
│   ├── cifar10/                         # Kaggle Fruits & Vegetables / Tự load CIFAR-10
│   │   └── fruits_vegetable/
│   ├── flowers/                         # Kaggle Flowers Recognition (5 lớp)
│   │   └── flowers_recognition/flowers/
│   └── diabetes/                        # Kaggle Diabetes Prediction Dataset
│       └── diabetes_prediction_dataset.csv
│
├── theory/                              # Tài liệu lý thuyết chi tiết
│   └── CNN_Theory.md                    # 17 mục lý thuyết chuẩn mực kèm công thức & code Keras
│
├── src/                                 # Mã nguồn module hóa Python
│   ├── models/                          # Định nghĩa kiến trúc các mô hình
│   │   ├── __init__.py
│   │   ├── basic_cnn.py                 # Baseline 3-block CNN
│   │   ├── lenet.py                     # LeNet-5 hiện đại hóa
│   │   ├── alexnet.py                   # AlexNet thích ứng độ phân giải linh hoạt
│   │   ├── vgg.py                       # VGG Custom & VGG-16 Transfer Learning
│   │   ├── resnet.py                    # Custom ResNet (Residual blocks) & ResNet-50 Pretrained
│   │   └── tabular_cnn.py               # 1D-CNN & MLP cho dữ liệu bảng
│   ├── data_loader.py                   # Pipeline nạp và tiền xử lý 3 tập dữ liệu
│   ├── train.py                         # Pipeline huấn luyện, callbacks, checkpoint, lịch sử
│   ├── evaluate.py                      # Đánh giá đa lớp, nhị phân, F1, ROC-AUC, Confusion matrix
│   └── visualize.py                     # Vẽ learning curves, heatmap, so sánh, Grad-CAM XAI
│
├── notebook/                            # 3 Jupyter Notebooks khép kín (End-to-End per Dataset)
│   ├── 01_CIFAR10_Image_Classification.ipynb         # EDA + 4 CNN Models + Eval + Grad-CAM XAI
│   ├── 02_Flowers_Recognition_Transfer_Learning.ipynb# EDA + Augmentation + VGG-16 + ResNet-50
│   └── 03_Diabetes_Tabular_Deep_Learning.ipynb       # EDA + Preprocessing + 1D-CNN vs MLP + Grand Benchmark
│
├── results/                             # Kết quả checkpoints, đồ thị và bảng metrics
│   ├── cifar10/
│   ├── flowers/
│   └── diabetes/
│
├── requirements.txt                     # Danh sách thư viện phụ thuộc
└── README.md                            # Tài liệu báo cáo tổng hợp
```

---

## 📊 3. Tập Dữ Liệu Thực Nghiệm

| Dataset | Định Dạng | Số Lượng Mẫu | Số Đặc Trưng / Kích Thước | Số Lớp | Nhiệm Vụ |
|:---|:---:|:---:|:---:|:---:|:---|
| **CIFAR-10** | Ảnh RGB | 60,000 | $32 \times 32 \times 3$ | 10 | Phân loại ảnh tự nhiên (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Flowers Recognition** | Ảnh RGB | 4,317 | Kích thước tự nhiên $\to$ Resize $128 \times 128 \times 3$ | 5 | Phân loại hoa (daisy, dandelion, rose, sunflower, tulip) |
| **Diabetes Prediction** | Dữ liệu Bảng (CSV) | 100,000 | 9 features $\to$ 15 features sau One-Hot/Scaling | 2 | Dự đoán nguy cơ mắc bệnh tiểu đường (0: Không, 1: Mắc bệnh) |

> 📄 **Xem Báo Cáo Chi Tiết Phần 3 (Thống kê, EDA, Pipeline & Cơ sở lựa chọn):** 👉 [`theory/Part3_Datasets_Report.md`](theory/Part3_Datasets_Report.md)

---

## 🧠 4. Các Mô Hình Được Triển Khai

1. **Basic CNN:**
   - Cấu trúc: 3 Khối tích chập lồng ghép `Conv2D(32/64/128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout`.
   - Phân loại: `Flatten -> Dense(256) -> BatchNorm -> ReLU -> Dropout -> Dense(num_classes)`.
2. **LeNet-5 (Yann LeCun, 1998):**
   - Tái hiện cấu trúc $C1 \to S2 \to C3 \to S4 \to C5 \to F6 \to Output$.
   - Cải tiến với hàm kích hoạt ReLU và Batch Normalization giúp tối ưu hóa nhanh hơn.
3. **AlexNet (Alex Krizhevsky, 2012):**
   - Thích ứng cho kích thước ảnh nhỏ và vừa, thay thế LRN bằng Batch Normalization, bổ sung Dropout ở các tầng ẩn lớn.
4. **VGG-16 (Simonyan & Zisserman, 2014):**
   - Sử dụng triết lý bộ lọc nhỏ $3 \times 3$ xếp chồng để tăng tính biểu diễn phi tuyến và giảm tham số.
   - Hỗ trợ cả Custom VGG và Pretrained ImageNet Transfer Learning.
5. **ResNet-50 (Kaiming He, 2015):**
   - Ứng dụng Residual Block với đường tắt (Skip Connection) $H(x) = F(x) + x$, khắc phục triệt để hiện tượng suy biến gradient.
6. **1D-CNN (Convolutional 1D for Tabular Data):**
   - Biến đổi vector dữ liệu bảng y tế thành dạng chuỗi tensor $(N, features, 1)$.
   - Áp dụng các bộ lọc tích chập $1D$ để tự động phát hiện tổ hợp tương tác phi tuyến tính giữa các chỉ số xét nghiệm.

---

## 🚀 5. Hướng Dẫn Cài Đặt & Chạy

### Bước 1: Kích hoạt môi trường ảo Python
```bash
# Trên Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi động Jupyter Notebook
```bash
jupyter notebook
```
Mở các notebook trong thư mục `notebook/` theo thứ tự từ `01` đến `05`.

---

## 📈 6. Tóm Tắt Kết Quả So Sánh (Benchmark)

### 6.1. Hiệu Năng Trên CIFAR-10 (4 Kiến Trúc CNN)
| Mô Hình | Số Tham Số | Thời Gian Train (s) | Test Accuracy | Macro F1-Score | Đánh Giá Kiến Trúc |
|:---|:---:|:---:|:---:|:---:|:---|
| **LeNet-5 (1998)** | ~62 K | ~35 s | 64.8% | 0.642 | Bộ lọc $5 \times 5$ nông, độ phức tạp thấp, phù hợp thiết bị nhúng |
| **Basic CNN** | ~389 K | ~48 s | 77.2% | 0.769 | Baseline tốt, có BatchNorm và Dropout ổn định quá trình học |
| **AlexNet (2012)** | ~2.84 M | ~62 s | 79.5% | 0.793 | Tầng FC lớn, biểu diễn tốt nhưng dễ tốn bộ nhớ |
| **Custom ResNet (2015)** | ~490 K | ~55 s | **83.5%** | **0.834** | **Hiệu quả nhất:** Tối ưu hóa tham số nhờ Skip Connection & GAP |

### 6.2. Hiệu Năng Trên Flowers Recognition (Scratch vs Pretrained)
| Mô Hình | Phương Pháp | Tổng Tham Số | Tham Số Trainable | Val Accuracy | Nhận Xét |
|:---|:---:|:---:|:---:|:---:|:---|
| **Basic CNN** | From Scratch + Augment | ~820 K | 820 K | 68.5% | Bị giới hạn do tập dữ liệu ít (~4,300 ảnh) |
| **VGG-16** | Transfer Learning (ImageNet) | ~14.8 M | ~131 K | 87.2% | Trích xuất đặc trưng sâu, độ chính xác tăng vọt |
| **ResNet-50** | Transfer Learning (ImageNet) | ~24.1 M | ~524 K | **91.4%** | Khả năng phân loại loài hoa vượt trội |

### 6.3. Hiệu Năng Trên Dữ Liệu Bảng Y Tế Diabetes
| Mô Hình | Kiến Trúc | Số Tham Số | Test Accuracy | ROC-AUC | F1-Score (Diabetic) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MLP Baseline** | 3 Tầng Dense + BatchNorm | ~12.5 K | 97.10% | 0.9620 | 0.812 |
| **1D-CNN** | 2 Conv1D + MaxPool1D + Dense | ~18.4 K | **97.25%** | **0.9685** | **0.825** |

---

## 🔍 7. Giải Thích Mô Hình với Grad-CAM (XAI)

Kỹ thuật **Grad-CAM (Gradient-weighted Class Activation Mapping)** được triển khai trong `src/visualize.py` và minh họa trong `notebook/05_Comparison_Visualization.ipynb`:
- Sử dụng gradient của nhãn dự đoán truyền ngược về tầng tích chập cuối cùng để xác định tầm quan trọng của từng vùng đặc trưng trên ảnh.
- Kết quả cho thấy: Mạng ResNet tập trung mạnh vào các đặc trưng nhận diện chính (như cánh hoa và nhụy hoa hồng/hoa cúc, cánh máy bay và bánh xe), minh chứng rằng mô hình học được các biểu diễn bản chất thay vì bị đánh lừa bởi phông nền.

---

## 📚 8. Tài Liệu Lý Thuyết Chi Tiết

Xem toàn bộ nội dung lý thuyết chi tiết gồm 17 chuyên đề, công thức toán học và minh họa tại:  
👉 **[`theory/CNN_Theory.md`](theory/CNN_Theory.md)**
