# Báo Cáo Phần 3: Khảo Sát, Phân Tích & Tiền Xử Lý Ba Tập Dữ Liệu Thực Nghiệm
## Convolutional Neural Networks (CNN) — Experimental Datasets Analysis

> **Môn học:** Phát triển các Hệ thống Thông minh (Intelligent Systems Development)  
> **Học phần:** Assignment 05 — Mạng Nơ-ron Tích chập (CNN)  
> **Cấu phần:** Phần 3 — Khảo sát chi tiết 3 Tập Dữ liệu (CIFAR-10, Flowers Recognition, Diabetes Prediction)  

---

# MỤC LỤC BÁO CÁO PHẦN 3

1. [Tổng Quan & Động Lực Lựa Chọn Ba Tập Dữ Liệu](#1-tổng-quan--động-lực-lựa-chọn-ba-tập-dữ-liệu)
   - 1.1. Mục tiêu và tiêu chí lựa chọn
   - 1.2. Bảng tổng hợp đối sánh 3 tập dữ liệu
2. [Tập Dữ Liệu 1: CIFAR-10 (Ảnh Chuẩn Benchmark Thị Giác Máy Tính)](#2-tập-dữ-liệu-1-cifar-10-ảnh-chuẩn-benchmark-thị-giác-máy-tính)
   - 2.1. Nguồn gốc & Cấu trúc tập dữ liệu
   - 2.2. Phân bố nhãn lớp & Phân chia Train/Val/Test
   - 2.3. Đặc điểm thị giác & Thách thức mô hình hóa
   - 2.4. Quy trình tiền xử lý & Data Loader
3. [Tập Dữ Liệu 2: Flowers Recognition (Ảnh Tự Nhiên Chuyên Biệt)](#3-tập-dữ-liệu-2-flowers-recognition-ảnh-tự-nhiên-chuyên-biệt)
   - 3.1. Nguồn gốc & Ý nghĩa bài toán
   - 3.2. Thống kê phân bố 5 loài hoa
   - 3.3. Thách thức thị giác thực tế & Động lực Transfer Learning
   - 3.4. Chiến lược Tăng cường Dữ liệu (Data Augmentation)
   - 3.5. Pipeline xử lý và cấu hình TensorFlow Dataset
4. [Tập Dữ Liệu 3: Diabetes Prediction Dataset (Dữ Liệu Bảng Y Tế)](#4-tập-dữ-liệu-3-diabetes-prediction-dataset-dữ-liệu-bảng-y-tế)
   - 4.1. Bối cảnh bài toán y tế & Nguồn dữ liệu
   - 4.2. Từ điển dữ liệu (Data Dictionary) & Thống kê mô tả
   - 4.3. Phân tích bài toán mất cân bằng lớp (Class Imbalance)
   - 4.4. Phân tích tương quan lâm sàng (HbA1c & Blood Glucose)
   - 4.5. Pipeline Tiền xử lý: Scaler, One-Hot Encoding & Tránh rò rỉ dữ liệu
   - 4.6. Biến đổi Không gian Tensor 1D cho Mạng 1D-CNN
5. [So Sánh Tổng Thể & Bài Học Thực Nghiệm Trên 3 Dữ Liệu](#5-so-sánh-tổng-thể--bài-học-thực-nghiệm-trên-3-dữ-liệu)
   - 5.1. Ma trận so sánh đa chiều
   - 5.2. Sự tương thích giữa Kiến trúc Mô hình và Đặc trưng Dữ liệu
6. [Hướng Dẫn Trích Xuất & Trực Quan Hóa (EDA Figures) Vào Báo Cáo](#6-hướng-dẫn-trích-xuất--trực-quan-hóa-eda-figures-vào-báo-cáo)

---

## 1. Tổng Quan & Động Lực Lựa Chọn Ba Tập Dữ Liệu

### 1.1. Mục tiêu và tiêu chí lựa chọn

Trong khuôn khổ môn học **Phát triển các Hệ thống Thông minh**, việc đánh giá một kiến trúc học sâu không thể chỉ dừng lại ở một bài toán đồ chơi (toy dataset) hoặc một kiểu dữ liệu duy nhất. Để kiểm chứng toàn diện năng lực biểu diễn, khả năng kháng nhiễu, hiện tượng quá khớp (overfitting) và tính tổng quát hóa của **Mạng Nơ-ron Tích chập (CNN)**, đề tài đã lựa chọn **3 tập dữ liệu** với những sắc thái kỹ thuật và miền ứng dụng hoàn toàn khác biệt:

1. **Tính đa dạng về không gian dữ liệu:** Kết hợp hài hòa giữa dữ liệu có cấu trúc lưới 2D liên tục (Ảnh màu pixel) và dữ liệu bảng có cấu trúc bảng biểu 1D (Tabular / Clinical features).
2. **Tính phân cấp về quy mô và độ phân giải:** Từ ảnh độ phân giải nhỏ cố định ($32 \times 32$ pixels trên CIFAR-10) đến ảnh tự nhiên chụp máy ảnh có độ phân giải lớn, không đồng nhất trên Flowers Recognition.
3. **Mục tiêu thực nghiệm rõ ràng cho từng kiến trúc:**
   - **CIFAR-10:** Đóng vai trò làm mốc đối chuẩn (Standard Academic Benchmark) để so sánh 4 kiến trúc CNN: *Basic CNN*, *LeNet-5*, *AlexNet*, và *ResNet*.
   - **Flowers Recognition:** Tập dữ liệu nhỏ (~4,300 ảnh) nhưng đa dạng góc chụp và độ phức tạp cao, là môi trường lý tưởng để chứng minh sức mạnh của kỹ thuật **Chuyển giao tri thức (Transfer Learning)** với các mô hình nạp sẵn trọng số ImageNet (*VGG-16*, *ResNet-50*) so với việc huấn luyện từ đầu (*Training from Scratch*).
   - **Diabetes Prediction Dataset:** Tập dữ liệu y tế quy mô lớn (100,000 bản ghi), chứng minh tính phổ quát của cơ chế tích chập khi mở rộng sang dạng **1D-CNN (One-dimensional Convolutional Neural Networks)** để tự động học các tổ hợp tương tác phi tuyến giữa các chỉ số xét nghiệm lâm sàng.

### 1.2. Bảng tổng hợp đối sánh 3 tập dữ liệu

| Tiêu Chí So Sánh | Tập Dữ Liệu 1: CIFAR-10 | Tập Dữ Liệu 2: Flowers Recognition | Tập Dữ Liệu 3: Diabetes Prediction |
| :--- | :--- | :--- | :--- |
| **Miền bài toán (Domain)** | Thị giác máy tính (Computer Vision) | Thị giác máy tính (Computer Vision) | Dữ liệu bảng Y tế (Clinical Tabular Data) |
| **Nguồn gốc / Tác giả** | Alex Krizhevsky (Kaggle / Keras Datasets) | Alexander Mamaev (Kaggle) | Mustafa Turgut (Kaggle) |
| **Định dạng gốc** | Tensor mảng số nguyên `uint8` | Tệp tin ảnh nén (`.jpg`) | Bảng dữ liệu định dạng `.csv` |
| **Tổng số lượng mẫu** | 60,000 ảnh | 4,317 ảnh | 100,000 bản ghi (99,982 sau làm sạch) |
| **Kích thước mẫu gốc** | $32 \times 32 \times 3$ (Cố định) | Không đồng nhất (từ $300\times 200$ đến $1024\times 768$) | 9 thuộc tính (Đặc trưng số & chuỗi phân loại) |
| **Kích thước đầu vào mô hình** | $(32, 32, 3)$ | $(128, 128, 3)$ hoặc $(224, 224, 3)$ | $(15, 1)$ cho 1D-CNN và $(15,)$ cho MLP |
| **Số lượng nhãn lớp** | 10 lớp cân bằng hoàn hảo | 5 lớp (loài hoa) | 2 lớp nhị phân (0: Không bệnh, 1: Mắc bệnh) |
| **Tính cân bằng lớp** | Cân bằng tuyệt đối (6,000 mẫu/lớp) | Mất cân bằng nhẹ (733 – 1,052 ảnh/lớp) | Mất cân bằng nghiêm trọng (91.5% âm tính / 8.5% dương tính) |
| **Phương pháp tiền xử lý** | Min-Max Normalization $[0, 1]$, One-Hot | Resize, Rescale $[0, 1]$, Data Augmentation | StandardScaler, One-Hot Encoding, 1D Reshape |
| **Mô hình triển khai chính** | LeNet-5, Basic CNN, AlexNet, ResNet Custom | Basic CNN Scratch, Pretrained VGG-16, ResNet-50 | 1D-CNN, Multi-Layer Perceptron (MLP) |

---

## 2. Tập Dữ Liệu 1: CIFAR-10 (Ảnh Chuẩn Benchmark Thị Giác Máy Tính)

### 2.1. Nguồn gốc & Cấu trúc tập dữ liệu

**CIFAR-10 (Canadian Institute For Advanced Research)** là một trong những bộ dữ liệu kinh điển và uy tín nhất thế giới trong lĩnh vực học máy và thị giác máy tính, được biên soạn bởi GS. Alex Krizhevsky, Vinod Nair và Geoffrey Hinton (2009).

- **Quy mô tập mẫu:** 60,000 ảnh màu RGB.
- **Kích thước không gian:** Chiều cao $H = 32$, Chiều rộng $W = 32$, Số kênh màu $C = 3$.
- **Dải giá trị pixel:** Số nguyên không dấu từ $0$ đến $255$.
- **Cấu trúc lưu trữ:** Được tích hợp trực tiếp qua module `tensorflow.keras.datasets.cifar10` và thư mục `data/cifar10/`.

### 2.2. Phân bố nhãn lớp & Phân chia Train/Val/Test

CIFAR-10 có ưu điểm học thuật nổi bật là **tính cân bằng hoàn hảo** giữa các lớp. Toàn bộ 60,000 ảnh được chia đều cho 10 lớp thực thể độc lập:

$$\text{Số lượng mẫu mỗi lớp} = \frac{60,000}{10} = 6,000 \text{ ảnh}$$

**Danh sách 10 lớp thực thể:**
1. `airplane` (Máy bay)
2. `automobile` (Xe hơi con)
3. `bird` (Chim chóc)
4. `cat` (Mèo)
5. `deer` (Hươu / Nai)
6. `dog` (Chó)
7. `frog` (Ếch)
8. `horse` (Ngựa)
9. `ship` (Tàu thủy)
10. `truck` (Xe tải)

**Chiến lược phân chia tập dữ liệu (Dataset Splitting):**
Trong quy trình hiện thực tại `src/data_loader.py` và `notebook/01_CIFAR10_Image_Classification.ipynb`, dữ liệu được phân chia chặt chẽ như sau:

- **Tập Huấn luyện (Train set):** 45,000 ảnh (chiếm 75% toàn bộ dữ liệu, lấy 90% từ tập train gốc 50,000 ảnh).
- **Tập Kiểm định (Validation set):** 5,000 ảnh (chiếm 8.33%, dùng để theo dõi loss/accuracy và kích hoạt Early Stopping nhằm tránh rò rỉ dữ liệu).
- **Tập Kiểm tra (Test set):** 10,000 ảnh (chiếm 16.67%, tập dữ liệu hoàn toàn độc lập dùng cho đánh giá cuối cùng).

```
+-----------------------------------------------------------------------------------+
|                            TỔNG THỂ CIFAR-10: 60,000 ẢNH                          |
+---------------------------------------------------------+-------------------------+
|                  TRAIN + VAL: 50,000 ẢNH                 |     TEST: 10,000 ẢNH    |
+------------------------------------+--------------------+-------------------------+
|        TRAIN: 45,000 ẢNH (75%)     |  VAL: 5,000 (8.3%) |    TEST: 10,000 (16.7%) |
+------------------------------------+--------------------+-------------------------+
```

### 2.3. Đặc điểm thị giác & Thách thức mô hình hóa

1. **Độ phân giải siêu nhỏ ($32 \times 32$):** Các chi tiết cạnh viền và cấu trúc cục bộ bị mờ nhạt (low-frequency dominance). Ví dụ: Mắt của mèo hoặc cánh của máy bay chỉ chiếm diện tích từ $2 \times 2$ đến $3 \times 3$ pixel.
2. **Hiện tượng nhầm lẫn giữa các lớp tương đồng (Intra-class & Inter-class variance):**
   - Lớp `cat` và `dog` có các đặc trưng hình thể, màu lông rất giống nhau.
   - Lớp `automobile` và `truck` có cấu trúc khung kim loại, bánh xe và nền đường tương tự nhau.
3. **Phông nền phức tạp (Complex Background):** Động vật thường xuất hiện trên đồng cỏ, trong rừng rậm rạp; tàu thủy và máy bay xuất hiện trên nền trời/biển có màu sắc gần trùng với vật thể.
4. **Ý nghĩa thực nghiệm:** Do kích thước $32 \times 32$, các mô hình có kernel lớn như LeNet-5 ($5 \times 5$) hoặc AlexNet gốc ($11 \times 11$) sẽ làm suy giảm kích thước không gian rất nhanh (Feature map co lại quá sớm). Đây là minh chứng thực nghiệm sinh động cho việc tại sao VGG ($3 \times 3$) và ResNet (Skip connection) lại đạt độ chính xác vượt trội hơn hẳn.

### 2.4. Quy trình tiền xử lý & Data Loader

Đoạn pipeline tiền xử lý cho CIFAR-10 được thực hiện tối ưu trong `src/data_loader.py`:

1. **Chuẩn hóa giá trị Pixel (Min-Max Scaling):**
   Chuyển đổi kiểu dữ liệu từ số nguyên `uint8` sang số thực dấu phẩy động `float32`, đồng thời chia tỉ lệ cho $255.0$:

   $$X_{\text{norm}} = \frac{X_{\text{raw}}}{255.0} \in [0.0, 1.0]$$

   *Mục đích:* Giúp các hàm kích hoạt (ReLU, Sigmoid) và quá trình lan truyền ngược (Backpropagation) không bị bão hòa trọng số hoặc dao động gradient mạnh.

2. **Mã hóa Nhãn One-Hot (One-Hot Encoding):**
   Biến đổi nhãn số nguyên $y \in \{0, 1, ..., 9\}$ thành vector xác suất nhị phân 10 chiều:

   $$y = 3 \implies [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]^T$$

   *Mục đích:* Phục vụ hàm mất mát `CategoricalCrossentropy` kết hợp tầng xuất `Softmax`.

```python
# Trích xuất từ src/data_loader.py
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full.squeeze(),
    test_size=0.1, random_state=42, stratify=y_train_full.squeeze()
)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

---

## 3. Tập Dữ Liệu 2: Flowers Recognition (Ảnh Tự Nhiên Chuyên Biệt)

### 3.1. Nguồn gốc & Ý nghĩa bài toán

Bộ dữ liệu **Flowers Recognition** do tác giả Alexander Mamaev tổng hợp và công bố trên nền tảng Kaggle. Tập dữ liệu tập trung vào bài toán nhận dạng 5 loài hoa phổ biến trong tự nhiên. Khác với CIFAR-10 mang tính hàn lâm, Flowers Recognition đại diện cho một bài toán thị giác máy tính công nghiệp thực tế: tập dữ liệu chuyên biệt theo ngành, số lượng mẫu hữu hạn nhưng chất lượng ảnh chụp cao.

### 3.2. Thống kê phân bố 5 loài hoa

Tổng số lượng ảnh sau khi duyệt thực tế trong thư mục `data/flowers/flowers_recognition/flowers/` là **4,317 ảnh**, được phân bổ vào 5 thư mục con tương ứng với 5 loài hoa:

| Tên Loài Hoa | Tên Tiếng Việt | Số Lượng Ảnh | Tỷ Lệ Phần Trăm (%) | Đặc Điểm Hình Thái Nhận Diện |
| :--- | :--- | :---: | :---: | :--- |
| **daisy** | Hoa Cúc Họa Mi | 764 | 17.70% | Cánh hoa trắng nhỏ tỏa tròn, nhụy hoa tròn màu vàng nổi bật |
| **dandelion** | Hoa Bồ Công Anh | 1,052 | 24.37% | Cánh hoa màu vàng rực rỡ, sợi nhuyễn, khi tàn tạo cầu bông trắng |
| **rose** | Hoa Hồng | 784 | 18.16% | Cánh hoa xếp lớp cuộn xoắn đồng tâm, đa dạng màu (đỏ, hồng, trắng, vàng) |
| **sunflower** | Hoa Hướng Dương | 733 | 16.98% | Kích thước nhụy hoa nâu đen cực lớn ở tâm, bao quanh bởi cánh vàng lớn |
| **tulip** | Hoa Uất Kim Hương | 984 | 22.79% | Cánh hoa hình chuông đứng hoặc chén úp, búp hoa mượt mà vươn cao |
| **TỔNG CỘNG** | — | **4,317** | **100.0%** | — |

```
Phân bố số lượng mẫu 5 loài hoa:
[dandelion]  █████████████████████████ 1,052 (24.4%)
[tulip]      ███████████████████████ 984 (22.8%)
[rose]       ██████████████████ 784 (18.2%)
[daisy]      ██████████████████ 764 (17.7%)
[sunflower]  █████████████████ 733 (17.0%)
```

*Nhận xét về phân phối:* Mức độ mất cân bằng giữa lớp nhiều nhất (`dandelion`: 1,052) và lớp ít nhất (`sunflower`: 733) chỉ ở mức nhẹ (~1.43 lần). Sự chênh lệch này hoàn toàn nằm trong ngưỡng kiểm soát tự nhiên của mạng nơ-ron mà không bắt buộc phải áp dụng kỹ thuật Resampling phức tạp.

### 3.3. Thách thức thị giác thực tế & Động lực Transfer Learning

Khi phân tích sâu về đặc tính hình ảnh của tập Flowers Recognition trong notebook `02_Flowers_Recognition_Transfer_Learning.ipynb`, chúng ta phát hiện 3 rào cản kỹ thuật lớn:

1. **Kích thước ảnh không đồng nhất:** Không giống ảnh $32 \times 32$ cố định, ảnh gốc có độ phân giải từ thấp đến cao (tỷ lệ khung hình $4:3$, $16:9$, $1:1$ biến đổi).
2. **Nhiễu bối cảnh tự nhiên cao (High Background Clutter):** Hoa thường được chụp ngoài tự nhiên, lẫn với thân cây, cỏ dại, đất cát, hàng rào, hoặc bị côn trùng (ong, bướm) che khuất một phần.
3. **Biến thiên nội lớp rất lớn (High Intra-class Variance):** Cùng là loài hoa hồng (`rose`), nhưng có thể là hoa màu đỏ thẫm, hoa màu vàng, hoa màu trắng, bông đang nở rộ hoặc mới chỉ là nụ búp.
4. **Động lực áp dụng Transfer Learning:** Với chỉ ~4,300 ảnh cho 5 lớp phức tạp, nếu huấn luyện một mạng CNN sâu từ đầu (From Scratch), mô hình sẽ nhanh chóng ghi nhớ dữ liệu huấn luyện (Overfitting), dẫn đến khoảng cách lớn giữa Training Loss và Validation Loss. Do đó, việc tận dụng bộ trích xuất đặc trưng (Feature Extractor) đã được tiền huấn luyện trên hàng triệu ảnh của tập dữ liệu **ImageNet** thông qua mạng **VGG-16** và **ResNet-50** là giải pháp tối ưu mang tính quyết định.

### 3.4. Chiến lược Tăng cường Dữ liệu (Data Augmentation)

Nhằm nhân tạo hóa sự đa dạng của tập huấn luyện và triệt tiêu tính bất biến theo phép biến đổi affine, đồ án xây dựng một lớp `Sequential` chứa các phép tăng cường ngẫu nhiên ngay trong pipeline của TensorFlow:

```python
# Tăng cường dữ liệu thời gian thực (On-the-fly Data Augmentation)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),       # Lật ngang ngẫu nhiên
    layers.RandomRotation(0.15),           # Xoay góc ngẫu nhiên trong khoảng [-15%, +15%]
    layers.RandomZoom(0.15),               # Phóng to / thu nhỏ ngẫu nhiên +/- 15%
    layers.RandomContrast(0.10),           # Điều chỉnh độ tương phản ngẫu nhiên +/- 10%
], name="data_augmentation")
```

*Cơ sở toán học:* Các phép biến đổi này bảo toàn tính chất ngữ nghĩa (Invariant Semantics) của loài hoa (một bông hoa hồng bị lật ngược hoặc xoay góc vẫn giữ nguyên bản chất là hoa hồng), nhưng cung cấp cho mạng nơ-ron những biểu diễn ma trận pixel hoàn toàn mới mẻ trong từng epoch huấn luyện.

### 3.5. Pipeline xử lý và cấu hình TensorFlow Dataset

Hệ thống nạp dữ liệu được cấu trúc trong `src/data_loader.py` bằng hàm `load_flowers()` sử dụng `tf.keras.utils.image_dataset_from_directory`:

- **Kích thước định hình lại (Resizing):** Đưa toàn bộ ảnh về kích thước cố định $128 \times 128 \times 3$ (hoặc $224 \times 224 \times 3$ khi nạp vào backbone Pretrained).
- **Phân chia tập mẫu:** 80% cho Training (3,454 ảnh) và 20% cho Validation (863 ảnh) với hạt giống ngẫu nhiên `seed=42`.
- **Tối ưu hóa bộ nhớ đệm (Performance Optimization):** Áp dụng kỹ thuật `tf.data.AUTOTUNE` kết hợp với `prefetch()` và `map(Rescaling(1./255))` giúp việc nạp dữ liệu từ ổ đĩa và tính toán trên GPU diễn ra song song, loại bỏ hiện tượng nghẽn cổ chai I/O.

---

## 4. Tập Dữ Liệu 3: Diabetes Prediction Dataset (Dữ Liệu Bảng Y Tế)

### 4.1. Bối cảnh bài toán y tế & Nguồn dữ liệu

**Diabetes Prediction Dataset** được thu thập từ các hồ sơ bệnh án điện tử và công bố trên Kaggle bởi Mustafa Turgut. Bài toán đặt ra là: **Dự đoán sớm nguy cơ một bệnh nhân có mắc bệnh đái tháo đường (Diabetes) hay không dựa trên các chỉ số nhân khẩu học và kết quả xét nghiệm sinh hóa máu.**

- **Tổng số bản ghi ban đầu:** 100,000 dòng dữ liệu.
- **Tiền xử lý sơ bộ:** Trong tập gốc tồn tại một số ít bản ghi có thuộc tính `gender == "Other"` (chiếm khoảng 18 dòng, không đủ kích thước thống kê), do đó hệ thống lọc bỏ để đảm bảo tính chuẩn hóa nhị phân giới tính, giữ lại **99,982 bản ghi sạch**.
- **Tính thời sự và giá trị y sinh:** Bộ dữ liệu chứa hai chỉ số vàng trong chẩn đoán đái tháo đường của Hiệp hội Đái tháo đường Hoa Kỳ (ADA): nồng độ **HbA1c** và **Mức đường huyết lúc đói (Blood Glucose Level)**.

### 4.2. Từ điển dữ liệu (Data Dictionary) & Thống kê mô tả

Bảng dưới đây chi tiết hóa 9 trường thông tin lâm sàng của bộ dữ liệu:

| Tên Thuộc Tính | Kiểu Dữ Liệu | Miền Giá Trị / Đơn Vị | Ý Nghĩa Y Sinh Học & Lâm Sàng |
| :--- | :---: | :---: | :--- |
| **gender** | Categorical | Female, Male | Giới tính sinh học của bệnh nhân |
| **age** | Numerical (Float) | $0.08 \to 80.0$ tuổi | Tuổi của bệnh nhân (nguy cơ tiểu đường tăng theo tuổi) |
| **hypertension** | Binary (0/1) | 0: Không, 1: Có | Tiền sử bệnh lý tăng huyết áp |
| **heart_disease** | Binary (0/1) | 0: Không, 1: Có | Tiền sử bệnh lý tim mạch mãn tính |
| **smoking_history**| Categorical | never, current, former, ever, not current, No Info | Tiền sử và thói quen hút thuốc lá |
| **bmi** | Numerical (Float) | $10.01 \to 95.69 \text{ kg/m}^2$ | Chỉ số khối cơ thể (Body Mass Index) |
| **HbA1c_level** | Numerical (Float) | $3.5\% \to 9.0\%$ | Tỷ lệ huyết sắc tố bị glycate hóa trong hồng cầu |
| **blood_glucose** | Numerical (Int) | $80 \to 300 \text{ mg/dL}$ | Nồng độ glucose trong máu |
| **diabetes** (Target)| Binary (0/1) | 0: Bình thường, 1: Mắc bệnh | **Biến mục tiêu (Ground Truth)** |

**Bảng thống kê mô tả các biến số liên tục (Trích xuất từ `raw_df.describe()`):**

| Thuộc Tính | Trung Bình (Mean) | Độ Lệch Chuẩn (Std) | Tối Thiểu (Min) | Phân Vị 25% | Trung Vị (50%) | Phân Vị 75% | Tối Đa (Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **age** | 41.88 | 22.52 | 0.08 | 24.00 | 43.00 | 60.00 | 80.00 |
| **bmi** | 27.32 | 6.64 | 10.01 | 23.63 | 27.32 | 29.58 | 95.69 |
| **HbA1c_level** | 5.53 | 1.07 | 3.50 | 4.80 | 5.80 | 6.20 | 9.00 |
| **blood_glucose**| 138.06 | 40.71 | 80.00 | 100.00 | 140.00 | 159.00 | 300.00 |

### 4.3. Phân tích bài toán mất cân bằng lớp (Class Imbalance)

Trong 99,982 bệnh nhân:
- **Lớp 0 (Không mắc bệnh):** ~91,482 người, chiếm **91.50%**
- **Lớp 1 (Mắc bệnh tiểu đường):** ~8,500 người, chiếm **8.50%**

```
Biểu đồ phân phối nhãn mục tiêu (Class Distribution):
[Lớp 0 - Khỏe mạnh]   ████████████████████████████████████ 91.5% (91,482 mẫu)
[Lớp 1 - Tiểu đường]  ████ 8.5% (8,500 mẫu)
```

**Hệ quả toán học & Thách thức đánh giá (The Accuracy Paradox):**
Nếu một mô hình phân loại ngây thơ (Naive Baseline) luôn luôn dự đoán mọi bệnh nhân đều "Không mắc bệnh" ($y = 0$), thì mô hình đó vẫn đạt được độ chính xác toàn cục (Accuracy) lên tới **91.5%**. Tuy nhiên, mô hình đó hoàn toàn vô giá trị trong thực tiễn y tế vì đã bỏ sót 100% bệnh nhân thực sự mắc bệnh (False Negative = 8,500).

*Giải pháp kỹ thuật của đồ án:*
1. **Phân chia dữ liệu phân tầng (Stratified Splitting):** Đảm bảo tỷ lệ 8.5% bệnh nhân tiểu đường xuất hiện đồng đều trong cả 3 tập Train, Validation và Test.
2. **Không đánh giá đơn thuần bằng Accuracy:** Bắt buộc sử dụng các độ đo nhạy cảm với lớp thiểu số: **ROC-AUC**, **Precision-Recall AUC (PR-AUC)**, **F1-Score riêng cho lớp 1**, và ma trận nhầm lẫn (**Confusion Matrix**).

### 4.4. Phân tích tương quan lâm sàng (HbA1c & Blood Glucose)

Khi chạy ma trận tương quan Pearson trong notebook `03_Diabetes_Tabular_Deep_Learning.ipynb`, kết quả phản ánh sự tương đồng tuyệt đối với y học thực tế:

$$\rho(\text{HbA1c\_level}, \text{diabetes}) \approx 0.40 \quad (\text{Mức tương quan cao nhất})$$
$$\rho(\text{blood\_glucose\_level}, \text{diabetes}) \approx 0.42 \quad (\text{Mức tương quan thứ hai})$$
$$\rho(\text{age}, \text{diabetes}) \approx 0.26 \quad (\text{Nguy cơ gia tăng theo tuổi})$$

- **Ý nghĩa lâm sàng:** Theo tiêu chuẩn ADA, người bình thường có $\text{HbA1c} < 5.7\%$. Giai đoạn tiền đái tháo đường là $5.7\% \to 6.4\%$, và nếu $\text{HbA1c} \geq 6.5\%$ thì được chẩn đoán mắc đái tháo đường. Biểu đồ phân tán (Scatter/Boxplot) giữa HbA1c và Blood Glucose trong đồ án phân tách rất rõ cụm bệnh nhân dương tính tập trung dày đặc ở vùng $\text{HbA1c} \geq 6.5$ và $\text{blood\_glucose} \geq 180$.

### 4.5. Pipeline Tiền xử lý: Scaler, One-Hot Encoding & Tránh rò rỉ dữ liệu

Quá trình tiền xử lý được đóng gói trong `ColumnTransformer` của `scikit-learn` và tích hợp tại `src/data_loader.py`:

1. **Phân chia tập dữ liệu trước khi Fit (Preventing Data Leakage):**
   - Tập Train (70%): Dùng để fit các tham số thống kê ($\mu, \sigma$, categories).
   - Tập Validation (10%): Dùng để tinh chỉnh siêu tham số và Early Stopping.
   - Tập Test (20%): Đóng vai trò dữ liệu độc lập mù (unseen blind test).
2. **Xử lý biến số liên tục (Numerical Features):**
   Áp dụng `StandardScaler` để đưa các thuộc tính về phân phối chuẩn có trung bình bằng 0 và phương sai bằng 1:

   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

3. **Xử lý biến phân loại (Categorical Features):**
   Áp dụng `OneHotEncoder(drop="first")` cho `gender` và `smoking_history`. Tham số `drop="first"` loại bỏ một cột nhị phân ngẫu nhiên cho mỗi thuộc tính, triệt tiêu hiện tượng cộng tuyến hoàn hảo (Multicollinearity / Dummy Variable Trap).
4. **Không gian đặc trưng sau chuyển đổi:**
   Từ 8 thuộc tính đầu vào ban đầu, sau khi mã hóa One-Hot, không gian đặc trưng mở rộng thành **15 chiều**:
   - 6 biến số: `age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`, `blood_glucose_level`.
   - 1 biến giới tính: `gender_Male`.
   - 5 biến tiền sử hút thuốc: `smoking_current`, `smoking_ever`, `smoking_former`, `smoking_never`, `smoking_not current`.

### 4.6. Biến đổi Không gian Tensor 1D cho Mạng 1D-CNN

Điểm sáng tạo cốt lõi trong phần xử lý dữ liệu của đề tài là việc ứng dụng mạng **1D-CNN** trên dữ liệu bảng:

- **Dữ liệu cho MLP thông thường:** Là ma trận 2D có kích thước `(batch_size, 15)`.
- **Dữ liệu cho 1D-CNN:** Được mở rộng số chiều (dimension expansion) thành tensor 3D:

  $$\mathbf{X}_{\text{1D}} \in \mathbb{R}^{N \times 15 \times 1}$$

Trong đó:
- $N$: Kích thước lô huấn luyện (Batch size).
- $15$: Chiều dài của chuỗi đặc trưng (Sequence length / Spatial steps).
- $1$: Số kênh thông tin (Channel dimension = 1).

*Nguyên lý hoạt động của 1D-CNN trên dữ liệu y tế:* Bộ lọc tích chập 1 chiều $\text{Conv1D}(filters=32, kernel\_size=3)$ sẽ trượt dọc qua các đặc trưng lâm sàng lân cận. Cơ chế này đóng vai trò như một **cỗ máy tự động trích xuất tương tác đặc trưng cục bộ (Local Feature Interaction)**. Ví dụ, khi bộ lọc kích thước $3$ quét qua bộ ba đặc trưng `[bmi, HbA1c_level, blood_glucose_level]`, nó sẽ tự động kết hợp các trọng số nhân để tạo ra các đặc trưng phi tuyến liên kết giữa cân nặng và đường huyết mà không đòi hỏi chuyên gia y tế phải tạo thủ công các biến tương tác.

---

## 5. So Sánh Tổng Thể & Bài Học Thực Nghiệm Trên 3 Dữ Liệu

### 5.1. Ma trận so sánh đa chiều

| Thuộc Tính Kỹ Thuật | CIFAR-10 | Flowers Recognition | Diabetes Prediction |
| :--- | :--- | :--- | :--- |
| **Bản chất biểu diễn** | Thị giác tĩnh $32 \times 32$ | Thị giác tự nhiên đa dạng | Bảng hồ sơ bệnh án điện tử |
| **Đặc thù phân bố lớp** | Cân bằng 100% | Cân bằng tương đối | Mất cân bằng nặng (8.5% dương tính) |
| **Thách thức kỹ thuật lớn nhất**| Độ phân giải nhỏ, mất chi tiết | Quá ít mẫu ảnh, dễ Overfitting | Độ chênh lệch nhãn lớp lớn |
| **Biện pháp xử lý chủ đạo** | Min-Max $[0, 1]$, One-Hot | Data Augmentation + Transfer Learning | StandardScaler + One-Hot + 1D Reshape |
| **Mô hình CNN tối ưu nhất** | ResNet Custom (83.5% Accuracy) | ResNet-50 Pretrained (91.4% Accuracy) | 1D-CNN (ROC-AUC: 0.9685, F1: 0.825) |
| **Bài học rút ra** | Chiều sâu & Skip connection giúp học tốt hơn filter lớn | Không nên train scratch tập ảnh nhỏ; Transfer Learning là chìa khóa | CNN hoàn toàn có thể trích xuất tương tác đặc trưng dữ liệu bảng hiệu quả hơn MLP |

### 5.2. Sự tương thích giữa Kiến trúc Mô hình và Đặc trưng Dữ liệu

1. **Từ LeNet đến ResNet trên CIFAR-10:**
   LeNet-5 sử dụng kernel $5 \times 5$ không có padding (Valid padding), khiến tensor ảnh $32 \times 32$ bị co lại thành $28 \times 28 \to 14 \times 14 \to 10 \times 10 \to 5 \times 5$ quá nhanh, làm mất thông tin không gian. Ngược lại, ResNet sử dụng kernel nhỏ $3 \times 3$ kèm padding và đường tắt Skip Connection $H(x) = F(x) + x$, giúp bảo toàn tín hiệu gradient và trích xuất đặc trưng sâu sắc hơn rất nhiều.

2. **Sự vượt trội của Pretrained Features trên Flowers:**
   Khi huấn luyện Basic CNN từ đầu trên Flowers, mô hình chỉ đạt độ chính xác ~68.5% và có dấu hiệu phân kỳ overfitting sau epoch thứ 15. Tuy nhiên, khi sử dụng VGG-16 và ResNet-50 với trọng số nạp sẵn từ ImageNet (đã học các bộ lọc phát hiện cạnh, góc, vân hoa, màu sắc từ 1.4 triệu ảnh), độ chính xác nhảy vọt lên **87.2% (VGG-16)** và **91.4% (ResNet-50)**.

3. **Tính ưu việt của 1D-CNN so với MLP trên Diabetes:**
   Mô hình MLP kết nối đầy đủ toàn bộ 15 đặc trưng với tầng ẩn, dễ dẫn đến số lượng trọng số thừa và không tận dụng được tính phụ thuộc cục bộ giữa các nhóm chỉ số. Mạng 1D-CNN áp dụng cơ chế chia sẻ trọng số (Weight Sharing) và vùng tiếp nhận cục bộ (Receptive Field) trên trục đặc trưng, mang lại hệ số ROC-AUC đạt **0.9685** và F1-Score đạt **0.825** trên nhóm bệnh nhân tiểu đường, nhỉnh hơn rõ rệt so với MLP truyền thống.

---

## 6. Hướng Dẫn Trích Xuất & Trực Quan Hóa (EDA Figures) Vào Báo Cáo

Để hoàn thiện báo cáo bản in (Word/PDF) đạt điểm tối đa, sinh viên trích xuất trực tiếp các biểu đồ đã được hiển thị trong 3 file Jupyter Notebook vào bài viết theo danh mục sau:

### 6.1. Hình ảnh cần trích từ `01_CIFAR10_Image_Classification.ipynb`:
1. **Lưới mẫu ảnh 10 lớp CIFAR-10:** Trích xuất lưới ảnh $2 \times 5$ hoặc $3 \times 4$ hiển thị trực quan các lớp xe, động vật kèm nhãn tương ứng.
2. **Biểu đồ cột phân phối 10 lớp:** Biểu đồ Bar chart chứng minh mỗi lớp có chính xác 6,000 mẫu (thể hiện tính cân bằng tuyệt đối).

### 6.2. Hình ảnh cần trích từ `02_Flowers_Recognition_Transfer_Learning.ipynb`:
1. **Biểu đồ cột phân bố 5 loài hoa:** Bar chart thể hiện số lượng ảnh của từng loài hoa (`dandelion`, `tulip`, `rose`, `daisy`, `sunflower`).
2. **Lưới mẫu ảnh hoa thế giới thực:** Hình ảnh minh họa 5 bông hoa với độ phân giải và góc chụp tự nhiên.
3. **Minh họa Data Augmentation:** Lưới ảnh so sánh một bông hoa gốc sau khi qua các phép biến đổi lật ngang, xoay góc, phóng to và thay đổi tương phản.

### 6.3. Hình ảnh cần trích từ `03_Diabetes_Tabular_Deep_Learning.ipynb`:
1. **Biểu đồ tỷ lệ mất cân bằng lớp:** Donut chart hoặc Bar chart thể hiện 91.5% nhãn 0 và 8.5% nhãn 1.
2. **Ma trận tương quan nhiệt (Pearson Correlation Heatmap):** Thể hiện hệ số tương quan giữa 9 biến, làm nổi bật hai ô `HbA1c_level` ($0.40$) và `blood_glucose_level` ($0.42$) với cột `diabetes`.
3. **Biểu đồ phân phối phân tán (Scatter/Boxplot):** Minh họa nồng độ đường huyết và HbA1c phân tầng theo tình trạng mắc bệnh, chỉ ra ngưỡng chẩn đoán lâm sàng $\geq 6.5\%$.

---
*Tài liệu thuộc khuôn khổ Assignment 05 — Intelligent Systems Development.*  
*Tác giả hiện thực: Sinh viên thực hiện đề tài.*
