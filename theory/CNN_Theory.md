# Lý Thuyết Mạng Nơ-ron Tích Chập (CNN)
## Convolutional Neural Network — Lý thuyết & Cài đặt

> **Assignment 05 — Phát triển các Hệ thống Thông minh**

---

# MỤC LỤC

## PHẦN 1: CÁC KHÁI NIỆM CƠ BẢN CNN
1. [Tổng quan về CNN](#1-tổng-quan-về-cnn)
2. [Tích chập (Convolution)](#2-tích-chập-convolution)
3. [Hàm kích hoạt (Activation Function)](#3-hàm-kích-hoạt-activation-function)
4. [Pooling (Subsampling)](#4-pooling-subsampling)
5. [Batch Normalization](#5-batch-normalization)
6. [Dropout](#6-dropout)
7. [Fully Connected Layer](#7-fully-connected-layer)
8. [Hàm mất mát (Loss Function)](#8-hàm-mất-mát-loss-function)
9. [Backpropagation trong CNN](#9-backpropagation-trong-cnn)
10. [Data Augmentation](#10-data-augmentation)

## PHẦN 2: CÁC MÔ HÌNH CẢI TIẾN
11. [LeNet-5 (1998)](#11-lenet-5-1998)
12. [AlexNet (2012)](#12-alexnet-2012)
13. [VGGNet (2014)](#13-vggnet-2014)
14. [GoogLeNet / Inception (2014)](#14-googlenet--inception-2014)
15. [ResNet (2015)](#15-resnet-2015)
16. [MobileNet (2017)](#16-mobilenet-2017)
17. [So sánh tổng thể các mô hình](#17-so-sánh-tổng-thể-các-mô-hình)

## PHẦN 3: TẬP DỮ LIỆU THỰC NGHIỆM
- 👉 **[Báo cáo chi tiết Phần 3: Khảo sát & Tiền xử lý 3 Tập Dữ Liệu](Part3_Datasets_Report.md)**

---

# PHẦN 1: CÁC KHÁI NIỆM CƠ BẢN CNN

---

## 1. Tổng quan về CNN

### 1.1 CNN là gì?

**Mạng nơ-ron tích chập (Convolutional Neural Network — CNN)** là một lớp mạng học sâu (deep learning) được thiết kế đặc biệt để xử lý dữ liệu có cấu trúc lưới (grid-like), đặc biệt là ảnh 2D. CNN tự động học các đặc trưng (features) từ dữ liệu thô thông qua quá trình huấn luyện, thay vì phải trích xuất đặc trưng thủ công như các phương pháp truyền thống.

**Ba tính chất cốt lõi của CNN:**

| Tính chất | Ý nghĩa | Lợi ích |
|-----------|---------|---------|
| **Local Connectivity** | Mỗi neuron chỉ kết nối với một vùng nhỏ (receptive field) | Giảm số tham số |
| **Parameter Sharing** | Cùng một bộ lọc (filter) được dùng trên toàn bộ ảnh | Học được đặc trưng bất biến vị trí |
| **Translation Invariance** | Phát hiện đặc trưng bất kể vị trí trong ảnh | Robust với dịch chuyển |

### 1.2 Kiến trúc tổng quát

```
Input Image -> [Conv -> Activation -> Pool] x N -> Flatten -> [FC -> Activation] x M -> Output
```

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_general_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Kiến trúc CNN tổng quát

    Luồng dữ liệu:
      Input (H, W, C) -> Conv Blocks -> Flatten -> FC Layers -> Softmax Output

    Args:
        input_shape: (height, width, channels)
        num_classes: số lớp phân loại
    Returns:
        model: Keras Sequential model
    """
    model = models.Sequential([
        # === FEATURE EXTRACTION ===
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # === CLASSIFICATION ===
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Xem tóm tắt kiến trúc
model = build_general_cnn()
model.summary()
```

### 1.3 So sánh CNN với MLP

```python
def compare_parameters():
    """
    So sánh số tham số giữa CNN và MLP
    cho ảnh đầu vào 32x32x3
    """
    # MLP: Mỗi pixel nối với mỗi neuron ở lớp ẩn
    mlp_input  = 32 * 32 * 3     # = 3,072
    mlp_hidden = 512
    mlp_params = mlp_input * mlp_hidden + mlp_hidden  # W + b

    # CNN: Mỗi filter chỉ có (kernel_h * kernel_w * channels + 1) tham số
    cnn_filters = 32
    kernel_size = 3 * 3 * 3       # 3x3 kernel, 3 channels
    cnn_params  = cnn_filters * (kernel_size + 1)

    print("=" * 50)
    print(f"MLP lớp ẩn đầu tiên : {mlp_params:,} tham số")
    print(f"CNN Conv lớp đầu tiên: {cnn_params:,} tham số")
    print(f"Giảm {mlp_params / cnn_params:.0f}x số tham số!")
    print("=" * 50)

compare_parameters()
# MLP lớp ẩn đầu tiên : 1,573,376 tham số
# CNN Conv lớp đầu tiên: 896 tham số
# Giảm ~1,756x số tham số!
```

---

## 2. Tích Chập (Convolution)

### 2.1 Định nghĩa toán học

Phép tích chập rời rạc 2D giữa ảnh **I** và kernel **K**:

```
(I * K)[i, j] = Sum_m Sum_n  I[i+m, j+n] * K[m, n]
```

Trong đó:
- `I[i, j]` — giá trị pixel tại hàng i, cột j
- `K[m, n]` — giá trị kernel tại hàng m, cột n
- Output tại `[i, j]` là tổng tích có trọng số của vùng lân cận

> **Lưu ý:** Trong thực tế, Keras/PyTorch thực hiện *cross-correlation* (không lật kernel), nhưng thường được gọi là convolution theo quy ước học sâu.

### 2.2 Cài đặt tích chập thủ công

```python
import numpy as np

def manual_conv2d(image, kernel, stride=1, padding=0):
    """
    Thực hiện phép tích chập 2D thủ công

    Args:
        image  : ndarray shape (H, W) — ảnh grayscale
        kernel : ndarray shape (kH, kW) — bộ lọc
        stride : bước nhảy (default=1)
        padding: số pixel thêm vào biên (default=0)
    Returns:
        output : ndarray — feature map sau tích chập

    Công thức kích thước output:
        H_out = (H_in + 2*padding - kH) / stride + 1
        W_out = (W_in + 2*padding - kW) / stride + 1
    """
    H, W = image.shape
    kH, kW = kernel.shape

    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
        H += 2 * padding
        W += 2 * padding

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            region = image[i*stride : i*stride+kH,
                           j*stride : j*stride+kW]
            output[i, j] = np.sum(region * kernel)

    return output


def demo_kernels():
    """Minh hoạ tác dụng của các kernel khác nhau"""
    image = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 2, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=float)

    # Phát hiện cạnh ngang (Sobel-x)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    # Làm mờ (Gaussian blur)
    gaussian = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float) / 16.0
    # Làm sắc nét (Sharpen)
    sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float)

    edge_map  = manual_conv2d(image, sobel_x,  stride=1, padding=1)
    blurred   = manual_conv2d(image, gaussian, stride=1, padding=1)
    sharpened = manual_conv2d(image, sharpen,  stride=1, padding=1)

    print("Original:\n", image)
    print("\nEdge detection (Sobel-x):\n", edge_map)
    print("\nBlurred (Gaussian):\n", blurred.round(2))
    print("\nSharpened:\n", sharpened)

demo_kernels()
```

### 2.3 Tính kích thước Feature Map

```python
def calc_output_size(H_in, W_in, kernel_size, stride=1, padding=0):
    """
    Tính kích thước feature map sau Conv2D

    Công thức:
        H_out = floor((H_in + 2*P - K) / S) + 1
        W_out = floor((W_in + 2*P - K) / S) + 1
    """
    H_out = (H_in + 2*padding - kernel_size) // stride + 1
    W_out = (W_in + 2*padding - kernel_size) // stride + 1
    return H_out, W_out

# Ví dụ
scenarios = [
    (32, 32, 3, 1, 0, "CIFAR-10, Conv 3x3, no padding"),
    (32, 32, 3, 1, 1, "CIFAR-10, Conv 3x3, same padding"),
    (224,224, 7, 2, 3, "ImageNet, Conv 7x7 stride=2"),
    (32, 32, 2, 2, 0, "MaxPool 2x2 stride=2"),
]
for H, W, K, S, P, desc in scenarios:
    Ho, Wo = calc_output_size(H, W, K, S, P)
    print(f"{desc}")
    print(f"  {H}x{W} -> {Ho}x{Wo}\n")
```

### 2.4 Receptive Field

```python
def calc_receptive_field(layers_config):
    """
    Tính receptive field tích lũy qua các lớp Conv/Pool

    Công thức đệ quy:
        RF_l = RF_{l-1} + (K_l - 1) * prod_{i=1}^{l-1} S_i

    Args:
        layers_config: list of (kernel_size, stride)
    Returns:
        rf_history: receptive field tại từng lớp
    """
    rf = 1
    stride_prod = 1
    rf_history = [rf]

    for (K, S) in layers_config:
        rf = rf + (K - 1) * stride_prod
        stride_prod *= S
        rf_history.append(rf)

    return rf_history

# Ví dụ: 3 lớp Conv 3x3 stride=1
config = [(3,1), (3,1), (3,1)]
print("Receptive field:", calc_receptive_field(config))
# [1, 3, 5, 7] → sau 3 lớp conv 3x3 thấy được vùng 7x7 ảnh gốc
```

---

## 3. Hàm Kích Hoạt (Activation Function)

Hàm kích hoạt đưa **phi tuyến** vào mạng, cho phép CNN học các biểu diễn phức tạp.

### 3.1 Sigmoid

```
sigma(x) = 1 / (1 + e^{-x})   ∈ (0, 1)
Gradient: sigma'(x) = sigma(x) * (1 - sigma(x))   max = 0.25 tại x=0
```

```python
import numpy as np

def sigmoid(x):
    """
    sigma(x) = 1 / (1 + e^{-x})
    Vấn đề: Vanishing Gradient khi |x| lớn (gradient ≈ 0)
    Dùng: Output layer binary classification
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

print(f"Sigmoid max gradient: {sigmoid_derivative(0):.4f}")  # 0.25
print(f"Sigmoid gradient tại x=5: {sigmoid_derivative(5):.6f}")  # ≈ 0.006
```

### 3.2 Tanh

```
tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})   ∈ (-1, 1)
Gradient: tanh'(x) = 1 - tanh^2(x)   max = 1 tại x=0
```

```python
def tanh(x):
    """
    tanh — zero-centered, tốt hơn Sigmoid
    Ưu điểm: Gradient lớn hơn Sigmoid (max=1 vs 0.25)
    Dùng: Hidden layers LeNet, RNN
    """
    return np.tanh(x)

print(f"Tanh max gradient: {1 - np.tanh(0)**2:.4f}")  # 1.0
```

### 3.3 ReLU — Phổ biến nhất trong CNN

```
ReLU(x) = max(0, x)
Gradient: 1 nếu x > 0; 0 nếu x <= 0
```

```python
def relu(x):
    """
    ReLU(x) = max(0, x)

    Lý do phổ biến nhất:
    - Gradient = 1 khi x > 0 → không bị vanishing gradient
    - Tính toán đơn giản, nhanh
    - Tạo biểu diễn sparse (nhiều neuron = 0)

    Vấn đề "Dying ReLU": nếu x luôn < 0 → neuron chết
    Khắc phục: Leaky ReLU, ELU, GELU
    """
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

x_neg = np.array([-5.0, -3.0, -1.0])
print(f"ReLU output  : {relu(x_neg)}")            # [0, 0, 0]
print(f"ReLU gradient: {relu_derivative(x_neg)}")  # [0, 0, 0] → dead neurons!
```

### 3.4 Leaky ReLU & ELU & GELU

```python
def leaky_relu(x, alpha=0.01):
    """
    LeakyReLU(x) = x nếu x > 0;  alpha*x nếu x <= 0
    Khắc phục dying ReLU: luôn có gradient ≠ 0
    PReLU: alpha là tham số học được
    """
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """
    ELU(x) = x nếu x > 0;  alpha*(e^x - 1) nếu x <= 0
    Ưu điểm: smooth, zero-centered, self-normalizing
    Nhược điểm: chậm hơn ReLU (tính exp)
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    """
    GELU(x) = x * Phi(x)   với Phi là CDF phân phối chuẩn
    Dùng trong: BERT, GPT, Transformer
    """
    return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

x_test = np.array([-1.0])
print(f"ReLU(-1)      : {relu(x_test)}")             # [0.0]
print(f"LeakyReLU(-1) : {leaky_relu(x_test)}")       # [-0.01]
print(f"ELU(-1)       : {elu(x_test).round(4)}")      # [-0.6321]
print(f"GELU(-1)      : {gelu(x_test).round(4)}")     # [-0.1588]
```

### 3.5 Softmax (Output Layer)

```
Softmax(z_i) = e^{z_i} / Sum_j e^{z_j}
```

```python
def softmax(z):
    """
    Softmax: chuyển vector logit → phân phối xác suất
    - Tổng tất cả output = 1
    - Mỗi output ∈ (0, 1)
    - Dùng: Output layer multi-class classification

    Trick ổn định số học: trừ max để tránh overflow
    """
    z_stable = z - np.max(z)
    exp_z    = np.exp(z_stable)
    return exp_z / np.sum(exp_z)

logits = np.array([2.0, 1.0, 0.1])
probs  = softmax(logits)
print(f"Logits  : {logits}")
print(f"Softmax : {probs.round(4)}")   # [0.6590, 0.2424, 0.0986]
print(f"Sum     : {probs.sum():.4f}")  # 1.0
```

---

## 4. Pooling (Subsampling)

Pooling **giảm chiều không gian** (spatial downsampling) của feature maps:
- Giảm số tham số và tính toán
- Tăng tính bất biến với dịch chuyển nhỏ
- Mở rộng receptive field

### 4.1 Max Pooling

```python
import numpy as np

def max_pooling2d(feature_map, pool_size=2, stride=2):
    """
    Max Pooling: lấy giá trị lớn nhất trong mỗi vùng pool

    Cơ chế: Giữ lại đặc trưng nổi bật nhất
    Gradient: Chỉ truyền về vị trí có max (switch)
    """
    H, W = feature_map.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            region = feature_map[i*stride : i*stride+pool_size,
                                 j*stride : j*stride+pool_size]
            output[i, j] = np.max(region)
    return output


def avg_pooling2d(feature_map, pool_size=2, stride=2):
    """
    Average Pooling: lấy giá trị trung bình

    Cơ chế: Làm mượt (smooth) đặc trưng
    Gradient: Phân phối đều về tất cả vị trí trong vùng
    Dùng: LeNet, Global Average Pooling
    """
    H, W = feature_map.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            region = feature_map[i*stride : i*stride+pool_size,
                                 j*stride : j*stride+pool_size]
            output[i, j] = np.mean(region)
    return output


fm = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 1, 4, 6],
    [0, 2, 5, 3]
], dtype=float)

print("Feature map (4x4):\n", fm)
print("\nMax Pooling 2x2:\n", max_pooling2d(fm))
# [[6, 4], [3, 6]]
print("\nAvg Pooling 2x2:\n", avg_pooling2d(fm))
# [[3.75, 2.25], [1.5, 4.5]]
```

### 4.2 Global Average Pooling (GAP)

```python
import tensorflow as tf

# GAP: Tính trung bình toàn bộ feature map mỗi channel
# (batch, 7, 7, 512) → (batch, 512)
# Thay thế Flatten + Dense → giảm tham số, chống overfitting
# Dùng trong: GoogLeNet, ResNet, MobileNet

gap_layer = tf.keras.layers.GlobalAveragePooling2D()
dummy_input = tf.random.normal((8, 7, 7, 512))
gap_output  = gap_layer(dummy_input)
print(f"Input shape : {dummy_input.shape}")  # (8, 7, 7, 512)
print(f"GAP output  : {gap_output.shape}")   # (8, 512)
```

---

## 5. Batch Normalization

### 5.1 Lý thuyết và Công thức

**Batch Normalization (BN)** chuẩn hoá đầu ra của mỗi lớp trong mini-batch, giải quyết vấn đề **Internal Covariate Shift** (phân phối input của lớp thay đổi theo từng batch).

```
Bước 1 — Tính statistics batch:
    mu_B  = (1/m) * Sum x_i
    var_B = (1/m) * Sum (x_i - mu_B)^2

Bước 2 — Chuẩn hoá:
    x_hat_i = (x_i - mu_B) / sqrt(var_B + epsilon)

Bước 3 — Scale và Shift (tham số học được):
    y_i = gamma * x_hat_i + beta
```

`gamma` (scale) và `beta` (shift) là tham số **có thể học được** (trainable).

```python
import numpy as np

def batch_normalization(x, gamma, beta, eps=1e-8):
    """
    Batch Normalization thủ công

    Args:
        x    : ndarray (batch_size, features)
        gamma: ndarray (features,) — scale (học được)
        beta : ndarray (features,) — shift (học được)
        eps  : tránh chia cho 0
    Returns:
        y    : normalized output
        cache: cho backward pass
    """
    mu  = np.mean(x, axis=0)
    var = np.var(x, axis=0)

    x_norm = (x - mu) / np.sqrt(var + eps)
    y = gamma * x_norm + beta

    return y, (x_norm, gamma, mu, var, eps)


np.random.seed(42)
batch  = np.random.randn(4, 3) * 10 + 5
gamma  = np.ones(3)
beta   = np.zeros(3)

out, _ = batch_normalization(batch, gamma, beta)
print("Input  — mean:", batch.mean(axis=0).round(2), "| std:", batch.std(axis=0).round(2))
print("Output — mean:", out.mean(axis=0).round(4),  "| std:", out.std(axis=0).round(4))
# Output mean ≈ [0, 0, 0], std ≈ [1, 1, 1]
```

### 5.2 Vị trí đặt BatchNorm và Lợi ích

```python
import tensorflow as tf
from tensorflow.keras import layers

# Thứ tự phổ biến nhất (được khuyến nghị):
# Conv → BN → Activation
model_bn = tf.keras.Sequential([
    layers.Conv2D(64, (3,3), use_bias=False, padding='same'),  # bias=False vì BN có beta
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2)
])

# Lợi ích của Batch Normalization:
# 1. Cho phép learning rate lớn hơn → huấn luyện nhanh hơn
# 2. Giảm phụ thuộc khởi tạo trọng số
# 3. Regularization nhẹ, giảm nhu cầu Dropout
# 4. Ổn định gradient, tăng tốc hội tụ
```

---

## 6. Dropout

### 6.1 Cơ chế hoạt động

**Dropout** ngẫu nhiên vô hiệu hoá tỷ lệ `p` neuron trong training, buộc mạng học đặc trưng **độc lập** và **phân tán**, chống overfitting.

```python
import numpy as np

def dropout(x, rate=0.5, training=True):
    """
    Dropout với Inverted Scaling

    Training : vô hiệu hoá xác suất `rate`, nhân 1/(1-rate) để giữ kỳ vọng
    Inference : giữ nguyên tất cả neuron (không dropout)

    Args:
        x       : ndarray — activations
        rate    : tỷ lệ dropout (0.2 - 0.5 thường dùng)
        training: bool
    Returns:
        out     : activations sau dropout
    """
    if not training:
        return x

    keep_prob = 1.0 - rate
    mask = (np.random.rand(*x.shape) < keep_prob).astype(float)
    out  = x * mask / keep_prob  # Inverted dropout

    return out


np.random.seed(0)
activations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

train_out = dropout(activations, rate=0.5, training=True)
infer_out = dropout(activations, rate=0.5, training=False)

print(f"Input     : {activations}")
print(f"Training  : {train_out}")   # Một số = 0, còn lại nhân 2x
print(f"Inference : {infer_out}")   # Giữ nguyên
```

### 6.2 Vị trí đặt Dropout

```python
import tensorflow as tf
from tensorflow.keras import layers

# Dropout đặt SAU Fully Connected layers
# Ít dùng sau Conv layers (Conv đã ít tham số)

model_dropout = tf.keras.Sequential([
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),         # Sau FC, trước FC cuối
    layers.Dense(10, activation='softmax')
])
```

---

## 7. Fully Connected Layer

### 7.1 Vai trò và Công thức

Sau Conv/Pool trích xuất đặc trưng, **Fully Connected (FC) layers** tổng hợp tất cả để phân loại.

```
y = f(W * x + b)
```

```python
import numpy as np

def fully_connected_forward(x, W, b, activation='relu'):
    """
    Forward pass của Fully Connected layer

    Args:
        x         : ndarray (batch, input_dim)
        W         : ndarray (input_dim, output_dim)
        b         : ndarray (output_dim,)
        activation: 'relu', 'sigmoid', 'softmax', 'none'
    Returns:
        out       : ndarray (batch, output_dim)
        cache     : (x, W, b, z) cho backward
    """
    z = x @ W + b  # Linear transform

    if   activation == 'relu'   : out = np.maximum(0, z)
    elif activation == 'sigmoid': out = 1 / (1 + np.exp(-z))
    elif activation == 'softmax':
        e_z = np.exp(z - z.max(axis=1, keepdims=True))
        out = e_z / e_z.sum(axis=1, keepdims=True)
    else:
        out = z

    return out, (x, W, b, z)


def flatten_and_fc_demo():
    """Minh hoạ Flatten + Fully Connected"""
    feature_maps = np.random.randn(2, 4, 4, 32)   # (batch=2, 4, 4, 32)
    flat = feature_maps.reshape(feature_maps.shape[0], -1)  # (2, 512)
    print(f"After Conv   : {feature_maps.shape}")
    print(f"After Flatten: {flat.shape}")

    W   = np.random.randn(512, 128) * 0.01
    b   = np.zeros(128)
    out, _ = fully_connected_forward(flat, W, b, 'relu')
    print(f"After FC     : {out.shape}")

flatten_and_fc_demo()
```

---

## 8. Hàm Mất Mát (Loss Function)

### 8.1 Cross-Entropy Loss

```
L = -(1/N) * Sum_i Sum_c  y_{i,c} * log(y_hat_{i,c})
```

```python
import numpy as np

def categorical_crossentropy(y_true, y_pred, eps=1e-15):
    """
    Categorical Cross-Entropy cho multi-class classification

    Args:
        y_true: ndarray (N, C) — one-hot encoded
        y_pred: ndarray (N, C) — softmax probabilities
        eps   : clip để tránh log(0)
    Returns:
        loss  : scalar — mean loss
    """
    N = y_true.shape[0]
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / N

def binary_crossentropy(y_true, y_pred, eps=1e-15):
    """
    Binary Cross-Entropy cho binary classification

    L = -(1/N) * Sum [y*log(y_hat) + (1-y)*log(1-y_hat)]
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


y_true = np.array([[0,1,0], [1,0,0]])
y_pred = np.array([[0.1,0.8,0.1], [0.7,0.2,0.1]])

loss = categorical_crossentropy(y_true, y_pred)
print(f"Cross-Entropy Loss: {loss:.4f}")
```

### 8.2 Loss Functions trong Keras

```python
import tensorflow as tf

# Sparse Categorical CE — labels là integer [0, 1, 2, ...]
loss_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Categorical CE — labels là one-hot [[1,0,0], [0,1,0], ...]
loss_cat    = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# Binary CE — binary classification
loss_bin    = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 9. Backpropagation trong CNN

### 9.1 Gradient Descent và Chain Rule

CNN huấn luyện bằng **SGD** + **Backpropagation** tính gradient hàm mất mát với từng tham số.

```
Cập nhật trọng số:
    W <- W - eta * dL/dW
    b <- b - eta * dL/db

Trong đó eta là learning rate.
```

### 9.2 Gradient qua Conv Layer

```python
import numpy as np

def conv2d_backward(d_out, x, kernel, stride=1):
    """
    Backward pass của Conv2D

    Chain rule:
        dL/dK[m,n] = Sum_{i,j} dL/dO[i,j] * I[i*s+m, j*s+n]
        dL/dI[i,j] = Sum_{m,n} dL/dO[(i-m)/s, (j-n)/s] * K[m,n]

    Args:
        d_out  : gradient dL/dOutput — (H_out, W_out)
        x      : input image — (H_in, W_in)
        kernel : — (kH, kW)
        stride : bước nhảy
    Returns:
        d_kernel: gradient với kernel (dL/dK)
        d_input : gradient với input  (dL/dI)
    """
    kH, kW = kernel.shape
    H_out, W_out = d_out.shape

    d_kernel = np.zeros_like(kernel)
    d_input  = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            region = x[i*stride : i*stride+kH, j*stride : j*stride+kW]
            d_kernel += d_out[i, j] * region
            d_input[i*stride : i*stride+kH, j*stride : j*stride+kW] += d_out[i, j] * kernel

    return d_kernel, d_input
```

### 9.3 Các Optimizer Hiện Đại

```python
import tensorflow as tf

# SGD với Momentum
optimizer_sgd = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,    # Momentum giúp vượt local minima
    nesterov=True    # Nesterov Accelerated Gradient
)

# Adam — phổ biến nhất, kết hợp Momentum + RMSProp
# Update rule:
#   m_t = beta1*m_{t-1} + (1-beta1)*g_t            (1st moment)
#   v_t = beta2*v_{t-1} + (1-beta2)*g_t^2           (2nd moment)
#   m_hat = m_t/(1-beta1^t),  v_hat = v_t/(1-beta2^t) (bias correction)
#   W   = W - lr * m_hat / (sqrt(v_hat) + eps)
optimizer_adam = tf.keras.optimizers.Adam(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

# RMSProp — tốt cho RNN
optimizer_rms = tf.keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)

print("SGD   — cần tune LR cẩn thận")
print("Adam  — adaptive LR, mặc định tốt, ít tune")
print("RMSProp — tốt cho sequence data")
```

---

## 10. Data Augmentation

**Data Augmentation** tăng cường dữ liệu training bằng phép biến đổi, giúp mô hình tổng quát hoá tốt hơn.

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Keras Augmentation Layers (chạy trên GPU) ===
augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),         # Lật ngang
    layers.RandomRotation(0.2),              # Xoay ±20%
    layers.RandomZoom(0.1),                  # Zoom ±10%
    layers.RandomTranslation(0.1, 0.1),      # Dịch chuyển ±10%
    layers.RandomContrast(0.2),              # Contrast
    layers.RandomBrightness(0.2),            # Brightness
], name="data_augmentation")


# === ImageDataGenerator (cách cũ, vẫn dùng nhiều) ===
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize [0,1]
    rotation_range=30,        # Xoay 0-30 độ
    width_shift_range=0.2,    # Dịch ngang 20%
    height_shift_range=0.2,   # Dịch dọc 20%
    shear_range=0.15,         # Biến dạng cắt
    zoom_range=0.15,          # Zoom ngẫu nhiên
    horizontal_flip=True,     # Lật ngang
    fill_mode='nearest'       # Điền pixel thiếu
)

# Tại sao cần Data Augmentation?
# - Dataset nhỏ → mô hình overfit
# - Ảnh thực tế có nhiều góc độ, ánh sáng khác nhau
# - Nhân đa dạng dữ liệu không cần thu thập thêm
```

---

# PHẦN 2: CÁC MÔ HÌNH CẢI TIẾN

---

## 11. LeNet-5 (1998)

### 11.1 Lịch sử và Ý nghĩa

**LeNet-5** do **Yann LeCun** và cộng sự đề xuất năm 1998. Đây là CNN đầu tiên áp dụng thành công vào thực tế (nhận dạng chữ số viết tay MNIST), đặt nền móng cho toàn bộ kiến trúc CNN hiện đại.

**Ý nghĩa lịch sử:**
- Lần đầu tiên chứng minh CNN có thể học đặc trưng tự động
- Được triển khai thực tế trong hệ thống đọc séc ngân hàng Mỹ
- Thiết lập quy trình: Conv → Pool → FC là "recipe" chuẩn

### 11.2 Kiến trúc chi tiết

```
Input (32x32x1)
    |
C1: Conv2D(6, 5x5, stride=1)    -> (28x28x6)     [156 params]
    |
S2: AvgPool(2x2, stride=2)      -> (14x14x6)
    |
C3: Conv2D(16, 5x5, stride=1)   -> (10x10x16)    [2,416 params]
    |
S4: AvgPool(2x2, stride=2)      -> (5x5x16)
    |
C5: Conv2D(120, 5x5, stride=1)  -> (1x1x120)     [48,120 params]
    |
F6: Dense(84, tanh)             -> (84,)          [10,164 params]
    |
Output: Dense(10, softmax)      -> (10,)          [850 params]
                                         Total ≈ 60,000 params
```

### 11.3 Cài đặt LeNet-5

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lenet5(input_shape=(32, 32, 1), num_classes=10):
    """
    LeNet-5 (LeCun et al., 1998)

    Kiến trúc gốc dùng:
      - Sigmoid/Tanh activation (trước thời ReLU)
      - Average Pooling (thay vì Max Pooling)
      - ~60,000 tham số

    Phiên bản này dùng tanh theo đúng paper gốc.
    """
    model = models.Sequential(name='LeNet-5')

    # C1: Phát hiện đặc trưng cơ bản (cạnh, góc...)
    model.add(layers.Conv2D(6, (5, 5), strides=1, activation='tanh',
                            input_shape=input_shape, name='C1_conv'))
    # → (28, 28, 6)

    # S2: Subsampling
    model.add(layers.AveragePooling2D((2, 2), strides=2, name='S2_pool'))
    # → (14, 14, 6)

    # C3: Kết hợp đặc trưng phức tạp hơn
    model.add(layers.Conv2D(16, (5, 5), strides=1, activation='tanh',
                            name='C3_conv'))
    # → (10, 10, 16)

    # S4: Subsampling
    model.add(layers.AveragePooling2D((2, 2), strides=2, name='S4_pool'))
    # → (5, 5, 16)

    # C5: Thực chất là FC vì output là 1x1
    model.add(layers.Conv2D(120, (5, 5), strides=1, activation='tanh',
                            name='C5_conv'))
    # → (1, 1, 120)

    model.add(layers.Flatten())
    # → (120,)

    # F6: Fully Connected
    model.add(layers.Dense(84, activation='tanh', name='F6_fc'))

    # Output
    model.add(layers.Dense(num_classes, activation='softmax', name='Output'))

    return model

lenet = build_lenet5(input_shape=(32, 32, 1), num_classes=10)
lenet.summary()
```

### 11.4 Ưu điểm và Hạn chế

| Ưu điểm | Hạn chế |
|---------|---------|
| Kiến trúc đơn giản, dễ hiểu | Tanh/Sigmoid → Vanishing gradient |
| Nền tảng cho CNN sau này | Average pooling kém MaxPooling |
| Ít tham số (~60K) | Không phù hợp ảnh phức tạp |
| Chứng minh CNN > thủ công | Rất nông (5 lớp có tham số) |

---

## 12. AlexNet (2012)

### 12.1 Lịch sử — Điểm khởi đầu của Deep Learning hiện đại

**AlexNet** (Krizhevsky, Sutskever, Hinton) **giành chiến thắng ImageNet ILSVRC 2012** với top-5 error 15.3% (giảm 10.8% so với thứ hai — một khoảng cách khổng lồ).

**Bốn đóng góp cốt lõi:**
1. **ReLU activation** → giải quyết vanishing gradient, nhanh hơn 6x
2. **Dropout (0.5)** → regularization mạnh, giảm overfitting
3. **Data Augmentation** → lật ngang, crop ngẫu nhiên, tăng dữ liệu 2048x
4. **Multi-GPU Training** → chia mạng sang 2 GPU GTX 580

### 12.2 Kiến trúc

```
Input (227x227x3)
    |
Conv1: 96 filters, 11x11, stride=4  -> (55x55x96)     [34,944 params]
BN1 + MaxPool(3x3, stride=2)        -> (27x27x96)
    |
Conv2: 256 filters, 5x5, pad=2      -> (27x27x256)    [614,656 params]
BN2 + MaxPool(3x3, stride=2)        -> (13x13x256)
    |
Conv3: 384 filters, 3x3, pad=1      -> (13x13x384)    [885,120 params]
    |
Conv4: 384 filters, 3x3, pad=1      -> (13x13x384)    [1,327,488 params]
    |
Conv5: 256 filters, 3x3, pad=1      -> (13x13x256)    [884,992 params]
MaxPool(3x3, stride=2)              -> (6x6x256)
    |
Flatten                             -> (9216,)
FC6: Dense(4096) + ReLU + Dropout  -> (4096,)
FC7: Dense(4096) + ReLU + Dropout  -> (4096,)
FC8: Dense(1000) + Softmax          -> (1000,)
                            Total ≈ 62.3M params
```

### 12.3 Cài đặt AlexNet

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_alexnet(input_shape=(227, 227, 3), num_classes=1000):
    """
    AlexNet (Krizhevsky et al., 2012)

    Thay LRN bằng BatchNormalization (hiệu quả hơn trong thực tế)
    """
    model = models.Sequential(name='AlexNet')

    # Conv Block 1 — phát hiện cạnh thô, texture lớn
    model.add(layers.Conv2D(96, (11,11), strides=4, activation='relu',
                            input_shape=input_shape, padding='valid', name='Conv1'))
    model.add(layers.BatchNormalization(name='BN1'))
    model.add(layers.MaxPooling2D((3,3), strides=2, name='Pool1'))

    # Conv Block 2 — pattern trung bình
    model.add(layers.Conv2D(256, (5,5), strides=1, activation='relu',
                            padding='same', name='Conv2'))
    model.add(layers.BatchNormalization(name='BN2'))
    model.add(layers.MaxPooling2D((3,3), strides=2, name='Pool2'))

    # Conv Block 3, 4, 5 — pattern phức tạp, semantic
    model.add(layers.Conv2D(384, (3,3), strides=1, activation='relu',
                            padding='same', name='Conv3'))
    model.add(layers.Conv2D(384, (3,3), strides=1, activation='relu',
                            padding='same', name='Conv4'))
    model.add(layers.Conv2D(256, (3,3), strides=1, activation='relu',
                            padding='same', name='Conv5'))
    model.add(layers.MaxPooling2D((3,3), strides=2, name='Pool5'))

    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu', name='FC6'))
    model.add(layers.Dropout(0.5, name='Drop6'))
    model.add(layers.Dense(4096, activation='relu', name='FC7'))
    model.add(layers.Dropout(0.5, name='Drop7'))
    model.add(layers.Dense(num_classes, activation='softmax', name='Output'))

    return model


def build_alexnet_small(input_shape=(32, 32, 3), num_classes=10):
    """
    AlexNet thu nhỏ cho CIFAR-10 / ảnh 32x32
    Không dùng stride=4 vì ảnh quá nhỏ
    """
    model = models.Sequential(name='AlexNet_Small')

    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(192, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(384, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
```

---

## 13. VGGNet (2014)

### 13.1 Triết lý thiết kế — "Uniformity through Depth"

**VGGNet** (Simonyan & Zisserman, Oxford, 2014). Tư tưởng: **độ sâu là yếu tố quan trọng nhất**. Chỉ dùng kernel 3x3, padding=1 (same) — **cực kỳ đồng nhất**.

### 13.2 Tại sao 3x3 tốt hơn 5x5 hay 7x7?

```python
def explain_3x3_advantage():
    """
    2 Conv 3x3 ≡ 1 Conv 5x5 về receptive field
    nhưng ÍT tham số hơn và CÓ THÊM phi tuyến

    Receptive field:
        1 Conv 3x3 -> RF = 3x3
        2 Conv 3x3 -> RF = 5x5 (tương đương 1 Conv 5x5)
        3 Conv 3x3 -> RF = 7x7 (tương đương 1 Conv 7x7)

    So sánh tham số (C channels):
        1 Conv 5x5: 5*5*C*C = 25*C^2 params
        2 Conv 3x3: 2*(3*3*C*C) = 18*C^2 params  → ít hơn 28%!

        1 Conv 7x7: 49*C^2 params
        3 Conv 3x3: 27*C^2 params  → ít hơn 45%!

    Thêm vào đó: 2 Conv 3x3 có 2 ReLU, 1 Conv 5x5 chỉ có 1 ReLU
    → Phi tuyến phong phú hơn!
    """
    C = 64
    print("=== So sánh tham số ===")
    print(f"1 Conv 5x5: {5*5*C*C:,} params")
    print(f"2 Conv 3x3: {2*3*3*C*C:,} params  (tiết kiệm {(1 - 2*9/(25))*100:.0f}%)")
    print()
    print(f"1 Conv 7x7: {7*7*C*C:,} params")
    print(f"3 Conv 3x3: {3*3*3*C*C:,} params  (tiết kiệm {(1 - 3*9/(49))*100:.0f}%)")

explain_3x3_advantage()
```

### 13.3 Cài đặt VGG-16

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg16(input_shape=(224, 224, 3), num_classes=1000):
    """
    VGG-16 (Simonyan & Zisserman, 2014)

    16 lớp có tham số: 13 Conv + 3 FC
    Uniform design: tất cả Conv đều 3x3, same padding
    Total: 138.4M parameters
    """
    model = models.Sequential(name='VGG-16')

    def vgg_conv_block(filters, num_convs, block_name):
        for i in range(1, num_convs + 1):
            model.add(layers.Conv2D(filters, (3,3), activation='relu',
                                    padding='same', name=f'{block_name}_conv{i}'))
        model.add(layers.MaxPooling2D((2,2), strides=2, name=f'{block_name}_pool'))

    model.add(layers.Input(shape=input_shape))

    vgg_conv_block(64,  2, 'block1')   # -> (112, 112, 64)
    vgg_conv_block(128, 2, 'block2')   # -> (56,  56, 128)
    vgg_conv_block(256, 3, 'block3')   # -> (28,  28, 256)
    vgg_conv_block(512, 3, 'block4')   # -> (14,  14, 512)
    vgg_conv_block(512, 3, 'block5')   # -> (7,   7,  512)

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu', name='fc2'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', name='predictions'))

    return model


def build_vgg16_transfer(num_classes=5):
    """
    VGG-16 Transfer Learning cho dataset nhỏ (ví dụ: Flowers 5 lớp)

    Phase 1 — Feature Extraction:
        Freeze base, train ONLY custom head. LR=1e-3
    Phase 2 — Fine-tuning:
        Unfreeze top N layers. LR=1e-5
    """
    from tensorflow.keras.applications import VGG16

    base_model = VGG16(input_shape=(224, 224, 3), include_top=False,
                       weights='imagenet')
    base_model.trainable = False  # Freeze

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name='VGG16_Transfer')
```

---

## 14. GoogLeNet / Inception (2014)

### 14.1 Vấn đề và Giải pháp

**GoogLeNet** (Szegedy et al., 2014) hỏi: *"Kernel nào tốt nhất: 1x1, 3x3, hay 5x5?"*. Thay vì chọn một — **dùng tất cả song song** → **Inception Module**.

### 14.2 Inception Module

```python
import tensorflow as tf
from tensorflow.keras import layers

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                     filters_5x5_reduce, filters_5x5, filters_pool_proj):
    """
    Inception Module (Szegedy et al., 2014)

    Thực hiện song song 4 nhánh rồi Concatenate:
      Nhánh 1: 1x1 Conv           — đặc trưng điểm
      Nhánh 2: 1x1 -> 3x3 Conv   — đặc trưng vùng nhỏ
      Nhánh 3: 1x1 -> 5x5 Conv   — đặc trưng vùng lớn
      Nhánh 4: MaxPool -> 1x1    — giữ thông tin pooling

    Bottleneck 1x1 giảm kênh trước 3x3 và 5x5 → giảm tham số
    """
    # Nhánh 1: 1x1 Conv
    branch1 = layers.Conv2D(filters_1x1, (1,1), padding='same',
                            activation='relu')(x)

    # Nhánh 2: 1x1 Bottleneck -> 3x3
    branch2 = layers.Conv2D(filters_3x3_reduce, (1,1), padding='same',
                            activation='relu')(x)
    branch2 = layers.Conv2D(filters_3x3, (3,3), padding='same',
                            activation='relu')(branch2)

    # Nhánh 3: 1x1 Bottleneck -> 5x5
    branch3 = layers.Conv2D(filters_5x5_reduce, (1,1), padding='same',
                            activation='relu')(x)
    branch3 = layers.Conv2D(filters_5x5, (5,5), padding='same',
                            activation='relu')(branch3)

    # Nhánh 4: MaxPool -> 1x1 Projection
    branch4 = layers.MaxPooling2D((3,3), strides=1, padding='same')(x)
    branch4 = layers.Conv2D(filters_pool_proj, (1,1), padding='same',
                            activation='relu')(branch4)

    # Concatenate 4 nhánh theo chiều channels
    output = layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    return output


def explain_1x1_conv():
    """
    1x1 Convolution — 'Network in Network' (Lin et al., 2013)

    Tác dụng:
    1. Giảm chiều (dimensionality reduction): 256 -> 64 channels
    2. Thêm phi tuyến tính (ReLU sau mỗi 1x1)
    3. Kết hợp thông tin across channels
    """
    C = 256  # Input channels
    bottleneck = 64

    no_bottle  = 3 * 3 * C * C                                 # 589,824
    with_bottle = (1*1*C*bottleneck) + (3*3*bottleneck*C)      # 163,840

    print(f"Không bottleneck (3x3): {no_bottle:,} params")
    print(f"Có bottleneck (1x1->3x3): {with_bottle:,} params")
    print(f"Tiết kiệm: {(1 - with_bottle/no_bottle)*100:.1f}%")  # ~72%

explain_1x1_conv()
```

---

## 15. ResNet (2015) — Skip Connections

### 15.1 Vấn đề Degradation

Khi tăng độ sâu, lẽ ra performance tốt hơn. Nhưng mạng >20 lớp lại cho **training error cao hơn** mạng nông do **vanishing gradient** và **degradation problem**.

**Giải pháp (He et al., 2015):** **Skip connections** cho gradient "bypass" qua các lớp.

### 15.2 Residual Block — Trái tim của ResNet

```
H(x) = F(x) + x

H(x) — đầu ra mong muốn
F(x) — đầu ra sau 2-3 lớp Conv (phần dư cần học)
x    — identity shortcut (đường tắt)

Thay vì học H(x) trực tiếp,
mạng học PHẦN DƯ: F(x) = H(x) - x
Dễ hơn vì: nếu H(x) ≈ x thì F(x) ≈ 0

Gradient qua ResNet:
    dL/dx_l = dL/dx_L * (1 + d/dx_l Sum F_i)
Số hạng "1" đảm bảo gradient KHÔNG BAO GIỜ = 0!
```

### 15.3 Cài đặt Residual Block và ResNet-50

```python
import tensorflow as tf
from tensorflow.keras import layers

def residual_block(x, filters, stride=1, name='res_block'):
    """
    Basic Residual Block (dùng trong ResNet-18, ResNet-34)

    Cấu trúc:
        x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> (+ x) -> ReLU

    Shortcut: Identity nếu shape không đổi, hoặc 1x1 Projection nếu thay đổi

    Args:
        x      : input tensor
        filters: số filters
        stride : stride của Conv đầu tiên (=2 để downsample)
        name   : tên block
    """
    shortcut = x

    # --- Nhánh chính F(x) ---
    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same',
                      use_bias=False, name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)

    x = layers.Conv2D(filters, (3,3), strides=1, padding='same',
                      use_bias=False, name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)

    # --- Shortcut Connection ---
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride,
                                 use_bias=False, name=f'{name}_proj')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_proj_bn')(shortcut)

    # H(x) = F(x) + x  ← Trái tim của ResNet
    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{name}_relu2')(x)

    return x


def bottleneck_block(x, filters, stride=1, name='bottleneck'):
    """
    Bottleneck Residual Block (dùng trong ResNet-50, 101, 152)

    Cấu trúc 3 lớp: 1x1 (giảm) -> 3x3 (main) -> 1x1 (tăng)
    Expansion factor = 4

    Ví dụ với filters=64:
        Conv 1x1: C_in -> 64      (giảm kênh)
        Conv 3x3: 64   -> 64      (main conv)
        Conv 1x1: 64   -> 256     (tăng kênh lại, expansion=4)
    """
    expansion = 4
    shortcut = x

    x = layers.Conv2D(filters, (1,1), use_bias=False, name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same',
                      use_bias=False, name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters * expansion, (1,1), use_bias=False,
                      name=f'{name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{name}_bn3')(x)

    if stride != 1 or shortcut.shape[-1] != filters * expansion:
        shortcut = layers.Conv2D(filters * expansion, (1,1), strides=stride,
                                 use_bias=False, name=f'{name}_proj')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_proj_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_resnet50(input_shape=(224, 224, 3), num_classes=1000):
    """
    ResNet-50 (He et al., 2015) — 25.6M tham số

    Cấu trúc:
        Conv1 (7x7, 64, stride=2)  -> (112, 112, 64)
        MaxPool (3x3, stride=2)     -> (56,  56,  64)
        Layer 1: 3 Bottlenecks 64   -> (56,  56,  256)
        Layer 2: 4 Bottlenecks 128  -> (28,  28,  512)
        Layer 3: 6 Bottlenecks 256  -> (14,  14, 1024)
        Layer 4: 3 Bottlenecks 512  -> (7,   7,  2048)
        GAP                         -> (2048,)
        Dense(num_classes)          -> (num_classes,)
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, (7,7), strides=2, padding='same',
                      use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same', name='pool1')(x)

    # Layer 1 — 3 bottlenecks (64 filters)
    for i in range(3):
        x = bottleneck_block(x, 64, stride=1, name=f'L1_b{i+1}')

    # Layer 2 — 4 bottlenecks (128, stride=2 đầu)
    x = bottleneck_block(x, 128, stride=2, name='L2_b1')
    for i in range(1, 4):
        x = bottleneck_block(x, 128, stride=1, name=f'L2_b{i+1}')

    # Layer 3 — 6 bottlenecks (256)
    x = bottleneck_block(x, 256, stride=2, name='L3_b1')
    for i in range(1, 6):
        x = bottleneck_block(x, 256, stride=1, name=f'L3_b{i+1}')

    # Layer 4 — 3 bottlenecks (512)
    x = bottleneck_block(x, 512, stride=2, name='L4_b1')
    for i in range(1, 3):
        x = bottleneck_block(x, 512, stride=1, name=f'L4_b{i+1}')

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='predictions')(x)

    return tf.keras.Model(inputs, outputs, name='ResNet-50')


def build_resnet50_transfer(num_classes=5):
    """ResNet-50 Transfer Learning"""
    from tensorflow.keras.applications import ResNet50
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(224, 224, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name='ResNet50_Transfer')
```

---

## 16. MobileNet (2017)

### 16.1 Động lực — CNN cho thiết bị di động

**MobileNet** (Howard et al., 2017) thiết kế CNN nhẹ cho thiết bị mobile/embedded, dựa trên **Depthwise Separable Convolution**.

### 16.2 Depthwise Separable Convolution

```python
import tensorflow as tf
from tensorflow.keras import layers

def explain_depthwise_separable():
    """
    Standard Conv vs Depthwise Separable Conv

    Standard Conv (D_K x D_K x M x N):
        Mỗi filter xử lý TẤT CẢ M kênh
        Phép tính: D_K^2 * M * N * D_F^2

    Depthwise Separable = 2 bước:

    Bước 1 — Depthwise Conv (D_K x D_K x M x 1):
        Mỗi kênh tích chập ĐỘC LẬP với 1 filter riêng
        Học spatial features PER channel
        Phép tính: D_K^2 * M * D_F^2

    Bước 2 — Pointwise Conv (1 x 1 x M x N):
        Kết hợp thông tin across channels
        Phép tính: M * N * D_F^2

    Tỷ lệ tiết kiệm:
        (D_K^2*M*D_F^2 + M*N*D_F^2) / (D_K^2*M*N*D_F^2)
        = 1/N + 1/D_K^2

        Với D_K=3, N=512: tiết kiệm ~8-9x!
    """
    D_K, M, N, D_F = 3, 256, 512, 14

    standard = D_K**2 * M * N * D_F**2
    depthwise = D_K**2 * M * D_F**2 + M * N * D_F**2

    print(f"Standard Conv  : {standard:,} phép tính")
    print(f"Depthwise Sep. : {depthwise:,} phép tính")
    print(f"Tiết kiệm      : {standard/depthwise:.1f}x")

explain_depthwise_separable()


def build_mobilenet_block(x, filters, stride=1, alpha=1.0, name='dw_block'):
    """
    MobileNet Depthwise Separable Block

    Args:
        x      : input tensor
        filters: số output channels
        stride : stride cho depthwise conv
        alpha  : width multiplier (0.25, 0.5, 0.75, 1.0)
        name   : tên block
    """
    filters = int(filters * alpha)

    # Depthwise Convolution — mỗi kênh tích chập riêng
    x = layers.DepthwiseConv2D((3,3), strides=stride, padding='same',
                               use_bias=False, name=f'{name}_dw')(x)
    x = layers.BatchNormalization(name=f'{name}_dw_bn')(x)
    x = layers.ReLU(6., name=f'{name}_dw_relu6')(x)  # ReLU6: min(max(0,x), 6)

    # Pointwise Convolution — kết hợp channels
    x = layers.Conv2D(filters, (1,1), padding='same', use_bias=False,
                      name=f'{name}_pw')(x)
    x = layers.BatchNormalization(name=f'{name}_pw_bn')(x)
    x = layers.ReLU(6., name=f'{name}_pw_relu6')(x)

    return x


def build_mobilenetv2_transfer(num_classes=5):
    """
    MobileNetV2 Transfer Learning — tốt nhất cho mobile deployment

    MobileNetV2 cải tiến V1:
    - Inverted Residuals: Expand -> Depthwise -> Project
    - Linear Bottleneck: không dùng ReLU tại lớp cuối bottleneck
      (ReLU phá thông tin khi chiều thấp)
    """
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False,
                       weights='imagenet', alpha=1.0)
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name='MobileNetV2_Transfer')
```

---

## 17. So Sánh Tổng Thể Các Mô Hình

### 17.1 Bảng so sánh

| Mô hình | Năm | Độ sâu | Tham số | Top-5 Err (ImageNet) | Đặc trưng chính |
|---------|-----|--------|---------|----------------------|-----------------|
| **LeNet-5** | 1998 | 5 | ~60K | N/A | Conv+Pool pioneer |
| **AlexNet** | 2012 | 8 | 62.3M | 15.3% | ReLU, Dropout, GPU |
| **VGG-16** | 2014 | 16 | 138.4M | 7.3% | 3x3 uniform depth |
| **VGG-19** | 2014 | 19 | 143.7M | 7.3% | Deeper VGG |
| **GoogLeNet** | 2014 | 22 | 6.8M | 6.7% | Inception module |
| **ResNet-50** | 2015 | 50 | 25.6M | 5.25% | Skip connections |
| **ResNet-152** | 2015 | 152 | 60.4M | 4.49% | Ultra-deep |
| **MobileNetV1** | 2017 | 28 | 4.2M | — | Depthwise Sep. |
| **MobileNetV2** | 2018 | 53 | 3.4M | — | Inverted Residuals |

### 17.2 Timeline tiến hóa

```
1998 — LeNet-5        : Nền tảng CNN (MNIST)
  |
2012 — AlexNet        : Cách mạng Deep Learning (GPU + ReLU + Dropout)
  |
2014 — VGGNet         : Uniformity, depth > large kernels
  |
2014 — GoogLeNet      : Inception module, multi-scale parallel
  |
2015 — ResNet         : Skip connections -> siêu sâu, vượt human
  |
2017 — MobileNet      : Efficient CNN cho thiết bị di động
  |
2018 — MobileNetV2    : Inverted residuals + linear bottleneck
  |
2019 — EfficientNet   : Neural Architecture Search, compound scaling
  |
2020 — ViT            : Vision Transformer - attention thay Conv
  |
2021+ — Swin, ConvNeXt, ...
```

### 17.3 Transfer Learning Pipeline Chuẩn

```python
import tensorflow as tf
from tensorflow.keras import layers

def transfer_learning_pipeline(base_model_name='resnet50',
                                num_classes=5,
                                input_shape=(224, 224, 3)):
    """
    Pipeline Transfer Learning chuẩn 2 phase

    Phase 1 — Feature Extraction (10-20 epochs):
        - Freeze toàn bộ base model
        - Train ONLY custom head
        - LR = 1e-3

    Phase 2 — Fine-tuning (10-20 epochs):
        - Unfreeze top N layers của base
        - Train với LR rất nhỏ
        - LR = 1e-5
    """
    from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2

    base_map = {'resnet50': ResNet50, 'vgg16': VGG16, 'mobilenetv2': MobileNetV2}
    base = base_map[base_model_name](weights='imagenet', include_top=False,
                                      input_shape=input_shape)
    base.trainable = False  # Phase 1: Freeze

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Phase 1 compile
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"=== Phase 1: Feature Extraction ===")
    print(f"Trainable params: {model.count_params():,}")

    # === Sau khi train Phase 1, chạy Phase 2: ===
    # Unfreeze 30 layers cuối
    base.trainable = True
    fine_tune_at = len(base.layers) - 30
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"\n=== Phase 2: Fine-tuning (last 30 layers) ===")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable params: {trainable:,}")

    return model


### 17.4 Gợi ý chọn mô hình

| Tình huống | Gợi ý mô hình | Lý do |
|-----------|---------------|-------|
| Dataset nhỏ (<5K) + PC | VGG16 / ResNet50 Transfer Learning (freeze) | ImageNet weights tổng quát |
| Dataset vừa (5K-100K) + PC | ResNet50 Fine-tuning | Cân bằng acc/speed |
| Dataset lớn (>100K) + Server | ResNet152 / EfficientNet from scratch | Đủ data để train sâu |
| Mobile / Embedded | MobileNetV2 Transfer Learning | Ít tham số, nhanh |
| Nghiên cứu / Học thuật | Custom ResNet | Hiểu sâu kiến trúc |
```

---

## Tóm Tắt Toàn Bộ Lý Thuyết

### Phần 1 — Các thành phần cốt lõi CNN

| Thành phần | Chức năng | Tham số quan trọng |
|-----------|----------|-------------------|
| **Conv2D** | Trích xuất đặc trưng không gian | filters, kernel_size, stride, padding |
| **ReLU** | Phi tuyến, chống vanishing gradient | — |
| **MaxPooling** | Giảm chiều, tăng invariance | pool_size, stride |
| **BatchNorm** | Ổn định training, regularization | epsilon, momentum |
| **Dropout** | Regularization, chống overfitting | rate (0.2-0.5) |
| **Dense + Softmax** | Phân loại cuối cùng | units, activation |

### Phần 2 — Tiến hóa kiến trúc CNN

| Mô hình | Đóng góp chính | Khi nào dùng |
|---------|---------------|-------------|
| **LeNet-5** | Nền tảng Conv+Pool | Học thuật, demo cơ bản |
| **AlexNet** | ReLU + Dropout | Baseline cho ảnh lớn |
| **VGGNet** | 3x3 uniform, depth | Transfer learning đơn giản |
| **GoogLeNet** | Inception, multi-scale | Balance acc/speed |
| **ResNet** | Skip connections | Default choice mọi bài toán |
| **MobileNet** | Depthwise separable | Mobile/embedded deployment |

---

*Tài liệu: Assignment 05 — Phát triển các Hệ thống Thông minh*
*Tham khảo:*
- *LeCun et al. (1998) — Gradient-Based Learning Applied to Document Recognition*
- *Krizhevsky et al. (2012) — ImageNet Classification with Deep Convolutional Neural Networks*
- *Simonyan & Zisserman (2014) — Very Deep Convolutional Networks for Large-Scale Image Recognition*
- *Szegedy et al. (2014) — Going Deeper with Convolutions*
- *He et al. (2015) — Deep Residual Learning for Image Recognition*
- *Howard et al. (2017) — MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*
