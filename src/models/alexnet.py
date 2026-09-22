"""
AlexNet Architecture (Krizhevsky et al., 2012).
Adapted for variable input sizes (CIFAR-10 32x32 and Flowers 128x128/224x224)
with Batch Normalization replacing Local Response Normalization.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_alexnet(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    dense_units: int = 512,
    dropout_rate: float = 0.5,
    model_name: str = "AlexNet",
) -> models.Model:
    """
    Build AlexNet architecture adapted for arbitrary image resolutions.

    Args:
        input_shape: (H, W, C), e.g. (32, 32, 3) or (128, 128, 3) or (224, 224, 3).
        num_classes: Number of classification targets.
        dense_units: Dimensionality of fully connected layers (512 for small, 4096 for ImageNet scale).
        dropout_rate: Dropout rate for dense layers.
        model_name: Model name.

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    h, w, _ = input_shape

    # For smaller images (e.g. 32x32), use smaller initial kernels and stride 1 to preserve spatial dimensions.
    if h <= 64:
        c1_kernel = (3, 3)
        c1_stride = (1, 1)
        c1_pad = "same"
    else:
        c1_kernel = (11, 11)
        c1_stride = (4, 4)
        c1_pad = "valid"

    # Layer 1: Conv + BN + ReLU + MaxPool
    x = layers.Conv2D(96, kernel_size=c1_kernel, strides=c1_stride, padding=c1_pad, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    # Layer 2: Conv + BN + ReLU + MaxPool
    x = layers.Conv2D(256, kernel_size=(5, 5), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool2")(x)

    # Layer 3: Conv + BN + ReLU
    x = layers.Conv2D(384, kernel_size=(3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)

    # Layer 4: Conv + BN + ReLU
    x = layers.Conv2D(384, kernel_size=(3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)

    # Layer 5: Conv + BN + ReLU + MaxPool
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.Activation("relu", name="relu5")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool5")(x)

    # Fully Connected Layers (FC6, FC7)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, name="fc6")(x)
    x = layers.BatchNormalization(name="bn6")(x)
    x = layers.Activation("relu", name="relu6")(x)
    x = layers.Dropout(dropout_rate, name="drop6")(x)

    x = layers.Dense(dense_units, name="fc7")(x)
    x = layers.BatchNormalization(name="bn7")(x)
    x = layers.Activation("relu", name="relu7")(x)
    x = layers.Dropout(dropout_rate, name="drop7")(x)

    # Output (FC8)
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    model = build_alexnet()
    model.summary()
