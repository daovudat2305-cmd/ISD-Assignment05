"""
Basic Convolutional Neural Network (CNN) Architecture.
Includes Conv2D, Batch Normalization, ReLU, MaxPooling2D, and Dense classification head.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_basic_cnn(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    dropout_rate: float = 0.3,
    model_name: str = "Basic_CNN",
) -> models.Model:
    """
    Build a modern baseline CNN model.

    Architecture:
      - Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> Conv2D(32, 3x3) -> ReLU -> MaxPool2D(2x2) -> Dropout
      - Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> Conv2D(64, 3x3) -> ReLU -> MaxPool2D(2x2) -> Dropout
      - Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout
      - Classification: Flatten -> Dense(256) -> BatchNorm -> ReLU -> Dropout -> Dense(num_classes)

    Args:
        input_shape: (height, width, channels) of input images.
        num_classes: Number of target categories.
        dropout_rate: Dropout probability for regularization.
        model_name: Model identifier string.

    Returns:
        Compiled or uncompiled tf.keras.Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    # Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, name="dense_hidden")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(dropout_rate * 1.5 if dropout_rate * 1.5 < 0.7 else 0.5, name="drop_dense")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    model = build_basic_cnn()
    model.summary()
