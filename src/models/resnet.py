"""
ResNet Architecture (He et al., 2015).
Provides:
  1. Custom ResNet architecture with explicit Residual Blocks F(x) + x.
  2. ResNet-50 Pretrained Transfer Learning model via keras.applications.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


def residual_block(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    name_prefix: str = "res",
) -> tf.Tensor:
    """
    Standard Residual block:
      Shortcut: x (or Conv 1x1 if dimension changes)
      Path: Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN
      Output: ReLU(Path + Shortcut)
    """
    shortcut = x

    # First conv
    y = layers.Conv2D(
        filters,
        (3, 3),
        strides=(stride, stride),
        padding="same",
        name=f"{name_prefix}_conv1",
    )(x)
    y = layers.BatchNormalization(name=f"{name_prefix}_bn1")(y)
    y = layers.Activation("relu", name=f"{name_prefix}_act1")(y)

    # Second conv
    y = layers.Conv2D(
        filters,
        (3, 3),
        strides=(1, 1),
        padding="same",
        name=f"{name_prefix}_conv2",
    )(y)
    y = layers.BatchNormalization(name=f"{name_prefix}_bn2")(y)

    # Shortcut connection (projection if shape changes)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            (1, 1),
            strides=(stride, stride),
            padding="same",
            name=f"{name_prefix}_proj",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    # Residual addition H(x) = F(x) + x
    out = layers.Add(name=f"{name_prefix}_add")([y, shortcut])
    out = layers.Activation("relu", name=f"{name_prefix}_out_act")(out)
    return out


def build_resnet_custom(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    dropout_rate: float = 0.2,
    model_name: str = "ResNet_Custom",
) -> models.Model:
    """
    Build custom lightweight ResNet (approx 18-layer style) suitable for CIFAR-10 / Flowers.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Initial Stem
    x = layers.Conv2D(32, (3, 3), padding="same", name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_act")(x)

    # Stage 1: 32 filters, 2 blocks
    x = residual_block(x, 32, stride=1, name_prefix="stage1_b1")
    x = residual_block(x, 32, stride=1, name_prefix="stage1_b2")

    # Stage 2: 64 filters, 2 blocks
    x = residual_block(x, 64, stride=2, name_prefix="stage2_b1")
    x = residual_block(x, 64, stride=1, name_prefix="stage2_b2")

    # Stage 3: 128 filters, 2 blocks
    x = residual_block(x, 128, stride=2, name_prefix="stage3_b1")
    x = residual_block(x, 128, stride=1, name_prefix="stage3_b2")

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="drop")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def build_resnet50_transfer(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 5,
    freeze_base: bool = True,
    fine_tune_at: Optional[int] = None,
    dropout_rate: float = 0.4,
    weights: str = "imagenet",
    model_name: str = "ResNet50_Transfer",
) -> models.Model:
    """
    Build ResNet-50 Transfer Learning model with pretrained ImageNet weights.
    """
    base_model = ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )

    if freeze_base:
        base_model.trainable = False
        if fine_tune_at is not None:
            base_model.trainable = True
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D(name="resnet_gap")(x)
    x = layers.Dense(256, activation="relu", name="resnet_dense")(x)
    x = layers.Dropout(dropout_rate, name="resnet_drop")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    m = build_resnet_custom()
    m.summary()
