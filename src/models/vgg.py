"""
VGG Architecture (Simonyan & Zisserman, 2014).
Provides both:
  1. Custom VGG-like CNN (trainable from scratch, optimized for small/medium datasets).
  2. VGG-16 Pretrained Transfer Learning model via keras.applications.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16


def build_vgg_custom(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    dropout_rate: float = 0.4,
    model_name: str = "VGG_Custom",
) -> models.Model:
    """
    Build custom VGG-like network with blocks of small 3x3 filters.

    Structure:
      - Block 1: [Conv2D(64, 3x3) * 2] -> MaxPool(2x2)
      - Block 2: [Conv2D(128, 3x3) * 2] -> MaxPool(2x2)
      - Block 3: [Conv2D(256, 3x3) * 3] -> MaxPool(2x2)
      - GlobalAveragePooling2D / Dense -> Dropout -> Dense(num_classes)
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding="same", name="block1_conv1")(inputs)
    x = layers.BatchNormalization(name="block1_bn1")(x)
    x = layers.Activation("relu", name="block1_act1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="block1_conv2")(x)
    x = layers.BatchNormalization(name="block1_bn2")(x)
    x = layers.Activation("relu", name="block1_act2")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="block1_drop")(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding="same", name="block2_conv1")(x)
    x = layers.BatchNormalization(name="block2_bn1")(x)
    x = layers.Activation("relu", name="block2_act1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="block2_conv2")(x)
    x = layers.BatchNormalization(name="block2_bn2")(x)
    x = layers.Activation("relu", name="block2_act2")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    x = layers.Dropout(dropout_rate * 0.75, name="block2_drop")(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding="same", name="block3_conv1")(x)
    x = layers.BatchNormalization(name="block3_bn1")(x)
    x = layers.Activation("relu", name="block3_act1")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="block3_conv2")(x)
    x = layers.BatchNormalization(name="block3_bn2")(x)
    x = layers.Activation("relu", name="block3_act2")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    x = layers.Dropout(dropout_rate, name="block3_drop")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(512, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_act")(x)
    x = layers.Dropout(dropout_rate, name="fc1_drop")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def build_vgg16_transfer(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 5,
    freeze_base: bool = True,
    fine_tune_at: Optional[int] = None,
    dropout_rate: float = 0.5,
    weights: str = "imagenet",
    model_name: str = "VGG16_Transfer",
) -> models.Model:
    """
    Build VGG-16 Transfer Learning model with pretrained ImageNet weights.

    Args:
        input_shape: Image input dimensions (minimum 32x32, recommend >= 128x128).
        num_classes: Number of classes.
        freeze_base: Whether to freeze backbone convolutional layers.
        fine_tune_at: If set, unfreezes layers starting from this index.
        dropout_rate: Dropout rate for top classifier.
        weights: Pretrained weights ('imagenet' or None).
        model_name: Model identifier name.

    Returns:
        tf.keras.Model
    """
    base_model = VGG16(
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
    # Preprocess input if needed for VGG
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D(name="vgg_gap")(x)
    x = layers.Dense(256, activation="relu", name="vgg_dense1")(x)
    x = layers.Dropout(dropout_rate, name="vgg_drop")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    m1 = build_vgg_custom()
    m1.summary()
