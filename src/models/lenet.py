"""
LeNet-5 Architecture (Yann LeCun et al., 1998).
Includes modern enhancements (ReLU activations, optional Batch Normalization, and Dropout).
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lenet5(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    use_modern_activations: bool = True,
    dropout_rate: float = 0.2,
    model_name: str = "LeNet5",
) -> models.Model:
    """
    Construct LeNet-5 model adapted for RGB / single channel images.

    Architecture (LeCun et al. 1998 adaptation):
      - C1: Conv2D (6 filters, 5x5, stride 1, padding 'same' or 'valid')
      - S2: AveragePooling2D (2x2, stride 2) [or MaxPooling2D in modern mode]
      - C3: Conv2D (16 filters, 5x5, stride 1)
      - S4: AveragePooling2D (2x2, stride 2) [or MaxPooling2D]
      - C5: Flatten -> Dense (120 units)
      - F6: Dense (84 units)
      - Output: Dense (num_classes, softmax)

    Args:
        input_shape: Image dimensions (H, W, C).
        num_classes: Number of output categories.
        use_modern_activations: If True, uses ReLU + BatchNorm; if False, uses classical Tanh.
        dropout_rate: Dropout rate before classification.
        model_name: Name of the model.

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    act_fn = "relu" if use_modern_activations else "tanh"

    # C1 Layer
    x = layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding="same", name="C1_conv")(inputs)
    if use_modern_activations:
        x = layers.BatchNormalization(name="C1_bn")(x)
    x = layers.Activation(act_fn, name="C1_act")(x)

    # S2 Layer
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S2_pool")(x)

    # C3 Layer
    x = layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding="valid", name="C3_conv")(x)
    if use_modern_activations:
        x = layers.BatchNormalization(name="C3_bn")(x)
    x = layers.Activation(act_fn, name="C3_act")(x)

    # S4 Layer
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S4_pool")(x)

    # Flatten & C5 Dense
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(120, name="C5_dense")(x)
    if use_modern_activations:
        x = layers.BatchNormalization(name="C5_bn")(x)
    x = layers.Activation(act_fn, name="C5_act")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="drop1")(x)

    # F6 Layer
    x = layers.Dense(84, name="F6_dense")(x)
    if use_modern_activations:
        x = layers.BatchNormalization(name="F6_bn")(x)
    x = layers.Activation(act_fn, name="F6_act")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="drop2")(x)

    # Output Layer
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=activation, name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    model = build_lenet5()
    model.summary()
