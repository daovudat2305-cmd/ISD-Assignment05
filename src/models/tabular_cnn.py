"""
Architectures for Tabular Data (Diabetes Prediction Dataset):
  1. 1D-CNN (Convolutional 1D Neural Network)
  2. Multi-Layer Perceptron (MLP) baseline for comparison.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_1d_cnn(
    num_features: int = 15,
    dropout_rate: float = 0.3,
    model_name: str = "1D_CNN_Tabular",
) -> models.Model:
    """
    Build 1D Convolutional Neural Network for tabular feature vectors.

    Input shape: (batch_size, num_features, 1)
    Convolutions slide across feature dimensions to discover local feature interactions.

    Args:
        num_features: Number of input features after one-hot/scaling.
        dropout_rate: Regularization rate.
        model_name: Model identifier name.

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=(num_features, 1), name="tabular_input")

    # Conv Block 1
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", name="conv1d_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="pool_1")(x)

    # Conv Block 2
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="pool_2")(x)

    # Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_act")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def build_mlp(
    num_features: int = 15,
    dropout_rate: float = 0.3,
    model_name: str = "MLP_Tabular",
) -> models.Model:
    """
    Build Multi-Layer Perceptron (MLP) baseline for tabular data.

    Input shape: (batch_size, num_features)
    """
    inputs = layers.Input(shape=(num_features,), name="tabular_input")

    x = layers.Dense(128, name="fc1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.Dense(64, name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    x = layers.Dense(32, name="fc3")(x)
    x = layers.Activation("relu", name="relu3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    m1 = build_1d_cnn(15)
    m1.summary()
    m2 = build_mlp(15)
    m2.summary()
