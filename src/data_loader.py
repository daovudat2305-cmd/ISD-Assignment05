"""
Data Loader Module for ISD-Assignment05.
Supports:
  1. CIFAR-10 (via Keras built-in datasets)
  2. Flowers Recognition (image directory loading with split & augmentation)
  3. Fruits and Vegetable (36 classes from Kaggle)
  4. Diabetes Prediction Dataset (Tabular preprocessing & 1D tensor formatting)
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent

if os.path.exists("/kaggle/input"):
    DATA_DIR = Path("/kaggle/input")
else:
    DATA_DIR = BASE_DIR / "data"

# =====================================================================
# 1. CIFAR-10 Dataset Loader
# =====================================================================

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_cifar10(
    val_size: float = 0.1,
    normalize: bool = True,
    one_hot: bool = True,
    random_state: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], List[str]]:
    """
    Load CIFAR-10 dataset with train/validation/test splits.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names
    """
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if normalize:
        X_train_full = X_train_full.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

    y_train_full = y_train_full.squeeze()
    y_test = y_test.squeeze()

    # Split train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    if one_hot:
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_val = tf.keras.utils.to_categorical(y_val, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), CIFAR10_CLASSES


# =====================================================================
# 2. Flowers Recognition Dataset Loader
# =====================================================================

def find_flowers_dir() -> Path:
    """Auto-locate flowers dataset directory."""
    candidates = [
        DATA_DIR / "flowers" / "flowers_recognition" / "flowers",
        DATA_DIR / "flowers" / "flowers",
        DATA_DIR / "flowers",
    ]
    
    if os.path.exists("/kaggle/input"):
        # Tìm kiếm đệ quy thư mục 'rose' trên toàn bộ Kaggle Input
        for p in Path('/kaggle/input').rglob('rose'):
            if p.is_dir():
                return p.parent
                
    for p in candidates:
        if p.exists() and (p / "rose").exists():
            return p
    raise FileNotFoundError("Could not find flowers dataset directory with subfolders (rose, daisy, etc.)")


def load_flowers(
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Load Flowers dataset using image_dataset_from_directory.

    Returns:
        train_ds, val_ds, class_names
    """
    flowers_dir = find_flowers_dir()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(flowers_dir),
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(flowers_dir),
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,  # Must be True with seed to split all 5 classes uniformly
    )

    class_names = train_ds.class_names

    # Normalize to [0, 1] and configure prefetching
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# =====================================================================
# 3. Fruits and Vegetable Dataset Loader
# =====================================================================

def find_fruits_vegetable_dir() -> Optional[Path]:
    """Auto-locate fruits and vegetables dataset directory."""
    candidates = [
        DATA_DIR / "cifar10" / "fruits_vegetable",
        DATA_DIR / "fruits_vegetable",
    ]
    
    if os.path.exists("/kaggle/input"):
        # Tìm kiếm đệ quy thư mục train của fruits_vegetable
        for p in Path('/kaggle/input').rglob('train'):
            if p.is_dir() and (p.parent / "validation").exists():
                return p.parent
                
    for p in candidates:
        if p.exists() and (p / "train").exists():
            return p
    return None


def load_fruits_vegetable(
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
) -> Optional[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]]:
    """
    Load Fruits and Vegetable 36-class dataset if available.
    """
    fv_dir = find_fruits_vegetable_dir()
    if fv_dir is None:
        return None

    train_dir = fv_dir / "train"
    val_dir = fv_dir / "validation"
    test_dir = fv_dir / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(val_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    norm_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (norm_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (norm_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# =====================================================================
# 4. Diabetes Tabular Dataset Loader & Preprocessor
# =====================================================================

def find_diabetes_file() -> Path:
    """Auto-locate diabetes dataset CSV."""
    candidates = [
        DATA_DIR / "diabetes" / "diabetes_prediction_dataset.csv",
        DATA_DIR / "diabetes_prediction_dataset.csv",
    ]
    
    if os.path.exists("/kaggle/input"):
        # Tìm kiếm đệ quy file csv
        for p in Path('/kaggle/input').rglob('diabetes_prediction_dataset.csv'):
            if p.is_file():
                return p
                
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find diabetes_prediction_dataset.csv")


def load_diabetes(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load and preprocess Diabetes Tabular dataset.

    Pipeline:
      - Categorical columns: One-Hot Encoded (gender, smoking_history)
      - Numerical columns: StandardScaled (age, bmi, HbA1c_level, blood_glucose_level, hypertension, heart_disease)
      - Target: 'diabetes' (binary 0 or 1)
      - Outputs 2D matrices for MLP and 3D tensors (N, features, 1) for 1D-CNN.

    Returns dictionary with:
      - 'X_train_2d', 'y_train', 'X_val_2d', 'y_val', 'X_test_2d', 'y_test'
      - 'X_train_1d', 'X_val_1d', 'X_test_1d'
      - 'feature_names', 'preprocessor', 'raw_df'
    """
    csv_path = find_diabetes_file()
    df = pd.read_csv(csv_path)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    # Filter invalid/rare values if present
    df = df[df["gender"] != "Other"].copy()

    X_raw = df.drop(columns=["diabetes"])
    y = df["diabetes"].values.astype("float32")

    categorical_cols = ["gender", "smoking_history"]
    numerical_cols = [col for col in X_raw.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ]
    )

    # First split: train+val and test
    X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train and val
    val_relative_size = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val_raw, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
    )

    # Fit preprocessor strictly on training data
    X_train_2d = preprocessor.fit_transform(X_train_raw).astype("float32")
    X_val_2d = preprocessor.transform(X_val_raw).astype("float32")
    X_test_2d = preprocessor.transform(X_test_raw).astype("float32")

    # Reshape for 1D-CNN: (batch, num_features, 1)
    X_train_1d = np.expand_dims(X_train_2d, axis=-1)
    X_val_1d = np.expand_dims(X_val_2d, axis=-1)
    X_test_1d = np.expand_dims(X_test_2d, axis=-1)

    # Extract transformed feature names
    cat_feature_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    feature_names = numerical_cols + cat_feature_names

    return {
        "X_train_2d": X_train_2d,
        "y_train": y_train,
        "X_val_2d": X_val_2d,
        "y_val": y_val,
        "X_test_2d": X_test_2d,
        "y_test": y_test,
        "X_train_1d": X_train_1d,
        "X_val_1d": X_val_1d,
        "X_test_1d": X_test_1d,
        "num_features": X_train_2d.shape[1],
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "raw_df": df,
    }


if __name__ == "__main__":
    print("Testing data loader...")
    d_data = load_diabetes(max_samples=2000)
    print("Diabetes features:", d_data["num_features"])
    print("1D shape:", d_data["X_train_1d"].shape)
