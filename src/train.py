"""
Training Pipeline for ISD-Assignment05.
Encapsulates compilation, callbacks, training execution, timing, and history logging.
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import tensorflow as tf


def get_default_callbacks(
    checkpoint_path: Optional[str] = None,
    patience_es: int = 10,
    patience_lr: int = 5,
    min_lr: float = 1e-6,
) -> list:
    """
    Generate standard training callbacks:
      - EarlyStopping to prevent overfitting.
      - ReduceLROnPlateau to fine-tune learning rate when progress stalls.
      - ModelCheckpoint (optional) to save best weights.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
        ),
    ]

    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

    return callbacks


def compile_and_train(
    model: tf.keras.Model,
    train_data: Union[tf.data.Dataset, tuple],
    val_data: Optional[Union[tf.data.Dataset, tuple]] = None,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adam",
    loss: Optional[str] = None,
    metrics: Optional[list] = None,
    checkpoint_path: Optional[str] = None,
    patience_es: int = 10,
    patience_lr: int = 5,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Compile and train a Keras model with standardized evaluation tracking.

    Args:
        model: tf.keras.Model
        train_data: (X_train, y_train) or tf.data.Dataset
        val_data: (X_val, y_val) or tf.data.Dataset
        epochs: Max epochs
        batch_size: Batch size (for numpy arrays)
        learning_rate: Initial learning rate
        optimizer_name: 'adam' or 'sgd'
        loss: Loss function string (auto-inferred if None)
        metrics: List of metrics (defaults to ['accuracy'])
        checkpoint_path: Optional path to save best model (.keras)
        patience_es: Early stopping patience
        patience_lr: Reduce LR patience
        verbose: Verbosity level

    Returns:
        Dict containing: 'model', 'history', 'train_time_sec', 'epochs_trained'
    """
    # Optimizer selection
    if optimizer_name.lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = optimizer_name

    # Auto-infer loss if None
    if loss is None:
        last_layer = model.layers[-1]
        out_units = getattr(last_layer, "units", 1)
        if out_units == 1:
            loss = "binary_crossentropy"
        else:
            loss = "categorical_crossentropy"

    if metrics is None:
        metrics = ["accuracy"]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    callbacks = get_default_callbacks(
        checkpoint_path=checkpoint_path,
        patience_es=patience_es,
        patience_lr=patience_lr,
    )

    start_time = time.time()

    if isinstance(train_data, tuple):
        X_train, y_train = train_data
        validation_data = val_data
        history = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        # tf.data.Dataset
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

    train_time = time.time() - start_time
    epochs_trained = len(history.history["loss"])

    return {
        "model": model,
        "history": history.history,
        "train_time_sec": train_time,
        "epochs_trained": epochs_trained,
    }


def save_training_history(history: dict, filepath: str) -> None:
    """Save history dict to JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy floats if any
    clean_history = {}
    for k, v in history.items():
        clean_history[k] = [float(x) for x in v]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean_history, f, indent=2)
