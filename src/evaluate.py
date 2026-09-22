"""
Evaluation and Metrics Module for ISD-Assignment05.
Supports multi-class image classification and binary tabular classification evaluation.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)
import tensorflow as tf


def evaluate_multiclass(
    model: tf.keras.Model,
    data: Union[tf.data.Dataset, tuple],
    class_names: List[str],
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Evaluate multi-class image classifier.

    Args:
        model: Trained tf.keras.Model
        data: (X_test, y_test) or tf.data.Dataset
        class_names: List of class strings
        batch_size: Batch size for prediction

    Returns:
        Dictionary with accuracy, confusion_matrix, report_dict, report_text, y_true, y_pred
    """
    if isinstance(data, tuple):
        X_test, y_test = data
        y_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
    else:
        # tf.data.Dataset
        y_true_list = []
        y_probs_list = []
        for x_batch, y_batch in data:
            probs = model.predict(x_batch, verbose=0)
            y_probs_list.append(probs)
            if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
                y_true_list.append(np.argmax(y_batch.numpy(), axis=1))
            else:
                y_true_list.append(y_batch.numpy())
        y_probs = np.concatenate(y_probs_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)

    y_pred = np.argmax(y_probs, axis=1)

    labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rep_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    rep_text = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
        "classification_report_dict": rep_dict,
        "classification_report_text": rep_text,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }


def evaluate_binary(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate binary tabular classification (e.g. Diabetes Prediction).

    Returns:
        Dictionary with accuracy, roc_auc, pr_auc, confusion_matrix, f1, precision, recall
    """
    y_probs = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fpr, tpr, roc_thresh = roc_curve(y_test, y_probs)
    precision_pts, recall_pts, pr_thresh = precision_recall_curve(y_test, y_probs)

    rep_dict = classification_report(
        y_test, y_pred, labels=labels, target_names=["Non-Diabetic", "Diabetic"], output_dict=True, zero_division=0
    )
    rep_text = classification_report(
        y_test, y_pred, labels=labels, target_names=["Non-Diabetic", "Diabetic"], zero_division=0
    )

    return {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report_dict": rep_dict,
        "classification_report_text": rep_text,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_pts,
        "recall_curve": recall_pts,
    }


def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """Calculate total, trainable, and non-trainable parameter counts."""
    trainable = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable = int(np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
    return {
        "total": trainable + non_trainable,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }
