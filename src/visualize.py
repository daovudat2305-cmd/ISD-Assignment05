"""
Visualization Module for ISD-Assignment05.
Supports:
  1. Training history curves (Loss & Accuracy)
  2. Confusion matrix heatmaps
  3. ROC and Precision-Recall curves
  4. Model benchmark comparison bar charts
  5. Grad-CAM (Gradient-weighted Class Activation Mapping) explainability
  6. Dataset EDA sample image grids
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_training_history(
    history: Dict[str, list],
    title: str = "Training & Validation History",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Loss and Accuracy side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss subplot
    axes[0].plot(epochs, history["loss"], "o-", color="#e74c3c", label="Train Loss", linewidth=2, markersize=4)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "s--", color="#3498db", label="Val Loss", linewidth=2, markersize=4)
    axes[0].set_title(f"{title} — Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy subplot
    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"

    if acc_key in history:
        axes[1].plot(epochs, history[acc_key], "o-", color="#2ecc71", label="Train Acc", linewidth=2, markersize=4)
        if val_acc_key in history:
            axes[1].plot(epochs, history[val_acc_key], "s--", color="#9b59b6", label="Val Acc", linewidth=2, markersize=4)
        axes[1].set_title(f"{title} — Accuracy", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend(frameon=True)
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot annotated confusion matrix heatmap."""
    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * 0.9), max(6, len(class_names) * 0.8)))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#f0f0f0",
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_pr_curves(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    precision: np.ndarray,
    recall: np.ndarray,
    title: str = "ROC & Precision-Recall Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curve and Precision-Recall curve side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_title(f"{title} — ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # PR curve
    axes[1].plot(recall, precision, color="#8e44ad", lw=2, label="Precision-Recall")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title} — Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_model_comparison_bar(
    model_names: List[str],
    accuracies: List[float],
    f1_scores: Optional[List[float]] = None,
    dataset_title: str = "CIFAR-10",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing multiple models on a given dataset."""
    x = np.arange(len(model_names))
    width = 0.35 if f1_scores else 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - (width / 2 if f1_scores else 0), accuracies, width, label="Accuracy", color="#3498db")

    if f1_scores:
        rects2 = ax.bar(x + width / 2, f1_scores, width, label="Macro F1", color="#2ecc71")

    ax.set_ylabel("Score (0 - 1)", fontweight="bold")
    ax.set_title(f"Model Performance Comparison on {dataset_title}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontweight="bold")
    ax.set_ylim([0, 1.08])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    if f1_scores:
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for an input image.

    Args:
        img_array: (1, H, W, C) preprocessed image tensor.
        model: Model containing convolutional layers.
        last_conv_layer_name: Target conv layer name.
        pred_index: Class index (if None, uses highest probability class).

    Returns:
        Heatmap array normalized between 0 and 1.
    """
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of top predicted class w.r.t feature map activations
    grads = tape.gradient(class_channel, conv_outputs)

    # Channel-wise mean pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU on heatmap to consider only positive contributions
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay heatmap onto RGB image without external opencv dependency."""
    from PIL import Image
    h, w = original_img.shape[:2]
    # Resize heatmap using PIL
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize((w, h), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    heatmap_colored = plt.get_cmap(colormap)(heatmap_resized)[:, :, :3]
    superimposed_img = heatmap_colored * alpha + original_img * (1.0 - alpha)
    return np.clip(superimposed_img, 0.0, 1.0)
