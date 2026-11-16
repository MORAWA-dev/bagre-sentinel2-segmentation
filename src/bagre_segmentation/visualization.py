from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

CLASS_COLORS = ["#E1C699", "#2E8B57", "#D73027", "#BDB76B", "#2166AC"]


def class_cmap(num_classes: int):
    if num_classes > len(CLASS_COLORS):
        raise ValueError("Add colors for additional classes.")
    return ListedColormap(CLASS_COLORS[:num_classes])


def rgb_composite(image: np.ndarray) -> np.ndarray:
    """Convert a CHW B2/B3/B4/B8 uint16 patch to a robust RGB display."""
    rgb = image[[2, 1, 0]].astype(np.float32).transpose(1, 2, 0) / 10000.0
    valid = np.isfinite(rgb)
    if valid.any():
        low, high = np.percentile(rgb[valid], [2, 98])
        if high > low:
            rgb = (rgb - low) / (high - low)
    return np.clip(rgb, 0, 1)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], title: str, path: Path):
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("WorldCover reference class")
    ax.set_title(title)
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            value = normalized[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.55 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_prediction_examples(images, labels, unet_predictions, rf_predictions,
                             indices, class_names, output_path: Path):
    rows = len(indices)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), squeeze=False)
    cmap = class_cmap(len(class_names))
    titles = ["Sentinel-2 RGB", "WorldCover reference", "U-Net prediction", "RF prediction"]
    for row, index in enumerate(indices):
        arrays = [None, labels[index], unet_predictions[index], rf_predictions[index]]
        for col in range(4):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(rgb_composite(images[index]))
            else:
                ax.imshow(arrays[col], cmap=cmap, vmin=0, vmax=len(class_names) - 1,
                          interpolation="nearest")
            ax.set_title(f"{titles[col]}\ntest patch {index}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history_path: Path, output_path: Path):
    with open(history_path, "r", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    train = [float(row["train_miou"]) for row in history]
    validation = [float(row["validation_miou"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    validation_loss = [float(row["validation_loss"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train, label="train")
    axes[0].plot(epochs, validation, label="validation")
    axes[0].set(title="U-Net mIoU", xlabel="Epoch", ylabel="mIoU")
    axes[0].legend()
    axes[1].plot(epochs, train_loss, label="train")
    axes[1].plot(epochs, validation_loss, label="validation")
    axes[1].set(title="U-Net loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
