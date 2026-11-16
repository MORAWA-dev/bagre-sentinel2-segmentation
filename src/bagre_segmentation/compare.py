from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data_contract import validate_comparison_contract
from .visualization import (
    plot_confusion_matrix,
    plot_prediction_examples,
    plot_training_curves,
)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_models(data_dir: str | Path, unet_dir: str | Path, rf_dir: str | Path,
                   output_dir: str | Path, seed: int = 42, num_examples: int = 4) -> None:
    data_dir, unet_dir, rf_dir, output_dir = map(Path, (data_dir, unet_dir, rf_dir, output_dir))
    validate_comparison_contract(data_dir, unet_dir, rf_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    class_names = metadata["class_names"]
    unet_metrics = _read_json(unet_dir / "metrics.json")
    rf_metrics = _read_json(rf_dir / "metrics.json")
    if (unet_metrics["support_pixels"] != rf_metrics["support_pixels"]
            or unet_metrics["reference_supported_classes"]
            != rf_metrics["reference_supported_classes"]):
        raise ValueError("RF and U-Net metrics do not share reference-supported classes.")

    rows = []
    for metrics in (unet_metrics, rf_metrics):
        row = {
            "model": metrics["model"],
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_iou_reference_supported": metrics["macro_iou_reference_supported"],
            "mean_iou": metrics["mean_iou"],
        }
        for index, name in enumerate(class_names):
            row[f"iou_{name}"] = metrics["per_class_iou"][index]
            row[f"f1_{name}"] = metrics["per_class_f1"][index]
            row[f"support_{name}"] = metrics["support_pixels"][index]
            row[f"predictions_{name}"] = metrics["prediction_pixels"][index]
        rows.append(row)
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_confusion_matrix(
        unet_metrics["confusion_matrix"], class_names, "U-Net test confusion matrix",
        output_dir / "unet_confusion_matrix.png"
    )
    plot_confusion_matrix(
        rf_metrics["confusion_matrix"], class_names, "Random Forest test confusion matrix",
        output_dir / "rf_confusion_matrix.png"
    )
    history_path = unet_dir / "history.csv"
    if history_path.exists():
        plot_training_curves(history_path, output_dir / "unet_training_curves.png")

    images = np.load(data_dir / "test" / "images.npy", mmap_mode="r")
    labels = np.load(data_dir / "test" / "labels.npy", mmap_mode="r")
    unet_predictions = np.load(unet_dir / "test_predictions.npy", mmap_mode="r")
    rf_predictions = np.load(rf_dir / "test_predictions.npy", mmap_mode="r")
    if unet_predictions.shape != labels.shape or rf_predictions.shape != labels.shape:
        raise ValueError("Prediction arrays must match the ordered test labels exactly.")
    rng = np.random.default_rng(seed)
    count = min(num_examples, len(images))
    indices = np.sort(rng.choice(len(images), size=count, replace=False))
    plot_prediction_examples(
        images, labels, unet_predictions, rf_predictions, indices, class_names,
        output_dir / "prediction_examples.png"
    )
    with open(output_dir / "figure_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({
            "example_selection": f"Uniform random test patches, seed={seed}, without replacement",
            "test_patch_indices": indices.tolist(),
            "interpretation": "Metrics measure agreement with ESA WorldCover, not field accuracy.",
        }, handle, indent=2)
