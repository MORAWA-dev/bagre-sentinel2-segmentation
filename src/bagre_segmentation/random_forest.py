from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from .config import json_safe, save_json, set_seed
from .data_contract import save_run_data_contract
from .metrics import SegmentationMeter


def _collect_training_pixels(images, labels, num_classes, ignore_index, quota, seed):
    rng = np.random.default_rng(seed)
    features = [[] for _ in range(num_classes)]
    targets = [[] for _ in range(num_classes)]
    totals = np.zeros(num_classes, dtype=int)
    order = rng.permutation(len(images))
    for index in tqdm(order, desc="Sampling RF training pixels"):
        image = np.asarray(images[index]).astype(np.float32) / 10000.0
        label = np.asarray(labels[index])
        for class_id in range(num_classes):
            remaining = quota - totals[class_id]
            if remaining <= 0:
                continue
            rows, cols = np.where(label == class_id)
            if len(rows) == 0:
                continue
            take = min(len(rows), remaining, 2048)
            selected = rng.choice(len(rows), size=take, replace=False)
            selected_rows, selected_cols = rows[selected], cols[selected]
            features[class_id].append(image[:, selected_rows, selected_cols].T)
            targets[class_id].append(np.full(take, class_id, dtype=np.uint8))
            totals[class_id] += take
        if np.all(totals >= quota):
            break
    if np.any(totals == 0):
        missing = np.flatnonzero(totals == 0).tolist()
        raise ValueError(f"No training pixels collected for class IDs: {missing}")
    x = np.concatenate([np.concatenate(parts) for parts in features]).astype(np.float32)
    y = np.concatenate([np.concatenate(parts) for parts in targets]).astype(np.int64)
    shuffled = rng.permutation(len(y))
    return x[shuffled], y[shuffled], totals

def train_random_forest(data_dir: str | Path, output_dir: str | Path, config: dict) -> dict:
    set_seed(int(config["seed"]))
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = config["classes"]["names"]
    num_classes = len(class_names)
    ignore_index = int(config["classes"]["ignore_index"])
    train_images = np.load(data_dir / "train" / "images.npy", mmap_mode="r")
    train_labels = np.load(data_dir / "train" / "labels.npy", mmap_mode="r")
    test_images = np.load(data_dir / "test" / "images.npy", mmap_mode="r")
    test_labels = np.load(data_dir / "test" / "labels.npy", mmap_mode="r")
    save_run_data_contract(data_dir, output_dir, len(test_labels))

    quota = int(config["random_forest"]["max_pixels_per_class"])
    x, y, totals = _collect_training_pixels(
        train_images, train_labels, num_classes, ignore_index, quota, int(config["seed"])
    )
    rf_config = config["random_forest"]
    classifier = RandomForestClassifier(
        n_estimators=int(rf_config["n_estimators"]),
        max_depth=int(rf_config["max_depth"]),
        min_samples_leaf=int(rf_config["min_samples_leaf"]),
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=0.7,
        n_jobs=int(rf_config["n_jobs"]),
        random_state=int(config["seed"]),
    )
    print(f"Training RF on {len(y):,} stratified pixels with four spectral bands.")
    classifier.fit(x, y)
    joblib.dump(classifier, output_dir / "random_forest.joblib", compress=3)

    predictions = np.lib.format.open_memmap(
        output_dir / "test_predictions.npy", mode="w+", dtype=np.uint8,
        shape=test_labels.shape
    )
    for index in tqdm(range(len(test_images)), desc="RF test predictions"):
        image = np.asarray(test_images[index]).astype(np.float32) / 10000.0
        label = np.asarray(test_labels[index])
        valid = label != ignore_index
        prediction = np.full(label.shape, ignore_index, dtype=np.uint8)
        pixels = image[:, valid].T
        prediction[valid] = classifier.predict(pixels).astype(np.uint8)
        predictions[index] = prediction
    predictions.flush()

    meter = SegmentationMeter(num_classes, ignore_index)
    meter.update(predictions, test_labels)
    metrics = meter.compute()
    metrics.update({
        "model": "Random Forest",
        "input_features": ["B2", "B3", "B4", "B8"],
        "training_pixels": len(y),
        "training_pixels_collected_by_class": totals.tolist(),
        "class_names": class_names,
        "evaluation_region": "spatially held-out test region",
    })
    save_json(json_safe(metrics), output_dir / "metrics.json")
    save_json(json_safe(config), output_dir / "run_config.json")
    return metrics
