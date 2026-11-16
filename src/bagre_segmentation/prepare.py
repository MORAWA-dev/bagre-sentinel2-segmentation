from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .config import json_safe, save_json


def remap_worldcover(labels: np.ndarray, mapping: dict, ignore_index: int = 255) -> np.ndarray:
    """Map original ESA WorldCover values to contiguous IDs; ignore all others."""
    output = np.full(labels.shape, ignore_index, dtype=np.uint8)
    for source, target in mapping.items():
        output[labels == int(source)] = int(target)
    return output


def assign_split(center_x: float, width: int, data_config: dict) -> str | None:
    """Assign a patch centre to disjoint west/central/east regions with buffers."""
    split = data_config["split"]
    if split["method"] != "vertical_stripes":
        raise ValueError(f"Unsupported split method: {split['method']}")
    patch_size = data_config["patch_size"]
    buffer_pixels = int(split["buffer_tiles"]) * patch_size
    train_end = width * float(split["train_fraction"])
    validation_end = width * (
        float(split["train_fraction"]) + float(split["validation_fraction"])
    )
    if center_x < train_end - buffer_pixels:
        return "train"
    if train_end + buffer_pixels < center_x < validation_end - buffer_pixels:
        return "validation"
    if center_x > validation_end + buffer_pixels:
        return "test"
    return None


def _read_aligned_rasters(image_path: Path, label_path: Path, mapping: dict, ignore_index: int):
    with rasterio.open(image_path) as image_source, rasterio.open(label_path) as label_source:
        if image_source.count != 4:
            raise ValueError(f"Expected four Sentinel-2 bands, found {image_source.count}.")
        if label_source.count != 1:
            raise ValueError("WorldCover labels must contain exactly one band.")
        properties = ("crs", "transform", "width", "height")
        mismatches = [
            name for name in properties
            if getattr(image_source, name) != getattr(label_source, name)
        ]
        if mismatches:
            raise ValueError(
                "Image and labels are not on the same grid: " + ", ".join(mismatches) +
                ". Re-export both with the same region, CRS, and 10 m scale."
            )
        image = image_source.read().astype(np.float32)
        raw_labels = label_source.read(1)
        profile = {
            "crs": str(image_source.crs),
            "transform": list(image_source.transform),
            "width": image_source.width,
            "height": image_source.height,
            "image_nodata": image_source.nodata,
            "label_nodata": label_source.nodata,
        }
        image_nodata = image_source.nodata
        label_nodata = label_source.nodata

    invalid = ~np.isfinite(image)
    if image_nodata is not None:
        invalid |= image == image_nodata
    # Broad physical sanity bounds, not a cloud filter. GEE performs cloud masking.
    invalid |= (image < -0.05) | (image > 1.6)
    valid_spectra = ~invalid.any(axis=0)
    image = np.clip(image, 0.0, 1.0)
    labels = remap_worldcover(raw_labels, mapping, ignore_index)
    if label_nodata is not None:
        labels[raw_labels == label_nodata] = ignore_index
    valid = valid_spectra & (labels != ignore_index)
    return image, labels, valid, profile


def _origins(length: int, patch_size: int, stride: int, include_edge: bool = False) -> list[int]:
    if length < patch_size:
        return []
    values = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    # Overlapping edge patches are allowed for training only. Evaluation omits
    # the uncovered remainder rather than counting edge-source pixels twice.
    if include_edge and values[-1] != final:
        values.append(final)
    return values


def _candidate_origins(image, labels, valid, data_config, split_name):
    height, width = labels.shape
    patch = data_config["patch_size"]
    stride = data_config["train_stride"] if split_name == "train" else data_config["eval_stride"]
    minimum_valid = float(data_config["min_valid_fraction"])
    candidates = []
    include_edge = split_name == "train"
    for row in _origins(height, patch, stride, include_edge):
        for col in _origins(width, patch, stride, include_edge):
            assigned = assign_split(col + patch / 2, width, data_config)
            if assigned != split_name:
                continue
            tile_valid = valid[row:row + patch, col:col + patch]
            valid_fraction = float(tile_valid.mean())
            if valid_fraction < minimum_valid:
                continue
            candidates.append((row, col, valid_fraction))
    return candidates


def _channel_stats(images: np.ndarray, labels: np.ndarray, ignore_index: int, seed: int):
    rng = np.random.default_rng(seed)
    count = min(128, len(images))
    indices = rng.choice(len(images), size=count, replace=False)
    sums = np.zeros(images.shape[1], dtype=np.float64)
    square_sums = np.zeros(images.shape[1], dtype=np.float64)
    pixel_count = 0
    for index in indices:
        image = images[index].astype(np.float32) / 10000.0
        mask = labels[index] != ignore_index
        pixels = image[:, mask]
        sums += pixels.sum(axis=1)
        square_sums += np.square(pixels).sum(axis=1)
        pixel_count += pixels.shape[1]
    if pixel_count == 0:
        raise ValueError("Cannot compute normalization without valid training pixels.")
    mean = sums / pixel_count
    variance = np.maximum(square_sums / pixel_count - mean**2, 1e-12)
    return mean.tolist(), np.sqrt(variance).tolist()


def _save_summary(output_dir: Path, split_counts: dict, class_support: dict, class_names: list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(split_counts.keys(), split_counts.values(), color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Retained patches by spatial split")
    axes[0].set_ylabel("Patches")
    x = np.arange(len(class_names))
    width = 0.25
    for index, (split_name, support) in enumerate(class_support.items()):
        proportions = np.array(support, dtype=float)
        if proportions.sum() > 0:
            proportions /= proportions.sum()
        axes[1].bar(x + (index - 1) * width, proportions, width, label=split_name)
    axes[1].set_xticks(x, class_names, rotation=25, ha="right")
    axes[1].set_ylabel("Proportion of labelled pixels")
    axes[1].set_title("Class composition by split")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "data_summary.png", dpi=160)
    plt.close(fig)


def _save_split_map(output_dir: Path, width: int, data_config: dict) -> None:
    codes = {"train": 1, "validation": 2, "test": 3, None: 0}
    split_row = np.array([
        codes[assign_split(column + 0.5, width, data_config)]
        for column in range(width)
    ], dtype=np.uint8)[None, :]
    colors = ["#D9D9D9", "#4C78A8", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.imshow(split_row, aspect="auto", cmap=ListedColormap(colors), vmin=0, vmax=3)
    ax.set_yticks([])
    ax.set_xlabel("Raster column, west to east")
    ax.set_title("Geographic split and discarded buffer zones")
    ax.legend(
        handles=[Patch(color=colors[1], label="train"),
                 Patch(color=colors[2], label="validation"),
                 Patch(color=colors[3], label="test"),
                 Patch(color=colors[0], label="buffer")],
        loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _save_data_contract(output_dir: Path, config: dict, profile: dict, image: np.ndarray,
                        labels: np.ndarray, valid: np.ndarray, normalization: dict,
                        tile_manifests: dict) -> str:
    test_manifest = {
        "version": 1,
        "ordering": "row-major patch origins; array index equals manifest order",
        "edge_policy": "omit partial final patches; never append an overlapping edge patch",
        "source_height": int(labels.shape[0]),
        "source_width": int(labels.shape[1]),
        "patch_size": int(config["data"]["patch_size"]),
        "stride": int(config["data"]["eval_stride"]),
        "tiles": tile_manifests["test"],
    }
    save_json(test_manifest, output_dir / "test_manifest.json")
    fingerprint_document = {
        "version": 1,
        "algorithm": "sha256",
        "source_arrays": {
            "image": _array_sha256(image),
            "remapped_labels": _array_sha256(labels),
            "valid_mask": _array_sha256(valid),
        },
        "raster_profile": profile,
        "class_names": config["classes"]["names"],
        "worldcover_map": config["classes"]["worldcover_map"],
        "ignore_index": int(config["classes"]["ignore_index"]),
        "data_config": config["data"],
        "normalization": normalization,
        "tile_counts": {name: len(tiles) for name, tiles in tile_manifests.items()},
    }
    canonical_payload = json.dumps(
        json_safe({**fingerprint_document, "tile_manifests": tile_manifests}),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    fingerprint = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    save_json(
        {**fingerprint_document, "sha256": fingerprint},
        output_dir / "dataset_fingerprint.json",
    )
    return fingerprint


def prepare_dataset(image_path: str | Path, label_path: str | Path, output_dir: str | Path, config: dict) -> dict:
    image_path, label_path, output_dir = Path(image_path), Path(label_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ignore_index = int(config["classes"]["ignore_index"])
    mapping = {int(k): int(v) for k, v in config["classes"]["worldcover_map"].items()}
    class_names = config["classes"]["names"]
    patch = int(config["data"]["patch_size"])

    image, labels, valid, profile = _read_aligned_rasters(
        image_path, label_path, mapping, ignore_index
    )
    split_counts, class_support, metadata_rows = {}, {}, []
    tile_manifests = {}
    normalization = None

    for split_name in ("train", "validation", "test"):
        candidates = _candidate_origins(image, labels, valid, config["data"], split_name)
        if not candidates:
            raise ValueError(
                f"No {split_name} patches survived. Reduce the AOI buffer or minimum valid "
                "fraction, or enlarge the AOI. Do not switch to random patch splitting."
            )
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        shape = (len(candidates), 4, patch, patch)
        images = np.lib.format.open_memmap(
            split_dir / "images.npy", mode="w+", dtype=np.uint16, shape=shape
        )
        targets = np.lib.format.open_memmap(
            split_dir / "labels.npy", mode="w+", dtype=np.uint8,
            shape=(len(candidates), patch, patch)
        )
        support = np.zeros(len(class_names), dtype=np.int64)
        tile_manifests[split_name] = []
        for index, (row, col, valid_fraction) in enumerate(candidates):
            tile_manifests[split_name].append({
                "index": index, "row": row, "col": col,
                "valid_fraction": round(valid_fraction, 6),
            })
            tile_image = image[:, row:row + patch, col:col + patch]
            tile_labels = labels[row:row + patch, col:col + patch]
            tile_valid = valid[row:row + patch, col:col + patch]
            tile_labels[~tile_valid] = ignore_index
            images[index] = np.rint(tile_image * 10000).astype(np.uint16)
            targets[index] = tile_labels
            counts = np.bincount(
                tile_labels[tile_labels != ignore_index], minlength=len(class_names)
            )
            support += counts
            metadata_rows.append({
                "split": split_name,
                "index": index,
                "row": row,
                "col": col,
                "valid_fraction": f"{valid_fraction:.6f}",
                **{f"pixels_{name}": int(counts[i]) for i, name in enumerate(class_names)},
            })
        images.flush()
        targets.flush()
        split_counts[split_name] = len(candidates)
        class_support[split_name] = support.tolist()
        if split_name == "train":
            mean, std = _channel_stats(images, targets, ignore_index, int(config["seed"]))
            normalization = {"mean": mean, "std": std, "scale_factor": 10000}
        absent = [class_names[i] for i, count in enumerate(support) if count == 0]
        if absent:
            print(f"WARNING: {split_name} has no pixels for: {', '.join(absent)}")

    with open(output_dir / "tiles.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    metadata = {
        "source_image": str(image_path),
        "source_labels": str(label_path),
        "raster_profile": profile,
        "class_names": class_names,
        "ignore_index": ignore_index,
        "patch_size": patch,
        "split_counts": split_counts,
        "class_support_pixels": class_support,
        "normalization": normalization,
        "split_method": "Disjoint vertical train/validation/test regions with discarded buffer zones",
        "label_limitation": "ESA WorldCover labels are model products, not field ground truth.",
    }
    metadata["dataset_fingerprint"] = _save_data_contract(
        output_dir, config, profile, image, labels, valid, normalization, tile_manifests
    )
    _save_summary(output_dir, split_counts, class_support, class_names)
    _save_split_map(output_dir, labels.shape[1], config["data"])
    return metadata
