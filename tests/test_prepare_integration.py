import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

from bagre_segmentation.prepare import prepare_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_dataset_end_to_end_on_synthetic_rasters(tmp_path):
    with open(ROOT / "configs" / "bagre.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data"].update({
        "patch_size": 64,
        "train_stride": 64,
        "eval_stride": 64,
        "min_valid_fraction": 0.5,
    })
    config["data"]["split"].update({
        "train_fraction": 0.60,
        "validation_fraction": 0.20,
        "buffer_tiles": 0,
    })
    height, width = 64, 1280
    transform = from_origin(500000, 1300000, 10, 10)
    common = {"driver": "GTiff", "height": height, "width": width, "crs": "EPSG:32630",
              "transform": transform, "dtype": "float32"}
    image = np.random.default_rng(42).uniform(0.01, 0.8, size=(4, height, width)).astype(np.float32)
    # Cycle all five merged classes throughout the raster to test class support.
    source_classes = np.array([40, 10, 50, 60, 80], dtype=np.uint8)
    labels = np.tile(source_classes, (height, width // 5 + 1))[:, :width]
    image_path, label_path = tmp_path / "image.tif", tmp_path / "labels.tif"
    with rasterio.open(image_path, "w", count=4, nodata=-9999, **common) as target:
        target.write(image)
    label_options = {**common, "dtype": "uint8"}
    with rasterio.open(label_path, "w", count=1, nodata=0, **label_options) as target:
        target.write(labels, 1)

    metadata = prepare_dataset(image_path, label_path, tmp_path / "processed", config)
    assert metadata["split_counts"] == {"train": 12, "validation": 4, "test": 4}
    for split, count in metadata["split_counts"].items():
        images = np.load(tmp_path / "processed" / split / "images.npy", mmap_mode="r")
        labels_out = np.load(tmp_path / "processed" / split / "labels.npy", mmap_mode="r")
        assert images.shape == (count, 4, 64, 64)
        assert labels_out.shape == (count, 64, 64)
        assert set(np.unique(labels_out)).issubset({0, 1, 2, 3, 4, 255})
    assert len(metadata["normalization"]["mean"]) == 4
    assert (tmp_path / "processed" / "spatial_split.png").exists()
    assert (tmp_path / "processed" / "dataset_fingerprint.json").exists()
    manifest_path = tmp_path / "processed" / "test_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [tile["index"] for tile in manifest["tiles"]] == list(range(4))
    assert [(tile["row"], tile["col"]) for tile in manifest["tiles"]] == [
        (0, 1024), (0, 1088), (0, 1152), (0, 1216)
    ]
