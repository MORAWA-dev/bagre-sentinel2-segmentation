from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bagre_segmentation.metrics import SegmentationMeter
from bagre_segmentation.model import build_model
from bagre_segmentation.prepare import assign_split, remap_worldcover

ROOT = Path(__file__).resolve().parents[1]


def load_test_config():
    with open(ROOT / "configs" / "bagre.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_worldcover_remap_and_ignore():
    labels = np.array([[40, 10, 50], [60, 80, 70]], dtype=np.uint8)
    config = load_test_config()
    mapped = remap_worldcover(labels, config["classes"]["worldcover_map"], 255)
    assert mapped.tolist() == [[0, 1, 2], [3, 4, 255]]
    assert mapped.dtype == np.uint8


def test_metrics_match_manual_confusion_matrix():
    meter = SegmentationMeter(num_classes=2, ignore_index=255)
    meter.update(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]))
    result = meter.compute()
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["overall_accuracy"] == pytest.approx(0.75)
    assert result["per_class_iou"][0] == pytest.approx(0.5)
    assert result["per_class_iou"][1] == pytest.approx(2 / 3)
    assert result["mean_iou"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_ignore_label_does_not_change_metrics():
    meter = SegmentationMeter(num_classes=2, ignore_index=255)
    meter.update(np.array([0, 1, 0]), np.array([0, 1, 255]))
    assert meter.compute()["overall_accuracy"] == 1.0
    assert meter.confusion.sum() == 2


def test_spatial_regions_and_buffers_are_disjoint():
    config = load_test_config()["data"]
    width = 5120
    train_end = width * config["split"]["train_fraction"]
    validation_end = width * (
        config["split"]["train_fraction"] + config["split"]["validation_fraction"]
    )
    assert assign_split(100, width, config) == "train"
    assert assign_split(train_end, width, config) is None
    assert assign_split((train_end + validation_end) / 2, width, config) == "validation"
    assert assign_split(validation_end, width, config) is None
    assert assign_split(width - 10, width, config) == "test"


def test_four_band_unet_forward_pass_without_downloading_weights():
    config = load_test_config()
    config["model"]["encoder_weights"] = None
    model = build_model(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 4, 64, 64))
    assert output.shape == (2, 5, 64, 64)
