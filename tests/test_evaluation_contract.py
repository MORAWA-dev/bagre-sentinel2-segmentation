import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bagre_segmentation.compare import compare_models
from bagre_segmentation.data_contract import save_run_data_contract
from bagre_segmentation.metrics import SegmentationMeter
from bagre_segmentation.prepare import _origins


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_contract(directory: Path, fingerprint: dict, manifest: dict) -> None:
    _write_json(directory / "dataset_fingerprint.json", fingerprint)
    _write_json(directory / "test_manifest.json", manifest)


def test_evaluation_origins_never_overlap_at_partial_edge():
    origins = _origins(600, patch_size=256, stride=256)
    assert origins == [0, 256]
    coverage = np.zeros(600, dtype=np.int32)
    for origin in origins:
        coverage[origin:origin + 256] += 1
    assert coverage.max() == 1
    # Training may still include the final overlapping edge patch.
    assert _origins(600, 256, 256, include_edge=True) == [0, 256, 344]


def test_absent_reference_class_is_excluded_from_macro_iou():
    meter = SegmentationMeter(num_classes=3)
    meter.update(
        prediction=np.array([0, 1, 2, 2]),
        target=np.array([0, 1, 0, 0]),
    )
    result = meter.compute()
    assert result["support_pixels"] == [3, 1, 0]
    assert result["prediction_pixels"] == [1, 1, 2]
    assert result["reference_supported_classes"] == [0, 1]
    assert result["per_class_iou"][2] == 0.0
    expected = (result["per_class_iou"][0] + result["per_class_iou"][1]) / 2
    assert result["macro_iou_reference_supported"] == pytest.approx(expected)
    assert result["mean_iou"] == pytest.approx(expected)


def test_run_saves_dataset_fingerprint_and_ordered_test_manifest(tmp_path):
    processed = tmp_path / "processed"
    fingerprint = {"algorithm": "sha256", "sha256": "abc123"}
    manifest = {"tiles": [{"index": 0, "row": 10, "col": 20}]}
    _write_contract(processed, fingerprint, manifest)
    run = tmp_path / "run"
    run.mkdir()
    save_run_data_contract(processed, run, expected_test_tiles=1)
    assert json.loads((run / "dataset_fingerprint.json").read_text()) == fingerprint
    assert json.loads((run / "test_manifest.json").read_text()) == manifest


def test_comparison_refuses_equal_shaped_predictions_with_different_manifests(tmp_path):
    processed, unet, rf = (tmp_path / name for name in ("processed", "unet", "rf"))
    for directory in (processed, unet, rf):
        (directory / "test").mkdir(parents=True) if directory == processed else directory.mkdir()
    fingerprint = {"algorithm": "sha256", "sha256": "same-data"}
    manifest = {"version": 1, "tiles": [{"index": 0, "row": 0, "col": 0}]}
    _write_contract(processed, fingerprint, manifest)
    _write_contract(unet, fingerprint, manifest)
    different_manifest = copy.deepcopy(manifest)
    different_manifest["tiles"][0]["row"] = 32
    _write_contract(rf, fingerprint, different_manifest)

    shape = (1, 32, 32)
    labels = np.zeros(shape, dtype=np.uint8)
    unet_predictions = np.zeros(shape, dtype=np.uint8)
    rf_predictions = np.zeros(shape, dtype=np.uint8)
    assert unet_predictions.shape == rf_predictions.shape == labels.shape
    np.save(processed / "test" / "labels.npy", labels)
    np.save(unet / "test_predictions.npy", unet_predictions)
    np.save(rf / "test_predictions.npy", rf_predictions)

    with pytest.raises(ValueError, match="manifest mismatch"):
        compare_models(processed, unet, rf, tmp_path / "comparison", num_examples=1)
    assert not (tmp_path / "comparison").exists()


def test_comparison_refuses_dataset_fingerprint_mismatch(tmp_path):
    processed, unet, rf = (tmp_path / name for name in ("processed", "unet", "rf"))
    manifest = {"version": 1, "tiles": [{"index": 0, "row": 0, "col": 0}]}
    _write_contract(processed, {"sha256": "dataset-a"}, manifest)
    _write_contract(unet, {"sha256": "dataset-b"}, manifest)
    _write_contract(rf, {"sha256": "dataset-a"}, manifest)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        compare_models(processed, unet, rf, tmp_path / "comparison")
