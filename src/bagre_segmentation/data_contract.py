from __future__ import annotations

import json
from pathlib import Path

from .config import save_json


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_pair(directory: Path) -> tuple[dict, dict]:
    return (
        _read_json(directory / "dataset_fingerprint.json"),
        _read_json(directory / "test_manifest.json"),
    )


def save_run_data_contract(data_dir: str | Path, output_dir: str | Path,
                           expected_test_tiles: int) -> None:
    """Attach the exact prepared-data identity and test order to a model run."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    fingerprint, manifest = _read_pair(data_dir)
    if len(manifest["tiles"]) != expected_test_tiles:
        raise ValueError("Test tile count does not match the ordered test manifest.")
    save_json(fingerprint, output_dir / "dataset_fingerprint.json")
    save_json(manifest, output_dir / "test_manifest.json")


def validate_comparison_contract(data_dir: str | Path, unet_dir: str | Path,
                                 rf_dir: str | Path) -> None:
    """Refuse stale or differently ordered predictions before reading metrics."""
    directories = {
        "prepared data": Path(data_dir),
        "U-Net run": Path(unet_dir),
        "RF run": Path(rf_dir),
    }
    contracts = {name: _read_pair(path) for name, path in directories.items()}
    reference_fingerprint, reference_manifest = contracts["prepared data"]
    for run_name in ("U-Net run", "RF run"):
        fingerprint, manifest = contracts[run_name]
        if fingerprint.get("sha256") != reference_fingerprint.get("sha256"):
            raise ValueError(f"Dataset fingerprint mismatch for {run_name}; comparison refused.")
        if manifest != reference_manifest:
            raise ValueError(
                f"Ordered test tile manifest mismatch for {run_name}; comparison refused."
            )
