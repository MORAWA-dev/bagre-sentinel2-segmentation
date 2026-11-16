from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    names = config["classes"]["names"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Class names must be non-empty and unique.")
    mapping = {int(k): int(v) for k, v in config["classes"]["worldcover_map"].items()}
    if set(mapping.values()) != set(range(len(names))):
        raise ValueError("Merged class IDs must be contiguous from 0 to N-1.")
    if config["data"]["patch_size"] < 32:
        raise ValueError("Patch size must be at least 32 pixels.")
    if config["data"]["eval_stride"] < config["data"]["patch_size"]:
        raise ValueError("Evaluation stride must be >= patch size to prevent duplicate pixels.")
    split = config["data"]["split"]
    train = float(split["train_fraction"])
    validation = float(split["validation_fraction"])
    if not 0 < train < train + validation < 1:
        raise ValueError("Spatial split fractions must leave a non-empty test region.")
    if config["model"]["in_channels"] != 4:
        raise ValueError("The MVP expects B2, B3, B4, and B8 (four channels).")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value
