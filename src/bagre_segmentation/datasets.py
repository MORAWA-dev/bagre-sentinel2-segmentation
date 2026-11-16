from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str, augment: bool = False):
        self.split_dir = Path(data_dir) / split
        with open(Path(data_dir) / "metadata.json", "r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.images = np.load(self.split_dir / "images.npy", mmap_mode="r")
        self.labels = np.load(self.split_dir / "labels.npy", mmap_mode="r")
        self.ignore_index = int(self.metadata["ignore_index"])
        self.mean = np.asarray(self.metadata["normalization"]["mean"], dtype=np.float32)
        self.std = np.asarray(self.metadata["normalization"]["std"], dtype=np.float32)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]) if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        # uint16 storage reduces Drive usage; convert only the current patch.
        image = np.asarray(self.images[index]).transpose(1, 2, 0).astype(np.float32) / 10000.0
        label = np.asarray(self.labels[index]).copy()
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image, label = transformed["image"], transformed["mask"]
        image = (image - self.mean) / self.std
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(np.ascontiguousarray(label)).long(),
        )
