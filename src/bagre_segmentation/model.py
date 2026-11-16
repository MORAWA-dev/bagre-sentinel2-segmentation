from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn


def build_model(config: dict) -> nn.Module:
    model_config = config["model"]
    if model_config["architecture"].lower() != "unet":
        raise ValueError("The MVP supports U-Net only.")
    return smp.Unet(
        encoder_name=model_config["encoder"],
        encoder_weights=model_config["encoder_weights"],
        in_channels=int(model_config["in_channels"]),
        classes=len(config["classes"]["names"]),
        activation=None,
    )


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(self, logits, target):
        return 0.5 * self.cross_entropy(logits, target) + 0.5 * self.dice(logits, target)
