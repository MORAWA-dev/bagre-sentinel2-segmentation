from __future__ import annotations

import numpy as np


class SegmentationMeter:
    """Accumulate a pixel confusion matrix without storing predictions."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction, target) -> None:
        if hasattr(prediction, "detach"):
            prediction = prediction.detach().cpu().numpy()
        if hasattr(target, "detach"):
            target = target.detach().cpu().numpy()
        prediction = np.asarray(prediction).reshape(-1)
        target = np.asarray(target).reshape(-1)
        valid = target != self.ignore_index
        prediction = prediction[valid]
        target = target[valid]
        if target.size == 0:
            return
        if prediction.min() < 0 or prediction.max() >= self.num_classes:
            raise ValueError("Prediction contains an out-of-range class ID.")
        if target.min() < 0 or target.max() >= self.num_classes:
            raise ValueError("Target contains an out-of-range class ID.")
        encoded = self.num_classes * target.astype(np.int64) + prediction.astype(np.int64)
        self.confusion += np.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        matrix = self.confusion.astype(np.float64)
        true_positive = np.diag(matrix)
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        union = support + predicted - true_positive
        iou = np.divide(
            true_positive,
            union,
            out=np.full(self.num_classes, np.nan),
            where=union > 0,
        )
        f1_denominator = support + predicted
        f1 = np.divide(
            2 * true_positive,
            f1_denominator,
            out=np.full(self.num_classes, np.nan),
            where=f1_denominator > 0,
        )
        total = matrix.sum()
        # Macro IoU uses reference support, never model-specific prediction support.
        # Thus an absent reference class is excluded for every model, even if one
        # model predicts it and therefore has a zero IoU for that class.
        supported_classes = np.flatnonzero(support > 0)
        macro_iou = (
            float(np.mean(iou[supported_classes])) if len(supported_classes) else None
        )
        return {
            "overall_accuracy": float(true_positive.sum() / total) if total else None,
            "mean_iou": macro_iou,
            "macro_iou_reference_supported": macro_iou,
            "reference_supported_classes": supported_classes.tolist(),
            "per_class_iou": iou.tolist(),
            "per_class_f1": f1.tolist(),
            "support_pixels": support.astype(np.int64).tolist(),
            "prediction_pixels": predicted.astype(np.int64).tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }
