from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import json_safe, save_json, set_seed
from .data_contract import save_run_data_contract
from .datasets import SegmentationDataset
from .metrics import SegmentationMeter
from .model import DiceCrossEntropyLoss, build_model


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _make_loader(dataset, batch_size: int, shuffle: bool, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _run_epoch(model, loader, criterion, device, num_classes, ignore_index, optimizer=None,
               scaler=None, accumulation_steps: int = 1, use_amp: bool = False, description=""):
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter(num_classes, ignore_index)
    loss_total = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=description, leave=False)
        for batch_index, (images, targets) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                (scaler.scale(loss) / accumulation_steps).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            meter.update(logits.argmax(dim=1), targets)
            loss_total += float(loss.detach())
            progress.set_postfix(loss=f"{loss_total / (batch_index + 1):.4f}")
    metrics = meter.compute()
    metrics["loss"] = loss_total / max(1, len(loader))
    return metrics


def _write_history(history: list[dict], path: Path) -> None:
    fields = ["epoch", "learning_rate", "train_loss", "train_miou", "train_accuracy",
              "validation_loss", "validation_miou", "validation_accuracy", "minutes"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


@torch.no_grad()
def predict_dataset(model, loader, device, output_path: Path, use_amp: bool = False):
    dataset = loader.dataset
    predictions = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.uint8, shape=dataset.labels.shape
    )
    start = 0
    for images, _ in tqdm(loader, desc="Test predictions", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        batch = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        predictions[start:start + len(batch)] = batch
        start += len(batch)
    predictions.flush()
    return predictions


def train_unet(data_dir: str | Path, output_dir: str | Path, config: dict,
               resume: str | Path | None = None) -> dict:
    set_seed(int(config["seed"]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable. Use a Colab GPU for the real run.")
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    workers = int(config["training"]["num_workers"])
    batch_size = int(config["training"]["batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    ignore_index = int(config["classes"]["ignore_index"])
    num_classes = len(config["classes"]["names"])

    train_dataset = SegmentationDataset(data_dir, "train", augment=True)
    validation_dataset = SegmentationDataset(data_dir, "validation", augment=False)
    test_dataset = SegmentationDataset(data_dir, "test", augment=False)
    save_run_data_contract(data_dir, output_dir, len(test_dataset))
    train_loader = _make_loader(train_dataset, batch_size, True, workers)
    validation_loader = _make_loader(validation_dataset, batch_size, False, workers)
    test_loader = _make_loader(test_dataset, batch_size, False, workers)

    model = build_model(config).to(device)
    criterion = DiceCrossEntropyLoss(num_classes, ignore_index).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(config["training"]["scheduler_patience"]),
        factor=float(config["training"]["scheduler_factor"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_miou, stale_epochs = 0, float("-inf"), 0
    history = []
    history_path = output_dir / "history.csv"
    if resume:
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
        checkpoint = _load_checkpoint(Path(resume), device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint["best_miou"])
        stale_epochs = int(checkpoint["stale_epochs"])
        print(f"Resumed from epoch {start_epoch}.")

    final_epoch = start_epoch
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        started = time.time()
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, num_classes, ignore_index,
            optimizer, scaler, accumulation, use_amp, f"Train {epoch + 1}"
        )
        validation_metrics = _run_epoch(
            model, validation_loader, criterion, device, num_classes, ignore_index,
            use_amp=use_amp, description=f"Validate {epoch + 1}"
        )
        validation_miou = validation_metrics["mean_iou"]
        if validation_miou is None:
            raise ValueError("Validation mIoU is undefined. Check class support and labels.")
        scheduler.step(validation_miou)
        improved = validation_miou > best_miou
        if improved:
            best_miou = validation_miou
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_miou": train_metrics["mean_iou"],
            "train_accuracy": train_metrics["overall_accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_miou": validation_miou,
            "validation_accuracy": validation_metrics["overall_accuracy"],
            "minutes": (time.time() - started) / 60,
        })
        _write_history(history, history_path)
        checkpoint = {
            "epoch": epoch,
            "best_miou": best_miou,
            "stale_epochs": stale_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": config,
        }
        _atomic_torch_save(checkpoint, output_dir / "last.pt")
        if improved:
            _atomic_torch_save(checkpoint, output_dir / "best.pt")
        print(
            f"Epoch {epoch + 1}: train mIoU={train_metrics['mean_iou']:.4f}, "
            f"val mIoU={validation_miou:.4f}, best={best_miou:.4f}"
        )
        final_epoch = epoch
        if stale_epochs >= int(config["training"]["early_stopping_patience"]):
            print("Early stopping triggered.")
            break

    best = _load_checkpoint(output_dir / "best.pt", device)
    model.load_state_dict(best["model_state"])
    model.eval()
    predictions = predict_dataset(model, test_loader, device, output_dir / "test_predictions.npy", use_amp)
    meter = SegmentationMeter(num_classes, ignore_index)
    meter.update(predictions, test_dataset.labels)
    test_metrics = meter.compute()
    test_metrics.update({
        "model": "U-Net",
        "encoder": config["model"]["encoder"],
        "best_validation_miou": best_miou,
        "epochs_completed": final_epoch + 1,
        "class_names": config["classes"]["names"],
        "evaluation_region": "spatially held-out test region",
    })
    save_json(json_safe(test_metrics), output_dir / "metrics.json")
    save_json(json_safe(config), output_dir / "run_config.json")
    return test_metrics
