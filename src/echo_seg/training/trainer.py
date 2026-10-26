"""Training loop with early stopping, logging, and reproducible checkpointing."""

import json
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader

from echo_seg.data import CamusDataset, build_train_transform, build_val_transform
from echo_seg.models import build_model
from echo_seg.utils import get_logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _CombinedLoss(nn.Module):
    def __init__(self, dice_w: float = 0.5, ce_w: float = 0.5) -> None:
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce = nn.CrossEntropyLoss()
        self.dice_w = dice_w
        self.ce_w = ce_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self.dice(logits, targets) + self.ce_w * self.ce(logits, targets)


def _dice_score_batch(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Mean Dice over classes 1..num_classes-1 (skip background) for a batch."""
    preds = logits.argmax(dim=1)
    scores = []
    for c in range(1, num_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        scores.append((2 * inter / (denom + 1e-6)).item())
    return float(np.mean(scores))


def train(cfg: dict, run_name: str = "baseline") -> Path:
    """Train a U-Net and return the checkpoint path.

    Args:
        cfg: loaded default.yaml contents
        run_name: subdirectory name under models_dir
    """
    seed = cfg["training"]["seed"]
    _set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log_dir = cfg["output"]["logs_dir"]
    logger = get_logger(f"train.{run_name}", log_dir=log_dir)
    logger.info("Run: %s | device: %s | seed: %d", run_name, device, seed)

    img_size = tuple(cfg["data"]["image_size"])
    aug_cfg = cfg["augmentation"]
    train_ds = CamusDataset(
        cfg["output"]["manifest_path"],
        "train",
        transform=build_train_transform(
            image_size=img_size,
            rotation_limit=aug_cfg["rotation_limit"],
            shift_limit=aug_cfg["shift_limit"],
            scale_limit=aug_cfg["scale_limit"],
            gamma_limit=tuple(aug_cfg["gamma_limit"]),
            brightness_limit=aug_cfg["brightness_limit"],
            noise_var_limit=tuple(aug_cfg["noise_var_limit"]),
        ),
    )
    val_ds = CamusDataset(
        cfg["output"]["manifest_path"],
        "val",
        transform=build_val_transform(img_size),
    )
    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    pin = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=pin,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=pin,
    )

    model_cfg = cfg["model"]
    model = build_model(
        encoder=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
    ).to(device)
    logger.info("Model: Unet/%s (%s weights)", model_cfg["encoder"], model_cfg["encoder_weights"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )
    criterion = _CombinedLoss(
        dice_w=cfg["training"]["dice_weight"],
        ce_w=cfg["training"]["ce_weight"],
    )
    num_classes = cfg["model"]["num_classes"]
    patience = cfg["training"]["patience"]

    ckpt_dir = Path(cfg["output"]["models_dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pth"

    best_val_dice = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                val_dice += _dice_score_batch(logits, masks, num_classes)
        val_loss /= len(val_dl)
        val_dice /= len(val_dl)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_dice=%.4f | %.1fs",
            epoch,
            cfg["training"]["epochs"],
            train_loss,
            val_loss,
            val_dice,
            elapsed,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice}
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_dice": val_dice, "cfg": cfg},
                best_ckpt,
            )
            logger.info("  -> saved best checkpoint (val_dice=%.4f)", val_dice)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val Dice: %.4f | checkpoint: %s", best_val_dice, best_ckpt)
    return best_ckpt
