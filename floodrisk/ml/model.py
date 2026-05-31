"""U-Net LightningModule (smp.Unet). См. SRS §7.2.1, FR-5."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

try:
    import pytorch_lightning as pl

    _BASE = pl.LightningModule
except Exception:  # pragma: no cover
    _BASE = nn.Module

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryJaccardIndex,
)


def build_unet(model_cfg: dict) -> nn.Module:
    """smp.Unet с resnet34; pretrained imagenet, при отсутствии сети — random init."""
    common = dict(
        encoder_name=model_cfg.get("encoder", "resnet34"),
        in_channels=int(model_cfg["in_channels"]),
        classes=int(model_cfg.get("out_channels", 1)),
    )
    try:
        return smp.Unet(encoder_weights="imagenet", **common)
    except Exception as exc:  # нет сети/весов → инициализация с нуля
        print(f"[unet] imagenet-веса недоступны ({exc}); encoder_weights=None")
        return smp.Unet(encoder_weights=None, **common)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    p = prob.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    denom = p.sum(1) + t.sum(1)
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t).pow(gamma) * bce).mean()


class UNetLitModule(_BASE):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        tr = cfg["training"]
        self.net = build_unet(cfg["model"])
        self.loss_name = tr.get("loss", "bce_dice")
        pos_weight = float(tr.get("pos_weight", 1.0))
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.lr = float(tr.get("lr", 1e-3))
        self.max_epochs = int(tr.get("max_epochs", 20))
        # Метрики (порог 0.5 для IoU/F1).
        self.val_iou = BinaryJaccardIndex(threshold=0.5)
        self.val_f1 = BinaryF1Score(threshold=0.5)
        self.val_auroc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        if self.loss_name == "focal_dice":
            return focal_loss(logits, y) + dice_loss(logits, y)
        return bce + dice_loss(logits, y)

    def training_step(self, batch, _idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        prob = torch.sigmoid(logits)
        yi = y.int()
        self.val_iou.update(prob, yi)
        self.val_f1.update(prob, yi)
        self.val_auroc.update(prob.flatten(), yi.flatten())
        self.val_ap.update(prob.flatten(), yi.flatten())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        self.log("val_iou", self.val_iou.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute())
        self.log("val_roc_auc", self.val_auroc.compute())
        self.log("val_pr_auc", self.val_ap.compute())
        for m in (self.val_iou, self.val_f1, self.val_auroc, self.val_ap):
            m.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
