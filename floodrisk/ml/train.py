"""Точка входа тренировки: U-Net (FR-5) или baseline (FR-6). См. SRS §7.6."""

from __future__ import annotations

from pathlib import Path

from floodrisk.ml import config as cfgmod
from floodrisk.ml.data import compute_norm_stats


def _ensure_norm_stats(cfg: dict) -> None:
    p = Path(cfg["data"]["norm_stats"])
    if not p.exists():
        print(f"[norm] статистик нет → считаю по train-split → {p}")
        compute_norm_stats(cfg)


def train_unet(cfg: dict) -> int:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import MLFlowLogger

    tr = cfg["training"]
    pl.seed_everything(int(tr.get("seed", 42)), workers=True)
    torch.set_num_threads(int(tr.get("torch_num_threads", 6)))

    from floodrisk.ml.data import FloodDataModule
    from floodrisk.ml.model import UNetLitModule

    dm = FloodDataModule(cfg)
    model = UNetLitModule(cfg)

    ckpt_cfg = cfg["checkpoint"]
    Path(ckpt_cfg["dirpath"]).mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_cfg["dirpath"],
        filename="unet-v1-{epoch:02d}-{val_iou:.3f}",
        monitor=ckpt_cfg.get("monitor", "val_iou"),
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=int(ckpt_cfg.get("save_top_k", 1)),
        save_last=True,
    )
    logger = MLFlowLogger(
        experiment_name=cfg["mlflow"]["experiment_name"],
        tracking_uri=cfgmod.tracking_uri(cfg),
    )
    logger.log_hyperparams({
        "encoder": cfg["model"].get("encoder"),
        "in_channels": cfg["model"]["in_channels"],
        "loss": tr.get("loss"),
        "pos_weight": tr.get("pos_weight"),
        "lr": tr.get("lr"),
        "max_epochs": tr.get("max_epochs"),
        "batch_size": cfg["data"].get("batch_size"),
    })

    trainer = pl.Trainer(
        max_epochs=int(tr.get("max_epochs", 20)),
        accelerator=tr.get("accelerator", "cpu"),
        devices=1,
        precision=str(tr.get("precision", 32)),
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=dm)
    print(f"[unet] best={checkpoint.best_model_path} ({checkpoint.best_model_score})")

    # Зафиксировать путь к лучшему чекпойнту для eval.
    best = checkpoint.best_model_path or checkpoint.last_model_path
    (Path(ckpt_cfg["dirpath"]) / "best.txt").write_text(best, encoding="utf-8")
    return 0


def train_baseline(cfg: dict) -> int:
    import mlflow

    from floodrisk.ml.baseline import train_baseline as _train

    mlflow.set_tracking_uri(cfgmod.tracking_uri(cfg))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params(cfg["model"].get("params", {}))
        out = _train(cfg)
        mlflow.log_artifact(str(out))
    return 0


def run_train(config_path: str) -> int:
    cfg = cfgmod.load_yaml(config_path)
    _ensure_norm_stats(cfg)
    if cfgmod.is_unet(cfg):
        return train_unet(cfg)
    if cfgmod.is_baseline(cfg):
        return train_baseline(cfg)
    print(f"[train] не распознан тип модели в {config_path}")
    return 2
