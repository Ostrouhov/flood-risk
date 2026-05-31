"""Dataset/DataModule для тайлов floodrisk. См. SRS §7, FR-5.

Тайлы `data/processed/v1/tiles/t_rRRRcCCC.tif` — 7 float32-каналов
(dem, slope, aspect, curvature, twi, worldcover, dist_to_water).
Лейблы `data/processed/v1/labels/<event>/t_rRRRcCCC.tif` — uint8 {0,1}, 1 band.

Каналы не нормированы → нормируем z-score по статистикам train-split
(`norm_stats.json`). Датасет мал (~390 МБ) → кэшируем в RAM, num_workers=0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from floodrisk import feature_transform as ft

try:  # Lightning опционален для чистой загрузки данных
    import pytorch_lightning as pl

    _LDM = pl.LightningDataModule
except Exception:  # pragma: no cover
    _LDM = object


def _read_tile(path: Path) -> np.ndarray:
    """Читает 7-канальный тайл → float32 [C, H, W], NaN/inf → 0."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _read_label(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1)
    return (arr > 0).astype("float32")


def tile_paths(cfg: dict, split: str) -> list[tuple[str, Path, Path]]:
    """(tile_id, tile_path, label_path) для заданного split по index.parquet."""
    d = cfg["data"]
    index = pd.read_parquet(d["index"])
    tile_dir = Path(d["tile_dir"])
    label_dir = Path(d["label_dir"])
    rows = index[index["split"] == split]
    out: list[tuple[str, Path, Path]] = []
    for tid in rows["tile_id"]:
        out.append((tid, tile_dir / f"{tid}.tif", label_dir / f"{tid}.tif"))
    return out


def compute_norm_stats(cfg: dict) -> dict:
    """mean/std непрерывных каналов по train-split (для feature_transform). → norm_stats.json."""
    items = tile_paths(cfg, "train")
    if not items:
        raise RuntimeError("train-split пуст — нечего нормировать")
    stats = ft.compute_stats(_read_tile(tpath) for _tid, tpath, _lpath in items)
    out_path = Path(cfg["data"]["norm_stats"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_norm_stats(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    return ft.load_stats(cfg["data"]["norm_stats"])


def _augment(x: np.ndarray, y: np.ndarray, names: list[str], rng: np.random.Generator):
    """Геометрические аугментации (rot90/hflip/vflip), общие для x[C,H,W] и y[H,W]."""
    if "rot90" in names:
        k = int(rng.integers(0, 4))
        if k:
            x = np.rot90(x, k=k, axes=(1, 2))
            y = np.rot90(y, k=k, axes=(0, 1))
    if "hflip" in names and rng.random() < 0.5:
        x = x[:, :, ::-1]
        y = y[:, ::-1]
    if "vflip" in names and rng.random() < 0.5:
        x = x[:, ::-1, :]
        y = y[::-1, :]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class FloodTileDataset(Dataset):
    """In-RAM кэш тайлов + нормировка + (для train) аугментации."""

    def __init__(self, cfg: dict, split: str, *, augment: bool, seed: int = 42):
        self.items = tile_paths(cfg, split)
        self.mean, self.std = load_norm_stats(cfg)
        self.aug_names = list(cfg["data"].get("augmentations", [])) if augment else []
        self.rng = np.random.default_rng(seed)
        # Кэш в RAM: нормированные признаки + лейблы.
        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        for _tid, tpath, lpath in self.items:
            self._x.append(ft.transform(_read_tile(tpath), self.mean, self.std))
            self._y.append(_read_label(lpath))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, y = self._x[idx], self._y[idx]
        if self.aug_names:
            x, y = _augment(x, y, self.aug_names, self.rng)
        return torch.from_numpy(x), torch.from_numpy(y[None, :, :])

    def tile_ids(self) -> list[str]:
        return [it[0] for it in self.items]


class FloodDataModule(_LDM):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.batch_size = int(cfg["data"].get("batch_size", 8))
        self.num_workers = int(cfg["data"].get("num_workers", 0))
        self.seed = int(cfg.get("training", {}).get("seed", 42))

    def setup(self, stage: str | None = None):
        self.train_ds = FloodTileDataset(self.cfg, "train", augment=True, seed=self.seed)
        self.val_ds = FloodTileDataset(self.cfg, "val", augment=False, seed=self.seed)
        self.test_ds = FloodTileDataset(self.cfg, "test", augment=False, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
