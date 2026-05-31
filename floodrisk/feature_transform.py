"""Преобразование сырых 7-канальных тайлов во вход модели. Общий для train и inference.

Сырые каналы (порядок в stack.tif): dem, slope, aspect, curvature, twi, worldcover, dist_to_water.
Корректное кодирование (вместо плоского z-score):
  - непрерывные (dem, slope, curvature, twi, dist_to_water) → z-score по train-статистикам;
  - aspect (циклический 0–360°) → sin/cos (2 канала);
  - worldcover (категориальные классы) → one-hot по каноническим классам WorldCover.
Итог: 5 + 2 + 11 = 18 каналов. Модуль — core (numpy), без extras; единственный источник истины,
чтобы train и inference давали идентичный вход.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floodrisk.normalization import normalize

# Порядок сырых каналов в data/processed/v1/features/stack.tif.
RAW_ORDER = ["dem", "slope", "aspect", "curvature", "twi", "worldcover", "dist_to_water"]
CONTINUOUS = ["dem", "slope", "curvature", "twi", "dist_to_water"]
ASPECT = "aspect"
WORLDCOVER = "worldcover"
# Канонические классы ESA WorldCover 2021 v200.
WC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def _idx(name: str) -> int:
    return RAW_ORDER.index(name)


def out_channels() -> int:
    return len(CONTINUOUS) + 2 + len(WC_CLASSES)  # 18


# Человекочитаемые подписи семантических признаков (для /explain, UI).
FEATURE_LABELS = {
    "dem": "Высота (DEM)",
    "slope": "Уклон",
    "curvature": "Кривизна",
    "twi": "Индекс влажности (TWI)",
    "dist_to_water": "Расстояние до воды",
    "aspect": "Экспозиция склона",
    "worldcover": "Землепокров",
}


def feature_groups() -> dict[str, list[int]]:
    """Семантический признак → индексы выходных каналов (для агрегации атрибуций IG).

    aspect = sin+cos (2 канала), worldcover = one-hot (11 каналов) → сворачиваются обратно
    в 7 исходных признаков. Порядок — как в transform().
    """
    groups: dict[str, list[int]] = {}
    i = 0
    for name in CONTINUOUS:
        groups[name] = [i]
        i += 1
    groups[ASPECT] = [i, i + 1]
    i += 2
    groups[WORLDCOVER] = list(range(i, i + len(WC_CLASSES)))
    return groups


def compute_stats(raw_tiles) -> dict:
    """Считает mean/std ТОЛЬКО для непрерывных каналов (в сыром пространстве).

    ``raw_tiles`` — итерируемое сырых тайлов [7,H,W]. Возвращает dict для norm_stats.json.
    """
    cont_idx = [_idx(n) for n in CONTINUOUS]
    n = len(cont_idx)
    csum = np.zeros(n, dtype="float64")
    csqsum = np.zeros(n, dtype="float64")
    count = 0
    for raw in raw_tiles:
        flat = raw[cont_idx].reshape(n, -1)
        csum += flat.sum(axis=1)
        csqsum += (flat.astype("float64") ** 2).sum(axis=1)
        count += flat.shape[1]
    mean = csum / count
    var = np.maximum(csqsum / count - mean**2, 1e-12)
    return {
        "continuous": CONTINUOUS,
        "count_pixels": int(count),
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
    }


def load_stats(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """mean/std непрерывных каналов → [n,1,1] для вещания."""
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype="float32").reshape(-1, 1, 1)
    std = np.asarray(stats["std"], dtype="float32").reshape(-1, 1, 1)
    return mean, std


def transform(raw_chw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Сырые [7,H,W] → вход модели [18,H,W] (float32). Порядок детерминирован."""
    cont = raw_chw[[_idx(n) for n in CONTINUOUS]]
    cont_z = normalize(cont, mean, std)

    aspect_rad = np.deg2rad(raw_chw[_idx(ASPECT)].astype("float32"))
    sin = np.sin(aspect_rad)[None]
    cos = np.cos(aspect_rad)[None]

    wc = raw_chw[_idx(WORLDCOVER)]
    onehot = np.stack([(wc == c).astype("float32") for c in WC_CLASSES])

    return np.ascontiguousarray(np.concatenate([cont_z, sin, cos, onehot], axis=0), dtype="float32")
