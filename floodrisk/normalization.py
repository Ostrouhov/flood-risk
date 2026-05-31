"""Единая нормировка признаков — общий источник для обучения и инференса.

Раньше логика дублировалась в floodrisk/ml/data.py и floodrisk/inference/service.py,
что грозило рассинхроном train↔inference. Модуль — core (только numpy/json), без extras.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_stats(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Читает norm_stats.json → (mean[C,1,1], std[C,1,1]) float32 для вещания на [C,H,W]."""
    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype="float32").reshape(-1, 1, 1)
    std = np.asarray(stats["std"], dtype="float32").reshape(-1, 1, 1)
    return mean, std


def normalize(chw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(x - mean) / std для тензора признаков [C,H,W]. Возвращает float32."""
    return ((chw - mean) / std).astype("float32")
