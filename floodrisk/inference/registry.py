"""Резидентный реестр моделей: загрузка U-Net (TorchScript) и baseline (sklearn). FR-13.

Модели грузятся один раз и держатся в памяти процесса. Только core-зависимости
(torch, sklearn) — без extras [ml]. predict_tile принимает УЖЕ нормированные признаки.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class LoadedModel:
    name: str
    version: str
    kind: str  # 'unet' | 'baseline'
    predict_tile: Callable[[np.ndarray], np.ndarray]


_CACHE: dict[tuple[str, str], LoadedModel] = {}


def _load_torchscript(path: str) -> Callable[[np.ndarray], np.ndarray]:
    module = torch.jit.load(path, map_location="cpu")
    module.eval()

    def predict_tile(chw: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(chw[None], dtype="float32"))
        with torch.no_grad():
            logit = module(t)
        return torch.sigmoid(logit)[0, 0].numpy().astype("float32")

    return predict_tile


def _load_sklearn(path: str) -> Callable[[np.ndarray], np.ndarray]:
    with open(path, "rb") as f:
        clf = pickle.load(f)["model"]

    def predict_tile(chw: np.ndarray) -> np.ndarray:
        c, h, w = chw.shape
        x = chw.reshape(c, -1).T
        p = clf.predict_proba(x)[:, 1]
        return p.reshape(h, w).astype("float32")

    return predict_tile


def load_model(name: str, version: str, checkpoint_path: str) -> LoadedModel:
    key = (name, version)
    if key in _CACHE:
        return _CACHE[key]
    suffix = Path(checkpoint_path).suffix
    if suffix == ".pt":
        kind, predict = "unet", _load_torchscript(checkpoint_path)
    elif suffix == ".pkl":
        kind, predict = "baseline", _load_sklearn(checkpoint_path)
    else:
        raise ValueError(f"неподдерживаемый артефакт модели: {checkpoint_path}")
    model = LoadedModel(name=name, version=version, kind=kind, predict_tile=predict)
    _CACHE[key] = model
    return model


def warm(records: list[tuple[str, str, str]]) -> int:
    """Прогрев: грузит модели по списку (name, version, checkpoint_path). Ошибки не валят старт."""
    n = 0
    for name, version, path in records:
        try:
            load_model(name, version, path)
            n += 1
        except Exception as exc:
            print(f"[registry] не загружена {name}-{version}: {exc}")
    return n


def clear_cache() -> None:
    _CACHE.clear()
