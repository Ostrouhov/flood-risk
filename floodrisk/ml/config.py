"""Загрузка YAML-конфигов тренировки/оценки. См. SRS §7.8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Конфиг {path} должен быть YAML-объектом")
    return cfg


def tracking_uri(cfg: dict[str, Any]) -> str:
    """Нормализует mlflow.tracking_uri к абсолютному file-URI (Windows-safe).

    'file:./mlruns' на Windows парсится в UNC '\\\\.\\mlruns' → ошибка.
    """
    raw = str(cfg["mlflow"]["tracking_uri"])
    # Удалённые/БД-бэкенды пропускаем как есть.
    for scheme in ("http://", "https://", "sqlite:", "postgresql:", "databricks"):
        if raw.startswith(scheme):
            return raw
    # Любая file-форма (file://./x, file:./x, ./x, x) → абсолютный file-URI.
    for prefix in ("file://", "file:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    return Path(raw).resolve().as_uri()


def is_unet(cfg: dict[str, Any]) -> bool:
    return cfg.get("model", {}).get("arch") == "unet"


def is_baseline(cfg: dict[str, Any]) -> bool:
    return "estimator" in cfg.get("model", {})


# Порядок каналов в тайлах data/processed/v1 (band descriptions из preprocess.py).
CHANNEL_NAMES: list[str] = [
    "dem",
    "slope",
    "aspect",
    "curvature",
    "twi",
    "worldcover",
    "dist_to_water",
]
