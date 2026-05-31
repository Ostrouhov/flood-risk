"""Типизированная загрузка ``configs/data.yaml`` (SRS §6.4, §11.1).

Чистая логика без сети и GDAL — тестируется на хост-Python. Все остальные модули
data-пайплайна принимают распарсенный :class:`DataConfig`, а не сырой YAML.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from floodrisk.settings import settings

DEFAULT_CONFIG_PATH = settings.configs_dir / "data.yaml"


class SplitConfig(BaseModel):
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42
    method: str = "geographic_bbox"

    @field_validator("test")
    @classmethod
    def _fractions_sum_to_one(cls, v: float, info) -> float:
        train = info.data.get("train", 0.0)
        val = info.data.get("val", 0.0)
        total = train + val + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"splits должны суммироваться в 1.0, получено {total}")
        return v


class EventConfig(BaseModel):
    """Событие с разливом — источник лейблов Sentinel-1 (FR-3)."""

    event_id: str
    role: str = "train_label"
    flood_window: list[date]  # [start, end] пик разлива (post-event)
    dry_window: list[date]  # [start, end] "сухое" окно для change detection
    reference: str | None = None  # эталон валидации, напр. copernicus_ems

    @field_validator("flood_window", "dry_window")
    @classmethod
    def _two_dates(cls, v: list[date]) -> list[date]:
        if len(v) != 2:
            raise ValueError("окно дат должно быть [start, end]")
        return v


class ValidationConfig(BaseModel):
    metric: str = "iou"
    target: float = 0.50
    fallback: float = 0.40


class LabelsConfig(BaseModel):
    """Параметры детектора затопления Sentinel-1 (OQ-2)."""

    method: str = "otsu_change_detection"
    primary_band: str = "VV"
    slope_mask_deg: float = 5.0
    min_blob_px: int = 8
    permanent_water_occurrence_pct: int = 50
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class DataConfig(BaseModel):
    """Корневой конфиг датасета. Соответствует ``configs/data.yaml``."""

    dataset_version: str = "v1"
    pilot_bbox_wgs84: list[float]  # [W, S, E, N]
    target_crs: str
    tile_size_px: int = 256
    tile_resolution_m: float = 30.0
    tile_overlap_px: int = 0
    inference_overlap_px: int = 32
    splits: SplitConfig = Field(default_factory=SplitConfig)
    events: list[EventConfig] = Field(default_factory=list)
    sources: dict[str, dict] = Field(default_factory=dict)
    labels_s1: LabelsConfig = Field(default_factory=LabelsConfig)

    @field_validator("pilot_bbox_wgs84")
    @classmethod
    def _valid_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("bbox должен быть [W, S, E, N]")
        w, s, e, n = v
        if not (w < e and s < n):
            raise ValueError(f"некорректный bbox: W<E и S<N нарушены ({v})")
        if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
            raise ValueError(f"bbox вне диапазона WGS84: {v}")
        return v

    @field_validator("target_crs")
    @classmethod
    def _epsg_form(cls, v: str) -> str:
        if not v.upper().startswith("EPSG:"):
            raise ValueError(f"target_crs ожидается в форме 'EPSG:XXXXX', получено {v!r}")
        return v.upper()

    def event(self, event_id: str) -> EventConfig:
        """Событие по id; KeyError, если нет."""
        for ev in self.events:
            if ev.event_id == event_id:
                return ev
        raise KeyError(f"событие {event_id!r} не найдено в data.yaml")


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Прочитать и провалидировать ``configs/data.yaml``."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"конфиг данных не найден: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return DataConfig.model_validate(raw)
