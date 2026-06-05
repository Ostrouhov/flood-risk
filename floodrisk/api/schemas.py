"""Pydantic-схемы запросов/ответов. Дополняются по мере роста API (см. SRS §8)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: datetime


class BBox(BaseModel):
    """bbox в EPSG:4326 как [west, south, east, north]."""

    coords: list[float] = Field(min_length=4, max_length=4)


class ScenarioOut(BaseModel):
    scenario_id: str
    name: str
    description: str | None = None
    params: dict


class ModelVersionOut(BaseModel):
    name: str
    version: str
    dataset_version: str
    metrics: dict | None = None


class PredictRequest(BaseModel):
    """См. SRS §8.1. bbox = [west, south, east, north] в EPSG:4326."""

    bbox: list[float] = Field(min_length=4, max_length=4)
    scenario_id: str
    model_version: str = "unet-v1"
    # источник признаков: auto (мозаика в покрытии, иначе онлайн), mosaic, online
    source: Literal["auto", "mosaic", "online"] = "auto"


class PredictResponse(BaseModel):
    run_id: str
    prediction_png_url: str
    prediction_tif_url: str
    bounds_wgs84: list[float]  # [south, west, north, east] для Leaflet
    aggregates: dict
    metadata: dict


class ExportResponse(BaseModel):
    export_url: str


class ExplainRequest(BaseModel):
    """См. SRS §8.1. Точка клика в EPSG:4326."""

    run_id: str
    lat: float
    lon: float


class ExplainResponse(BaseModel):
    explanation_id: int
    ranking: list[dict]
    attribution_layers: list[dict]


class PointResponse(BaseModel):
    """Значение вероятности в точке клика (инспектор точки, режим «Просмотр»)."""

    lat: float
    lon: float
    probability: float | None = None
    in_bounds: bool


class GroundTruthResponse(BaseModel):
    """Реальная маска затопления S1 и согласие с предсказанием на зоне запуска.

    available=False — зона вне размеченной области (онлайн/вне Тулуна): маски нет.
    """

    available: bool
    png_url: str | None = None
    bounds_wgs84: list[float] | None = None  # [south, west, north, east] для Leaflet
    metrics: dict | None = None
    threshold: float = 0.5
    # experimental=True, если маска из невалидированного региона (напр. Канск): признаки
    # корректны, но S1-лейблы не сверены с эталоном (пере-детекция).
    experimental: bool = False
    regions: list[str] | None = None


class RunOut(BaseModel):
    run_id: str
    bbox: list[float]
    scenario_id: str
    model_version_id: int
    prediction_png_url: str | None = None
    prediction_tif_url: str | None = None
    aggregates: dict | None = None
    latency_ms: int | None = None
    status: str


class HotspotsResponse(BaseModel):
    """Топ-N кластеров высокого риска по растру запуска (связные компоненты p≥threshold).

    Каждый hotspot: {rank, lat, lon, area_km2, mean_p, max_p, bounds_wgs84:[S,W,N,E]}.
    ``count`` — всего кластеров выше фильтра площади (может быть > len(hotspots)).
    """

    available: bool
    threshold: float = 0.5
    count: int = 0
    hotspots: list[dict] = []
