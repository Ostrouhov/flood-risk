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
