"""Pydantic-схемы запросов/ответов. Дополняются по мере роста API (см. SRS §8)."""

from datetime import datetime

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


class PredictResponse(BaseModel):
    run_id: str
    prediction_png_url: str
    prediction_tif_url: str
    bounds_wgs84: list[float]  # [south, west, north, east] для Leaflet
    aggregates: dict
    metadata: dict


class ExportResponse(BaseModel):
    export_url: str


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
