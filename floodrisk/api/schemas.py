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
