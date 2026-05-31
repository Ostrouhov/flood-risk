"""SQLModel-модели метаданных. См. SRS §5.1.

Все таблицы — реляционные, без геометрии. Геоданные живут как файлы;
БД хранит только пути и метки.
"""

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel, UniqueConstraint


def utcnow() -> datetime:
    return datetime.now(UTC)


class ModelVersion(SQLModel, table=True):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_versions_name_version"),)

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, description="например, unet, baseline")
    version: str = Field(description="например, v1")
    checkpoint_path: str
    config_path: str
    metrics_json: str | None = None
    dataset_version: str
    created_at: datetime = Field(default_factory=utcnow)


class Scenario(SQLModel, table=True):
    __tablename__ = "scenarios"

    id: int | None = Field(default=None, primary_key=True)
    scenario_id: str = Field(unique=True, index=True, description="p1y / p10y / p100y")
    name: str
    description: str | None = None
    params_json: str = Field(description="параметры из configs/scenarios.yaml")


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: str = Field(primary_key=True, description="UUID")
    bbox_w: float
    bbox_s: float
    bbox_e: float
    bbox_n: float
    scenario_id: str = Field(foreign_key="scenarios.scenario_id", index=True)
    model_version_id: int = Field(foreign_key="model_versions.id", index=True)
    prediction_tif_path: str | None = None
    prediction_png_path: str | None = None
    aggregates_json: str | None = None
    latency_ms: int | None = None
    status: str = Field(default="ok", description="ok / error")
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Export(SQLModel, table=True):
    __tablename__ = "exports"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    zip_path: str
    created_at: datetime = Field(default_factory=utcnow)


class Explanation(SQLModel, table=True):
    __tablename__ = "explanations"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    lat: float
    lon: float
    importance_json: str
    attribution_tif_paths_json: str
    created_at: datetime = Field(default_factory=utcnow)
