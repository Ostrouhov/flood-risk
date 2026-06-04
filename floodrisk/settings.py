"""Конфиг приложения через pydantic-settings (читает .env и переменные окружения)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "sqlite:///./app.db"
    log_level: str = "INFO"

    cds_api_key: str | None = None
    copernicus_token: str | None = None
    mlflow_tracking_uri: str = "file://./mlruns"

    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)

    @property
    def templates_dir(self) -> Path:
        return self.project_root / "floodrisk" / "templates"

    @property
    def static_dir(self) -> Path:
        return self.project_root / "static"

    @property
    def configs_dir(self) -> Path:
        return self.project_root / "configs"

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.project_root / "exports"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def online_cache_dir(self) -> Path:
        """Кэш тайлов онлайн-инференса (DEM/WorldCover/JRC) по произвольному bbox."""
        return self.project_root / "data" / "cache" / "online"


settings = Settings()
