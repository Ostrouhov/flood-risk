"""FastAPI app factory: монтирует статику, шаблоны, роуты; на старте создаёт схему БД."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from floodrisk.api.routes_api import router as api_router
from floodrisk.api.routes_html import router as html_router
from floodrisk.db.session import create_db_and_tables
from floodrisk.settings import settings


def _warm_models() -> None:
    """Резидентная загрузка зарегистрированных моделей в память (FR-13)."""
    import torch
    from sqlmodel import Session

    from floodrisk.db.repositories import list_model_versions
    from floodrisk.db.session import engine
    from floodrisk.inference.registry import warm

    torch.set_num_threads(min(6, max(1, os.cpu_count() or 4)))
    with Session(engine) as session:
        records = [(m.name, m.version, m.checkpoint_path) for m in list_model_versions(session)]
    n = warm(records)
    print(f"[app] резидентных моделей загружено: {n}")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    create_db_and_tables()
    try:
        _warm_models()
    except Exception as exc:  # прогрев не должен валить старт
        print(f"[app] прогрев моделей пропущен: {exc}")
    yield


def create_app() -> FastAPI:
    # StaticFiles требует, чтобы каталоги существовали на момент монтирования.
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="floodrisk",
        version="0.1.0",
        description="Прототип ИНС-оценки подверженности затоплению",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    app.mount("/runs", StaticFiles(directory=settings.runs_dir), name="runs")
    app.mount("/exports", StaticFiles(directory=settings.exports_dir), name="exports")

    app.include_router(html_router)
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()
