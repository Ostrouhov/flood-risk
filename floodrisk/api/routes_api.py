"""JSON API. На Этапе 0 — только /health; остальные эндпоинты приедут на Этапе 3 (см. SRS §8.1)."""

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlmodel import Session

from floodrisk import __version__
from floodrisk.api.schemas import (
    HealthResponse,
    ModelVersionOut,
    PredictRequest,
    PredictResponse,
    ScenarioOut,
)
from floodrisk.db.repositories import list_model_versions, list_scenarios
from floodrisk.db.session import get_session

router = APIRouter(tags=["api"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=datetime.now(UTC),
    )


@router.get("/scenarios", response_model=list[ScenarioOut])
def get_scenarios(session: Session = Depends(get_session)) -> list[ScenarioOut]:
    rows = list_scenarios(session)
    return [
        ScenarioOut(
            scenario_id=r.scenario_id,
            name=r.name,
            description=r.description,
            params=json.loads(r.params_json),
        )
        for r in rows
    ]


@router.get("/model-versions", response_model=list[ModelVersionOut])
def get_model_versions(session: Session = Depends(get_session)) -> list[ModelVersionOut]:
    rows = list_model_versions(session)
    return [
        ModelVersionOut(
            name=r.name,
            version=r.version,
            dataset_version=r.dataset_version,
            metrics=json.loads(r.metrics_json) if r.metrics_json else None,
        )
        for r in rows
    ]


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, session: Session = Depends(get_session)):
    """Инференс карты подверженности по bbox+сценарию+модели. См. SRS §8.1, FR-9/FR-12."""
    from floodrisk.inference import service

    try:
        return service.predict(session, req.bbox, req.scenario_id, req.model_version)
    except service.InvalidBBox as exc:
        return JSONResponse(status_code=400, content={"error": "invalid_bbox", "detail": str(exc)})
    except service.OutOfCoverage as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "bbox_out_of_coverage", "coverage_wgs84": exc.coverage_wgs84},
        )
    except service.UnknownScenario as exc:
        return JSONResponse(
            status_code=404, content={"error": "unknown_scenario", "available": exc.available}
        )
    except service.UnknownModel as exc:
        return JSONResponse(
            status_code=404, content={"error": "unknown_model", "available": exc.available}
        )


@router.post("/explain", status_code=501)
def explain() -> dict:
    raise HTTPException(
        status_code=501,
        detail="not_implemented: explain будет добавлен на Этапе 5 (см. SRS §17)",
    )
