"""JSON API. На Этапе 0 — только /health; остальные эндпоинты приедут на Этапе 3 (см. SRS §8.1)."""

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlmodel import Session

from floodrisk import __version__
from floodrisk.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    ExportResponse,
    GroundTruthResponse,
    HealthResponse,
    ModelVersionOut,
    PointResponse,
    PredictRequest,
    PredictResponse,
    RunOut,
    ScenarioOut,
)
from floodrisk.db.repositories import get_run, list_model_versions, list_scenarios
from floodrisk.db.session import get_session
from floodrisk.settings import settings

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
        return service.predict(
            session, req.bbox, req.scenario_id, req.model_version, source=req.source
        )
    except service.InvalidBBox as exc:
        return JSONResponse(status_code=400, content={"error": "invalid_bbox", "detail": str(exc)})
    except service.OutOfCoverage as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "bbox_out_of_coverage", "coverage_wgs84": exc.coverage_wgs84},
        )
    except service.BBoxTooLarge as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "bbox_too_large", "limit_km": exc.limit_km, "side_km": exc.side_km},
        )
    except service.OnlineFetchError as exc:
        return JSONResponse(
            status_code=502, content={"error": "online_fetch_failed", "detail": str(exc)}
        )
    except service.UnknownScenario as exc:
        return JSONResponse(
            status_code=404, content={"error": "unknown_scenario", "available": exc.available}
        )
    except service.UnknownModel as exc:
        return JSONResponse(
            status_code=404, content={"error": "unknown_model", "available": exc.available}
        )


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run_meta(run_id: str, session: Session = Depends(get_session)) -> RunOut:
    run = get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return RunOut(
        run_id=run.id,
        bbox=[run.bbox_w, run.bbox_s, run.bbox_e, run.bbox_n],
        scenario_id=run.scenario_id,
        model_version_id=run.model_version_id,
        prediction_png_url=run.prediction_png_path,
        prediction_tif_url=run.prediction_tif_path,
        aggregates=json.loads(run.aggregates_json) if run.aggregates_json else None,
        latency_ms=run.latency_ms,
        status=run.status,
    )


@router.get("/runs/{run_id}/point", response_model=PointResponse)
def get_run_point(run_id: str, lat: float, lon: float) -> PointResponse:
    """Вероятность затопления в точке (инспектор точки). См. FR-9, режим «Просмотр»."""
    from floodrisk.inference import service

    try:
        prob = service.sample_point(run_id, lat, lon)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run_not_found") from None
    return PointResponse(lat=lat, lon=lon, probability=prob, in_bounds=prob is not None)


@router.get("/runs/{run_id}/groundtruth", response_model=GroundTruthResponse)
def get_run_groundtruth(run_id: str, threshold: float = 0.5) -> GroundTruthResponse:
    """Реальная маска S1 «что затопило» + согласие с предсказанием на зоне.

    Доступно только для размеченных событий (пилот — Тулун). Вне покрытия /
    онлайн-режим → available=False. См. кластер «Научная валидация на карте».
    """
    import json

    from floodrisk.inference import validation

    run_dir = settings.runs_dir / run_id
    if not (run_dir / "prediction.tif").exists():
        raise HTTPException(status_code=404, detail="run_not_found")

    # dataset_version и источник — из metadata.json расчёта (онлайн заведомо без маски).
    meta_path = run_dir / "metadata.json"
    dataset_version = "v1"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source") == "online":
            return GroundTruthResponse(available=False, threshold=threshold)
        dataset_version = meta.get("dataset_version", "v1")

    result = validation.ground_truth_for_run(run_id, dataset_version, threshold)
    if result is None:
        return GroundTruthResponse(available=False, threshold=threshold)
    return GroundTruthResponse(
        available=True,
        png_url=result["png_url"],
        bounds_wgs84=result["bounds_wgs84"],
        metrics=result["metrics"],
        threshold=result["threshold"],
        experimental=result.get("experimental", False),
        regions=result.get("regions"),
    )


@router.post("/runs/{run_id}/export", response_model=ExportResponse)
def export_run(run_id: str, session: Session = Depends(get_session)):
    """Готовит ZIP (TIF + GeoJSON + PDF). См. SRS §6.6, FR-17."""
    from floodrisk.inference.exporter import RunNotFound, build_export

    try:
        url = build_export(session, run_id)
    except RunNotFound:
        return JSONResponse(status_code=404, content={"error": "run_not_found"})
    return ExportResponse(export_url=url)


@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, session: Session = Depends(get_session)):
    """Важность признаков в точке клика. См. SRS §8.1, §7.7, FR-10."""
    from floodrisk.inference import explain as explain_mod
    from floodrisk.inference import service

    try:
        return explain_mod.explain(session, req.run_id, req.lat, req.lon)
    except explain_mod.RunNotFound:
        return JSONResponse(status_code=404, content={"error": "run_not_found"})
    except explain_mod.PointOutOfCoverage:
        return JSONResponse(status_code=422, content={"error": "point_out_of_coverage"})
    except (service.OnlineFetchError, service.BBoxTooLarge) as exc:
        return JSONResponse(
            status_code=502, content={"error": "online_fetch_failed", "detail": str(exc)}
        )
