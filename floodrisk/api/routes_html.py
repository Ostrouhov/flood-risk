"""HTML-роуты для HTMX (Этап 4). Главная страница + /ui/predict + /ui/export.

Внутри используют тот же inference.service / exporter, что и JSON API (единая логика, SRS §8.2).
"""

import json

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from floodrisk.db.repositories import list_model_versions, list_scenarios
from floodrisk.db.session import get_session
from floodrisk.settings import settings

router = APIRouter(tags=["html"])

templates = Jinja2Templates(directory=str(settings.templates_dir))


def _coverage() -> list[float] | None:
    from floodrisk.inference.features import coverage_wgs84
    from floodrisk.inference.service import feature_stack_path

    try:
        return coverage_wgs84(feature_stack_path("v1"))
    except Exception:
        return None


@router.get("/", response_class=HTMLResponse)
def index(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    scenarios = list_scenarios(session)
    models = list_model_versions(session)
    coverage = _coverage()  # [W,S,E,N] или None
    model_label = f"{models[0].name}-{models[0].version}" if models else "— (не загружена)"
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "floodrisk",
            "scenarios": scenarios,
            "models": models,
            "model_label": model_label,
            "coverage": coverage,
        },
    )


@router.post("/ui/predict", response_class=HTMLResponse)
def ui_predict(
    request: Request,
    scenario_id: str = Form(...),
    model_version: str = Form(...),
    bbox: str = Form(...),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    from floodrisk.inference import service

    try:
        coords = [float(x) for x in bbox.split(",")]
        result = service.predict(session, coords, scenario_id, model_version)
    except service.OutOfCoverage:
        return templates.TemplateResponse(
            request,
            "fragments/predict_error.html",
            {"message": "bbox вне зоны покрытия пилотной территории"},
        )
    except (service.InvalidBBox, ValueError):
        return templates.TemplateResponse(
            request, "fragments/predict_error.html", {"message": "некорректный bbox"}
        )
    except (service.UnknownScenario, service.UnknownModel) as exc:
        return templates.TemplateResponse(
            request, "fragments/predict_error.html", {"message": str(exc)}
        )
    return templates.TemplateResponse(
        request,
        "fragments/predict_result.html",
        {
            "prediction_png_url": result["prediction_png_url"],
            "bounds_wgs84": result["bounds_wgs84"],
            "run_id": result["run_id"],
            "aggregates": result["aggregates"],
        },
    )


@router.post("/ui/export", response_class=HTMLResponse)
def ui_export(
    request: Request,
    run_id: str = Form(...),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    from floodrisk.inference.exporter import RunNotFound, build_export

    try:
        url = build_export(session, run_id)
    except RunNotFound:
        return templates.TemplateResponse(
            request, "fragments/predict_error.html", {"message": "запуск не найден"}
        )
    response = templates.TemplateResponse(
        request, "fragments/export_result.html", {"export_url": url}
    )
    response.headers["HX-Trigger"] = json.dumps({"download-ready": {"url": url}})
    return response
