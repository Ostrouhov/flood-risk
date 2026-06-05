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


def _static_version(name: str) -> int:
    """mtime статики для cache-busting (?v=…): инвалидирует кэш браузера при правках."""
    try:
        return int((settings.static_dir / name).stat().st_mtime)
    except OSError:
        return 0


templates.env.globals["static_v"] = _static_version


def _coverage() -> list[dict]:
    """Покрытия всех региональных мозаик (Тулун, Канск, …) для контуров на карте."""
    from floodrisk.inference.service import mosaic_coverages

    try:
        return mosaic_coverages()
    except Exception:
        return []


@router.get("/", response_class=HTMLResponse)
def index(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    models = list_model_versions(session)
    coverage = _coverage()  # [W,S,E,N] или None
    model_label = f"{models[0].name}-{models[0].version}" if models else "— (не загружена)"
    # View-model сценариев: распарсенные params (мм/сут, повторяемость) для подписей в UI (A5).
    scenarios = []
    for s in list_scenarios(session):
        params = json.loads(s.params_json) if s.params_json else {}
        scenarios.append(
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "precip_mm": params.get("precipitation_mm_24h"),
                "return_period": params.get("return_period_years"),
            }
        )
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
    source: str = Form("auto"),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    from floodrisk.inference import service

    try:
        coords = [float(x) for x in bbox.split(",")]
        result = service.predict(session, coords, scenario_id, model_version, source=source)
    except service.OutOfCoverage:
        return templates.TemplateResponse(
            request,
            "fragments/predict_error.html",
            {"message": "bbox вне зоны покрытия пилотной территории"},
        )
    except service.BBoxTooLarge as exc:
        return templates.TemplateResponse(
            request,
            "fragments/predict_error.html",
            {"message": f"слишком большая зона (>{exc.limit_km:.0f} км) — уменьшите выбор"},
        )
    except service.OnlineFetchError:
        return templates.TemplateResponse(
            request,
            "fragments/predict_error.html",
            {"message": "не удалось загрузить данные рельефа для этой зоны (сеть/нет покрытия)"},
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
            "experimental": result["metadata"]["experimental"],
            "metadata": result["metadata"],
        },
    )


@router.post("/ui/explain", response_class=HTMLResponse)
def ui_explain(
    request: Request,
    run_id: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    from floodrisk.inference import explain as explain_mod
    from floodrisk.inference import service

    try:
        res = explain_mod.explain(session, run_id, lat, lon)
    except explain_mod.RunNotFound:
        return templates.TemplateResponse(
            request, "fragments/explanation_error.html", {"message": "сначала постройте карту"}
        )
    except explain_mod.PointOutOfCoverage:
        return templates.TemplateResponse(
            request, "fragments/explanation_error.html", {"message": "точка вне зоны покрытия"}
        )
    except (service.OnlineFetchError, service.BBoxTooLarge):
        return templates.TemplateResponse(
            request,
            "fragments/explanation_error.html",
            {"message": "не удалось собрать данные рельефа для этой точки (сеть/нет покрытия)"},
        )
    return templates.TemplateResponse(
        request,
        "fragments/explanation_panel.html",
        {
            "explanation_id": res["explanation_id"],
            "ranking": res["ranking"],
            "attribution_layers": res["attribution_layers"],
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
