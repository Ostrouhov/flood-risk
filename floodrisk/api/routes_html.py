"""HTML-роуты для HTMX. На Этапе 0 — только GET / (главная страница с пустой картой)."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from floodrisk.settings import settings

router = APIRouter(tags=["html"])

templates = Jinja2Templates(directory=str(settings.templates_dir))


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"title": "floodrisk"})
