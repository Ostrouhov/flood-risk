"""Экспорт запуска в ZIP (TIF + GeoJSON + PDF). См. SRS §6.6, FR-17.

Зависимости только core: reportlab (PDF), стандартный json/zipfile. Кириллица в PDF —
через шрифт DejaVuSans из поставки matplotlib (портируемо в Docker).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from sqlmodel import Session

from floodrisk.db.models import Export, Run
from floodrisk.db.repositories import get_run
from floodrisk.settings import settings

LEGEND = [
    ("0.0", "#440154"),
    ("0.25", "#3b528b"),
    ("0.5", "#21918c"),
    ("0.75", "#5ec962"),
    ("1.0", "#fde725"),
]


class RunNotFound(Exception):
    pass


def _run_dir(run_id: str) -> Path:
    return settings.runs_dir / run_id


def _aggregates_geojson(run: Run, aggregates: dict) -> dict:
    w, s, e, n = run.bbox_w, run.bbox_s, run.bbox_e, run.bbox_n
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
                },
                "properties": {
                    "run_id": run.id,
                    "scenario_id": run.scenario_id,
                    "model_version_id": run.model_version_id,
                    **aggregates,
                },
            }
        ],
    }


def _build_pdf(pdf_path: Path, run: Run, aggregates: dict, png_path: Path) -> None:
    import matplotlib
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font = "DejaVu"
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    try:
        pdfmetrics.registerFont(TTFont(font, str(ttf)))
    except Exception:
        font = "Helvetica"

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    _pw, ph = A4
    y = ph - 25 * mm
    c.setFont(font, 16)
    c.drawString(20 * mm, y, "floodrisk — карта подверженности затоплению")
    y -= 12 * mm

    c.setFont(font, 10)
    meta = [
        f"run_id: {run.id}",
        f"сценарий: {run.scenario_id}",
        f"bbox (W,S,E,N): {run.bbox_w:.4f}, {run.bbox_s:.4f}, {run.bbox_e:.4f}, {run.bbox_n:.4f}",
        f"площадь: {aggregates.get('area_km2')} км²   средняя p: {aggregates.get('mean_p')}",
        f"доля p>0.5: {aggregates.get('fraction_p_gt_0_5')}   "
        f"доля p>0.8: {aggregates.get('fraction_p_gt_0_8')}",
    ]
    for line in meta:
        c.drawString(20 * mm, y, line)
        y -= 6 * mm

    # Карта (растр вероятностей).
    if png_path.exists():
        img = ImageReader(str(png_path))
        iw, ih = img.getSize()
        max_w = 170 * mm
        scale = min(max_w / iw, (y - 45 * mm) / ih)
        dw, dh = iw * scale, ih * scale
        y -= dh + 5 * mm
        c.drawImage(img, 20 * mm, y, width=dw, height=dh, preserveAspectRatio=True, mask="auto")

    # Легенда Viridis (5 градаций).
    y -= 12 * mm
    c.setFont(font, 9)
    c.drawString(20 * mm, y, "Легенда (вероятность затопления):")
    x = 20 * mm
    y -= 6 * mm
    for label, hexc in LEGEND:
        c.setFillColor(_hex(hexc))
        c.rect(x, y, 8 * mm, 5 * mm, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(x, y - 4 * mm, label)
        x += 30 * mm
    c.showPage()
    c.save()


def _hex(h: str):
    from reportlab.lib.colors import HexColor

    return HexColor(h)


def build_export(session: Session, run_id: str) -> str:
    """Собирает exports/<run_id>.zip и возвращает URL '/exports/<run_id>.zip'. FR-17."""
    run = get_run(session, run_id)
    if run is None:
        raise RunNotFound(run_id)

    rd = _run_dir(run_id)
    tif = rd / "prediction.tif"
    png = rd / "prediction.png"
    aggregates = json.loads(run.aggregates_json) if run.aggregates_json else {}

    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = rd / "aggregates.geojson"
    geojson_path.write_text(
        json.dumps(_aggregates_geojson(run, aggregates), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pdf_path = rd / "report.pdf"
    _build_pdf(pdf_path, run, aggregates, png)

    zip_path = settings.exports_dir / f"{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if tif.exists():
            zf.write(tif, "prediction.tif")
        zf.write(geojson_path, "aggregates.geojson")
        zf.write(pdf_path, "report.pdf")

    session.add(Export(run_id=run_id, zip_path=str(zip_path)))
    session.commit()
    return f"/exports/{run_id}.zip"
