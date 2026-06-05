"""Горячие точки риска: топ-N связных кластеров высокой вероятности затопления.

Карта подверженности показывает «где мокро», но не указывает на конкретные зоны.
Здесь по растру предсказания ``runs/<id>/prediction.tif`` находим связные компоненты
выше порога (``scipy.ndimage.label``), ранжируем их по «значимости» (площадь × средняя
вероятность) и отдаём топ-N с центроидом (WGS84) и охватывающим bbox для зума на карте.

Read-only над готовым растром — модель не вызывается; быстро (мс).
"""

from __future__ import annotations

import numpy as np
import rasterio

from floodrisk.inference.raster import WGS84
from floodrisk.settings import settings


def _centroid_lonlat(rows: np.ndarray, cols: np.ndarray, transform, crs) -> tuple[float, float]:
    """Центроид компоненты (пиксельные row/col) → (lon, lat) в EPSG:4326."""
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform

    x, y = xy(transform, float(rows.mean()), float(cols.mean()))
    lons, lats = warp_transform(crs, WGS84, [x], [y])
    return float(lons[0]), float(lats[0])


def _component_bounds_wgs84(rows: np.ndarray, cols: np.ndarray, transform, crs) -> list[float]:
    """Охватывающий bbox компоненты → [S, W, N, E] в EPSG:4326 (для fitBounds)."""
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    win_transform = window_transform(Window(cmin, rmin, w, h), transform)
    left, bottom, right, top = array_bounds(h, w, win_transform)
    w4, s4, e4, n4 = transform_bounds(crs, WGS84, left, bottom, right, top)
    return [float(s4), float(w4), float(n4), float(e4)]


def find_hotspots(
    run_id: str,
    *,
    threshold: float = 0.5,
    top_n: int = 5,
    min_area_px: int = 4,
) -> dict:
    """Топ-N кластеров риска для запуска. Бросает FileNotFoundError, если нет растра.

    Бинаризует prob ≥ ``threshold``, маркирует связные компоненты (8-связность),
    отбрасывает мельче ``min_area_px``, ранжирует по ``area_km2 × mean_p`` и берёт
    ``top_n``. Возвращает dict для API:

        {threshold, count, hotspots: [{rank, lat, lon, area_km2, mean_p, max_p,
                                       bounds_wgs84:[S,W,N,E]}]}

    ``count`` — число кластеров ВЫШЕ фильтра площади (может быть > len(hotspots)).
    """
    from scipy import ndimage

    tif_path = settings.runs_dir / run_id / "prediction.tif"
    if not tif_path.exists():
        raise FileNotFoundError(run_id)

    with rasterio.open(tif_path) as src:
        prob = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs

    res_m = abs(transform.a)
    px_km2 = (res_m * res_m) / 1e6

    mask = np.nan_to_num(prob, nan=0.0) >= threshold
    structure = np.ones((3, 3), dtype=int)  # 8-связность
    labels, n = ndimage.label(mask, structure=structure)

    clusters: list[dict] = []
    for lab in range(1, n + 1):
        sel = labels == lab
        area_px = int(np.count_nonzero(sel))
        if area_px < min_area_px:
            continue
        rows, cols = np.nonzero(sel)
        vals = prob[sel]
        lon, lat = _centroid_lonlat(rows, cols, transform, crs)
        clusters.append(
            {
                "area_km2": round(area_px * px_km2, 4),
                "mean_p": round(float(vals.mean()), 4),
                "max_p": round(float(vals.max()), 4),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "bounds_wgs84": _component_bounds_wgs84(rows, cols, transform, crs),
                "_score": area_px * px_km2 * float(vals.mean()),
            }
        )

    clusters.sort(key=lambda c: c["_score"], reverse=True)
    top = clusters[:top_n]
    for i, c in enumerate(top, start=1):
        c["rank"] = i
        del c["_score"]

    return {
        "threshold": round(float(threshold), 4),
        "count": len(clusters),
        "hotspots": top,
    }
