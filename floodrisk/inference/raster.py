"""Общие растровые утилиты инференса: запись GeoTIFF и репроекция в EPSG:4326.

Leaf-модуль (без импортов из floodrisk) — единый источник константы ``WGS84`` и
общего ядра репроекции для overlay'ев Leaflet. Раскраска (Viridis для вероятности,
сплошной цвет для маски) остаётся за вызывающей стороной — см.
``service._write_png_wgs84`` и ``validation._write_mask_png_wgs84``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject

WGS84 = "EPSG:4326"


def write_geotiff(path: Path, prob: np.ndarray, transform, crs) -> None:
    """Записать float32-растр [H,W] в одноканальный LZW-сжатый GeoTIFF (нативный CRS)."""
    h, w = prob.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(prob.astype("float32"), 1)


def reproject_to_wgs84(
    data: np.ndarray,
    transform,
    src_crs,
    *,
    resampling,
    fill: float,
    dst_nodata: float | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Репроецировать растр [H,W] из ``src_crs`` в EPSG:4326.

    Общее ядро для overlay'ев: считает целевой грид (``calculate_default_transform``),
    заполняет его ``fill`` и варпит ``data``. ``dst_nodata`` передаётся в ``reproject``
    только когда задан (для маски — не задаём, фон уже ``fill=0``; для вероятности —
    ``NaN``, чтобы вне покрытия оставался прозрачным).

    Возвращает ``(dst[Hd,Wd] float32, bounds [S, W, N, E])`` — bounds в порядке,
    который ждёт Leaflet ``imageOverlay``. Идентичная математика для вероятности и
    маски гарантирует совпадение их границ (нужно для шторки «реальность ↔ предсказание»).
    """
    h, w = data.shape
    left, bottom, right, top = array_bounds(h, w, transform)
    dst_transform, dw, dh = calculate_default_transform(
        src_crs, WGS84, w, h, left, bottom, right, top
    )
    dst = np.full((dh, dw), fill, dtype="float32")
    kwargs = {
        "source": data.astype("float32"),
        "destination": dst,
        "src_transform": transform,
        "src_crs": src_crs,
        "dst_transform": dst_transform,
        "dst_crs": WGS84,
        "resampling": resampling,
    }
    if dst_nodata is not None:
        kwargs["dst_nodata"] = dst_nodata
    reproject(**kwargs)

    w4, s4, e4, n4 = array_bounds(dh, dw, dst_transform)
    return dst, [float(s4), float(w4), float(n4), float(e4)]
