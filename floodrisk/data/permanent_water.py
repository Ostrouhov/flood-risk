"""Маска постоянной воды (FR-3, SRS §11.1, OQ-2).

Постоянная вода = JRC Global Surface Water (occurrence ≥ порога) ∪ водные
объекты OSM. Эта маска вычитается из детектированного Sentinel-1 разлива, чтобы
в лейблах осталось только паводковое затопление, а не реки/озёра/водохранилища.

``combine_permanent_water`` — чистое numpy-ядро (тестируется на хосте).
``build_permanent_water`` — гео-оркестрация (rasterio/geopandas).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floodrisk.data.grid import TargetGrid


def combine_permanent_water(
    occurrence_pct: np.ndarray,
    threshold_pct: float,
    osm_water: np.ndarray | None = None,
    nodata: float | None = None,
) -> np.ndarray:
    """Бинарная маска постоянной воды по occurrence-растру и (опц.) OSM-маске.

    Args:
        occurrence_pct: JRC GSW occurrence, доля времени с водой в % (0..100).
        threshold_pct: порог occurrence для «постоянной» воды.
        osm_water: bool-маска воды из OSM той же формы, либо None.
        nodata: значение occurrence, трактуемое как «нет данных» (не вода).
    """
    occ = np.asarray(occurrence_pct, dtype="float32")
    valid = np.ones(occ.shape, dtype=bool)
    if nodata is not None:
        valid &= occ != nodata
    valid &= ~np.isnan(occ)

    mask = (occ >= threshold_pct) & valid
    if osm_water is not None:
        osm = np.asarray(osm_water, dtype=bool)
        if osm.shape != mask.shape:
            raise ValueError(f"форма OSM-маски {osm.shape} != occurrence {mask.shape}")
        mask |= osm
    return mask


def overpass_water_geometries(osm_path: str | Path, dst_crs: str) -> list:
    """Разобрать Overpass JSON (out geom) → список shapely-геометрий в dst_crs.

    fetch сохраняет ответ Overpass в формате [out:json] — это НЕ GeoJSON, поэтому
    geopandas.read_file его не берёт. Парсим вручную: way с площадными тегами
    (natural=water, landuse=reservoir) или замкнутые → Polygon; линейные
    (waterway) → LineString. Координаты репроецируем в целевой CRS.
    """
    import json

    from pyproj import Transformer
    from shapely.geometry import LineString, Polygon
    from shapely.ops import transform as shp_transform

    data = json.loads(Path(osm_path).read_text(encoding="utf-8"))
    tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    geoms = []
    for el in data.get("elements", []):
        geom_pts = el.get("geometry")
        if not geom_pts:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom_pts if p.get("lon") is not None]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        is_area = (
            tags.get("natural") == "water"
            or tags.get("landuse") == "reservoir"
            or coords[0] == coords[-1]
        )
        try:
            geom = Polygon(coords) if is_area and len(coords) >= 4 else LineString(coords)
            if not geom.is_valid:
                geom = geom.buffer(0)
            geoms.append(shp_transform(lambda x, y: tr.transform(x, y), geom))
        except (ValueError, TypeError):
            continue
    return [g for g in geoms if g is not None and not g.is_empty]


def rasterize_osm_water(osm_path: str | Path, grid: TargetGrid) -> np.ndarray:
    """Растеризовать водные объекты OSM (Overpass JSON) на целевую сетку → bool-маска."""
    from rasterio.features import rasterize

    geoms = overpass_water_geometries(osm_path, grid.crs)
    if not geoms:
        return np.zeros(grid.shape, dtype=bool)
    burned = rasterize(
        ((g, 1) for g in geoms),
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return burned.astype(bool)


def build_permanent_water(
    grid: TargetGrid,
    jrc_occurrence_path: str | Path,
    osm_path: str | Path | None,
    threshold_pct: float,
    out_path: str | Path,
) -> Path:
    """Собрать маску постоянной воды и записать в GeoTIFF на целевой сетке."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # JRC occurrence → ресемпл на целевую сетку
    occ = np.zeros(grid.shape, dtype="float32")
    with rasterio.open(jrc_occurrence_path) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=occ,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.nearest,
        )

    osm_mask = rasterize_osm_water(osm_path, grid) if osm_path is not None else None
    mask = combine_permanent_water(occ, threshold_pct, osm_mask, nodata=src_nodata)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
        "nodata": 255,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)
    return out_path
