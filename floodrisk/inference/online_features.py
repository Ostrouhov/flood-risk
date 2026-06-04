"""Онлайн-сборка признаков по произвольному bbox (глобальный режим инференса).

Вместо чтения из готовой мозаики `data/processed/v1/features/stack.tif` (только Тулун)
строит 7-канальный стек признаков на лету для любого bbox: тянет DEM / WorldCover /
JRC GSW по окну через Planetary Computer STAC (оконное чтение COG, кэш на диске),
считает производные рельефа и расстояние до воды на UTM-сетке зоны bbox.

Признаки — статический рельеф (Sentinel-1/ERA5 не нужны: они только для лейблов).
Качество вне обучающего региона НЕ валидировано (domain shift) — режим экспериментальный.

Переиспользует data-пайплайн: fetch.fetch_* (загрузка), preprocess.* (производные),
permanent_water.build_permanent_water (вода), grid.TargetGrid/utm_epsg_for_bbox (сетка).
Контракт выхода идентичен features.read_feature_window: (raw[7,H,W], transform, crs).
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

from floodrisk import feature_transform as ft
from floodrisk.inference.features import InvalidBBox

# ── параметры онлайн-сборки ──
RES_M = 30.0  # как при обучении датасета v1
MAX_BBOX_KM = 30.0  # кап на сторону bbox (latency/трафик); большее → BBoxTooLarge
BUFFER_M = 2000.0  # буфер вокруг bbox: TWI/flow-accum и производные на краях иначе врут
JRC_OCCURRENCE_PCT = 50.0  # порог occurrence JRC GSW для «постоянной воды»
# Таймаут/ретраи чтения COG через /vsicurl: при троттлинге Planetary Computer чтение тайла
# может зависнуть надолго → ограничиваем, чтобы вместо «вечного спиннера» был быстрый отказ.
GDAL_HTTP_OPTS = {
    "GDAL_HTTP_TIMEOUT": "25",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
}

WGS84 = "EPSG:4326"


class BBoxTooLarge(Exception):
    """bbox больше допустимого для онлайн-сборки (см. MAX_BBOX_KM)."""

    def __init__(self, side_km: float, limit_km: float = MAX_BBOX_KM):
        self.side_km = side_km
        self.limit_km = limit_km
        super().__init__(f"bbox_too_large: {side_km:.1f} км > {limit_km:.0f} км")


class OnlineFetchError(Exception):
    """Не удалось собрать признаки онлайн (нет сцен/обрыв загрузки источника)."""


def _bbox_sides_km(bbox: list[float]) -> tuple[float, float]:
    """Размеры bbox по сторонам в км (грубо, по центру широты)."""
    w, s, e, n = bbox
    lat_c = (s + n) / 2.0
    width_km = (e - w) * 111.32 * math.cos(math.radians(lat_c))
    height_km = (n - s) * 111.32
    return abs(width_km), abs(height_km)


def _buffer_bbox(bbox: list[float], meters: float) -> list[float]:
    """Расширить bbox на ~meters во все стороны (в градусах, по центру широты)."""
    w, s, e, n = bbox
    lat_c = (s + n) / 2.0
    dlat = meters / 111320.0
    dlon = meters / (111320.0 * max(math.cos(math.radians(lat_c)), 1e-6))
    return [w - dlon, s - dlat, e + dlon, n + dlat]


def _tiles_intersecting(tiles: list[Path], bbox_wgs84: list[float]) -> list[Path]:
    """Оставить тайлы, чьи границы пересекают bbox (WGS84). Кэш онлайна персистентный и
    копит тайлы со ВСЕХ прошлых запросов (разные регионы) — без фильтра склейка дала бы
    глобальную мозаику и катастрофически медленный reproject."""
    import rasterio
    from rasterio.warp import transform_bounds

    w, s, e, n = bbox_wgs84
    keep: list[Path] = []
    for t in tiles:
        try:
            with rasterio.open(t) as src:
                tw, ts, te, tn = transform_bounds(src.crs, WGS84, *src.bounds)
        except Exception:
            continue
        if not (te < w or tw > e or tn < s or ts > n):  # есть пересечение
            keep.append(t)
    return keep


def _merge_tiles_to_file(collection_dir: Path, out_path: Path, bbox_wgs84=None) -> Path:
    """Склеить кэшированные тайлы коллекции в один GeoTIFF.

    Если задан ``bbox_wgs84``: оставить только пересекающие его тайлы и обрезать склейку
    до его окна (``merge(bounds=...)``) — иначе разросшийся кэш других регионов сделал бы
    мозаику глобальной и reproject зависал бы.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds

    tiles = sorted(collection_dir.glob("*.tif"))
    if not tiles:
        raise OnlineFetchError(f"нет тайлов источника в {collection_dir}")
    if bbox_wgs84 is not None:
        filtered = _tiles_intersecting(tiles, bbox_wgs84)
        if filtered:
            tiles = filtered
    if len(tiles) == 1 and bbox_wgs84 is None:
        return tiles[0]

    srcs = [rasterio.open(t) for t in tiles]
    try:
        merge_kwargs = {}
        if bbox_wgs84 is not None:
            # bounds для merge — в CRS источников (все тайлы коллекции в одной CRS).
            merge_kwargs["bounds"] = transform_bounds(WGS84, srcs[0].crs, *bbox_wgs84)
        mosaic, mtransform = merge(srcs, **merge_kwargs)
        profile = srcs[0].profile.copy()
    finally:
        for s in srcs:
            s.close()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mtransform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    return out_path


def _reproject_source(
    collection_dir: Path, grid, resampling: str, merged_path: Path, bbox_wgs84=None
) -> np.ndarray:
    """Склеить тайлы коллекции (обрезкой до bbox) и репроецировать на целевую сетку → 2D."""
    from floodrisk.data import preprocess as pp

    merged = _merge_tiles_to_file(collection_dir, merged_path, bbox_wgs84)
    return pp.reproject_to_grid(merged, grid, resampling)


def build_feature_window(bbox_wgs84: list[float], *, cache_dir: Path, catalog=None):
    """Собрать признаки [7,H,W] для произвольного bbox (WGS84 [W,S,E,N]) на лету.

    Возвращает (raw[7,H,W], transform, crs) — тот же контракт, что
    features.read_feature_window. Бросает InvalidBBox / BBoxTooLarge / OnlineFetchError.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import transform as window_transform

    from floodrisk.data import fetch
    from floodrisk.data import preprocess as pp
    from floodrisk.data.grid import TargetGrid, utm_epsg_for_bbox
    from floodrisk.data.permanent_water import build_permanent_water

    w, s, e, n = bbox_wgs84
    if not (e > w and n > s):
        raise InvalidBBox(f"ожидается W<E и S<N, получено {bbox_wgs84}")
    side_w, side_h = _bbox_sides_km(bbox_wgs84)
    if max(side_w, side_h) > MAX_BBOX_KM:
        raise BBoxTooLarge(max(side_w, side_h))

    cache_dir = Path(cache_dir)
    buffered = _buffer_bbox(bbox_wgs84, BUFFER_M)
    crs = utm_epsg_for_bbox(bbox_wgs84)
    grid = TargetGrid.from_bbox(buffered, RES_M, crs)

    # Сеть с таймаутом: любой сбой/зависание загрузки → быстрый OnlineFetchError (не 500/вечный спиннер)
    try:
        with rasterio.Env(**GDAL_HTTP_OPTS):
            catalog = catalog or fetch._pc_catalog()
            fetch.fetch_dem(buffered, cache_dir, catalog)
            fetch.fetch_worldcover(buffered, cache_dir, catalog)
            fetch.fetch_jrc_gsw(buffered, cache_dir, catalog)
    except Exception as exc:  # нет сцен / таймаут чтения COG / троттлинг PC
        raise OnlineFetchError(f"не удалось загрузить тайлы рельефа: {exc}") from exc

    # Склейки/маска — во временную папку на запрос (не в постоянный кэш): иначе перезапись
    # одного и того же _merged/*.tif между запросами упирается в file-lock на Windows.
    # Скачанные тайлы кэшируются в cache_dir (дорогая часть), их не трогаем.
    work = Path(tempfile.mkdtemp(prefix="floodrisk_online_"))
    try:
        dem = _reproject_source(
            cache_dir / "cop-dem-glo-30", grid, "bilinear", work / "dem.tif", buffered
        )
        worldcover = _reproject_source(
            cache_dir / "esa-worldcover", grid, "nearest", work / "worldcover.tif", buffered
        )
        slope, aspect = pp.slope_aspect_deg(dem, RES_M)
        curv = pp.curvature(dem, RES_M)
        twi = pp.twi(dem, RES_M)

        jrc_merged = _merge_tiles_to_file(cache_dir / "jrc-gsw", work / "jrc.tif", buffered)
        pw_path = build_permanent_water(
            grid, jrc_merged, None, JRC_OCCURRENCE_PCT, work / "permanent_water.tif"
        )
        with rasterio.open(pw_path) as src:
            permanent = src.read(1) == 1
        dist_water = pp.distance_to_water(permanent, RES_M)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    bands = {
        "dem": dem,
        "slope": slope,
        "aspect": aspect,
        "curvature": curv,
        "twi": twi,
        "worldcover": worldcover,
        "dist_to_water": dist_water,
    }
    raw = np.stack([bands[name] for name in ft.RAW_ORDER], axis=0).astype("float32")

    # обрезать буфер до исходного bbox
    left, bottom, right, top = transform_bounds(WGS84, crs, w, s, e, n)
    win = from_bounds(left, bottom, right, top, grid.transform).round_offsets().round_lengths()
    win = win.intersection(Window(0, 0, grid.width, grid.height))
    r0, c0 = int(win.row_off), int(win.col_off)
    r1, c1 = r0 + int(win.height), c0 + int(win.width)
    raw = np.ascontiguousarray(raw[:, r0:r1, c0:c1])
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    transform = window_transform(win, grid.transform)
    return raw, transform, crs
