"""Препроцессинг и нарезка тайлов (FR-2, SRS §6.2, §6.4).

Шаги: приведение источников к целевой сетке (см. grid.TargetGrid), построение
производных рельефа (уклон, экспозиция, кривизна, TWI, расстояние до воды),
стек многоканального растра признаков, нарезка на тайлы 256×256 и
географический split train/val/test (непересекающиеся по easting полосы).

Производные рельефа, геометрия тайлинга и split — чистый numpy/scipy (тест на
хосте). TWI (flow accumulation, pysheds), чтение/запись растров (rasterio) и
index.parquet (pandas) — ленивые импорты, выполняются под ``make data``/docker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ──────────────── производные рельефа (чистый numpy) ────────────────


def slope_aspect_deg(dem: np.ndarray, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Уклон (град) и экспозиция (град, 0..360, от севера по часовой)."""
    dem = np.asarray(dem, dtype="float64")
    gy, gx = np.gradient(dem, res_m)  # gy — вдоль строк (на юг), gx — вдоль столбцов
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.degrees(np.arctan2(gy, -gx))
    aspect = np.where(aspect < 0, aspect + 360.0, aspect)
    return slope.astype("float32"), aspect.astype("float32")


def curvature(dem: np.ndarray, res_m: float) -> np.ndarray:
    """Лапласова кривизна поверхности (вторые производные)."""
    dem = np.asarray(dem, dtype="float64")
    gy, gx = np.gradient(dem, res_m)
    gyy, _ = np.gradient(gy, res_m)
    _, gxx = np.gradient(gx, res_m)
    return (gxx + gyy).astype("float32")


def distance_to_water(water_mask: np.ndarray, res_m: float) -> np.ndarray:
    """Евклидово расстояние (м) до ближайшего водного пикселя."""
    from scipy import ndimage

    mask = np.asarray(water_mask, dtype=bool)
    if not mask.any():
        # воды нет — расстояние не определено; возвращаем большую константу
        return np.full(mask.shape, np.float32(1e6))
    return (ndimage.distance_transform_edt(~mask) * res_m).astype("float32")


def flow_accumulation_d8(dem: np.ndarray, res_m: float) -> np.ndarray:
    """D8 flow accumulation: площадь сбора (м²) на ячейку.

    Каждая ячейка стекает в соседа с наибольшим уклоном вниз (D8). Накопление
    считается обходом ячеек по убыванию высоты (O(N)). Реализация на numpy без
    внешних гидро-библиотек — выполняется один раз офлайн в ``make data``.
    """
    dem = np.asarray(dem, dtype="float64")
    h, w = dem.shape
    cell_area = res_m * res_m

    # смещения 8 соседей и расстояния до них
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dists = [np.sqrt(2), 1.0, np.sqrt(2), 1.0, 1.0, np.sqrt(2), 1.0, np.sqrt(2)]

    flat = dem.ravel()
    receiver = np.full(h * w, -1, dtype=np.int64)  # сток каждой ячейки (плоский индекс)
    best_slope = np.zeros(h * w, dtype="float64")

    rows, cols = np.mgrid[0:h, 0:w]
    rf, cf = rows.ravel(), cols.ravel()
    for (dr, dc), dd in zip(nbrs, dists, strict=True):
        nr, nc = rows + dr, cols + dc
        valid = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        nidx = np.where(valid, (nr * w + nc), 0).ravel()
        drop = (dem.ravel() - flat[nidx]) / (dd * res_m)  # уклон вниз к соседу
        better = valid.ravel() & (drop > best_slope)
        receiver[better] = nidx[better]
        best_slope[better] = drop[better]

    # накопление: обход по убыванию высоты, проталкивание площади вниз.
    # Списки python вместо numpy-скаляров — цикл на млн ячеек в разы быстрее.
    order = np.argsort(flat)[::-1].tolist()
    recv = receiver.tolist()
    acc = [cell_area] * (h * w)
    for i in order:
        r = recv[i]
        if r >= 0:
            acc[r] += acc[i]
    del rf, cf
    return np.asarray(acc, dtype="float32").reshape(h, w)


def twi(dem: np.ndarray, res_m: float, *, slope_floor_deg: float = 0.1) -> np.ndarray:
    """Topographic Wetness Index: ln(a / tan(β)).

    a — удельная площадь сбора (flow accumulation / ширину ячейки), β — локальный
    уклон. Высокий TWI = ложбины/поймы (склонны к накоплению воды), низкий —
    гребни. Классический признак подверженности затоплению.
    """
    acc = flow_accumulation_d8(dem, res_m)
    slope, _ = slope_aspect_deg(dem, res_m)
    tan_b = np.tan(np.radians(np.maximum(slope, slope_floor_deg)))
    spec_area = acc / res_m  # удельная площадь на ед. длины контура
    return np.log(spec_area / tan_b).astype("float32")


# ──────────────── тайлинг и split (чистая геометрия) ────────────────


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    row_off: int
    col_off: int
    row_idx: int  # индекс по сетке тайлов (для отладки)
    col_idx: int  # индекс по easting — основа географического split


def tile_windows(width: int, height: int, tile_px: int, overlap_px: int = 0) -> list[TileSpec]:
    """Полные тайлы ``tile_px × tile_px``, покрывающие растр (остаток отбрасывается).

    overlap_px=0 → стык в стык (режим трейна, SRS §6.4).
    """
    step = tile_px - overlap_px
    if step <= 0:
        raise ValueError("overlap_px должен быть меньше tile_px")
    specs: list[TileSpec] = []
    rows = range(0, height - tile_px + 1, step)
    cols = range(0, width - tile_px + 1, step)
    for ri, row_off in enumerate(rows):
        for ci, col_off in enumerate(cols):
            specs.append(
                TileSpec(
                    tile_id=f"t_r{ri:03d}c{ci:03d}",
                    row_off=row_off,
                    col_off=col_off,
                    row_idx=ri,
                    col_idx=ci,
                )
            )
    return specs


def assign_geographic_splits(
    specs: list[TileSpec], train: float, val: float, test: float
) -> dict[str, str]:
    """Географический split по полосам easting (col_idx) — без утечки между сетами.

    Левые полосы → train, средние → val, правые → test. Полосы по col_idx не
    пересекаются, поэтому bbox train/val/test заведомо не накладываются.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("доли split должны суммироваться в 1.0")
    cols = sorted({s.col_idx for s in specs})
    n = len(cols)
    n_train = round(n * train)
    n_val = round(n * val)
    by_band: dict[int, str] = {}
    for k, c in enumerate(cols):
        if k < n_train:
            by_band[c] = "train"
        elif k < n_train + n_val:
            by_band[c] = "val"
        else:
            by_band[c] = "test"
    return {s.tile_id: by_band[s.col_idx] for s in specs}


def splits_are_disjoint(specs: list[TileSpec], assignment: dict[str, str]) -> bool:
    """Проверка FR-2: множества col_idx у train/val/test не пересекаются."""
    bands: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    by_id = {s.tile_id: s for s in specs}
    for tile_id, split in assignment.items():
        bands[split].add(by_id[tile_id].col_idx)
    return (
        not (bands["train"] & bands["val"])
        and not (bands["train"] & bands["test"])
        and not (bands["val"] & bands["test"])
    )


# ──────────────── оркестрация (rasterio/pandas/pysheds — лениво) ────────────────


def cut_tiles(
    stack_path,
    out_dir,
    tile_px: int,
    overlap_px: int = 0,
):
    """Нарезать многоканальный растр на GeoTIFF-тайлы. Возвращает список TileSpec."""
    import rasterio
    from rasterio.windows import Window

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(stack_path) as src:
        specs = tile_windows(src.width, src.height, tile_px, overlap_px)
        base = src.profile
        for spec in specs:
            window = Window(spec.col_off, spec.row_off, tile_px, tile_px)
            data = src.read(window=window)
            profile = base.copy()
            profile.update(
                width=tile_px,
                height=tile_px,
                transform=src.window_transform(window),
            )
            with rasterio.open(out_dir / f"{spec.tile_id}.tif", "w", **profile) as dst:
                dst.write(data)
    return specs


def reproject_to_grid(src_path, grid, resampling_name: str = "bilinear") -> np.ndarray:
    """Прочитать растр и привести к целевой сетке (CRS+разрешение). Возвращает 2D-массив."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    resampling = getattr(Resampling, resampling_name)
    dst = np.zeros(grid.shape, dtype="float32")
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=resampling,
        )
    return dst


def tiles_overlapping_grid(src_paths, grid) -> list[Path]:
    """Тайлы, чьи границы пересекают целевую сетку (важно для общего raw на 2+ региона)."""
    import rasterio
    from rasterio.warp import transform_bounds

    gx0, gy0, gx1, gy1 = grid.bounds()
    w, s, e, n = transform_bounds(grid.crs, "EPSG:4326", gx0, gy0, gx1, gy1)
    keep: list[Path] = []
    for p in (Path(x) for x in src_paths):
        try:
            with rasterio.open(p) as src:
                tw, ts, te, tn = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        except Exception:
            continue
        if not (te < w or tw > e or tn < s or ts > n):
            keep.append(p)
    return keep


def mosaic_source_to_file(src_paths, grid, out_path) -> Path:
    """Склеить ВСЕ тайлы источника, пересекающие сетку, в один GeoTIFF (CRS источника,
    обрезка до bbox сетки), сохранив профиль/nodata.

    Чинит мультитайловые источники (DEM/WorldCover/JRC покрываются несколькими 1°-тайлами):
    раньше препроцесс брал ОДИН тайл (`_first`) → признаки рельефа были пусты вне его
    покрытия (для второго региона при общем raw — пусты полностью). См. pipeline.run_preprocess.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds

    keep = tiles_overlapping_grid(src_paths, grid)
    if not keep:
        gx0, gy0, gx1, gy1 = grid.bounds()
        wsen = transform_bounds(grid.crs, "EPSG:4326", gx0, gy0, gx1, gy1)
        raise FileNotFoundError(
            f"ни один тайл источника не пересекает сетку (bbox WGS84 {[round(x, 2) for x in wsen]})"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    srcs = [rasterio.open(p) for p in keep]
    try:
        gx0, gy0, gx1, gy1 = grid.bounds()
        w, s, e, n = transform_bounds(grid.crs, "EPSG:4326", gx0, gy0, gx1, gy1)
        bounds_src = transform_bounds("EPSG:4326", srcs[0].crs, w, s, e, n)
        mosaic, mtransform = merge(srcs, bounds=bounds_src)
        profile = srcs[0].profile.copy()
    finally:
        for sd in srcs:
            sd.close()

    profile.update(
        height=mosaic.shape[1], width=mosaic.shape[2], transform=mtransform, compress="deflate"
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    return out_path


def write_feature_stack(bands: dict[str, np.ndarray], grid, out_path) -> Path:
    """Записать многоканальный GeoTIFF признаков (имена каналов — в описании)."""
    import rasterio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(bands)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": len(names),
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, name in enumerate(names, start=1):
            dst.write(bands[name].astype("float32"), i)
            dst.set_band_description(i, name)
    return out_path


def write_index_parquet(specs, assignment, grid, tile_px, event_ids, out_path) -> None:
    """Записать index.parquet: tile_id, bbox_wgs84, event_ids, split (SRS §6.2)."""
    import pandas as pd
    from pyproj import Transformer
    from rasterio.windows import Window, bounds

    to_wgs = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    rows = []
    for s in specs:
        window = Window(s.col_off, s.row_off, tile_px, tile_px)
        left, bottom, right, top = bounds(window, grid.transform)
        # углы прямоугольника в целевом CRS → охватывающий bbox в WGS84
        xs, ys = to_wgs.transform([left, right, left, right], [bottom, bottom, top, top])
        rows.append(
            {
                "tile_id": s.tile_id,
                "split": assignment[s.tile_id],
                "event_ids": ",".join(event_ids),
                "min_lon": min(xs),
                "min_lat": min(ys),
                "max_lon": max(xs),
                "max_lat": max(ys),
            }
        )
    pd.DataFrame(rows).to_parquet(out_path, index=False)
