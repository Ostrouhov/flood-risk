"""Оркестрация data-пайплайна (FR-1…FR-4): склейка модулей и раскладка файлов.

Тонкий слой над config/fetch/preprocess/labels_s1/permanent_water/manifest.
Сетевые и GDAL-операции — рассчитаны на ``make data`` (docker, SRS §14.5);
ленивые импорты держат модуль загружаемым в тестах чистой логики.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from floodrisk.data import fetch, manifest
from floodrisk.data.config import DataConfig, load_data_config
from floodrisk.data.grid import TargetGrid
from floodrisk.data.manifest import DATA_DIR


def processed_dir(cfg: DataConfig) -> Path:
    return DATA_DIR / "processed" / cfg.dataset_version


def _first(raw_dir: Path, sub: str, pattern: str = "*.tif") -> Path:
    matches = sorted((raw_dir / sub).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"нет файлов {sub}/{pattern} — сначала `data fetch`")
    return matches[0]


def run_fetch(cfg: DataConfig | None = None, *, skip_era5: bool = False) -> dict:
    """FR-1: скачать сырые источники для пилота, записать провенанс."""
    cfg = cfg or load_data_config()
    return fetch.fetch_all(cfg, skip_era5=skip_era5)


def run_preprocess(cfg: DataConfig | None = None) -> Path:
    """FR-2: производные рельефа, стек признаков, тайлинг 256×256, index.parquet."""
    from floodrisk.data import preprocess as pp
    from floodrisk.data.permanent_water import build_permanent_water

    cfg = cfg or load_data_config()
    grid = TargetGrid.from_config(cfg)
    raw = fetch.RAW_DIR
    out = processed_dir(cfg)
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    dem = pp.reproject_to_grid(_first(raw, "cop-dem-glo-30"), grid, "bilinear")
    slope, aspect = pp.slope_aspect_deg(dem, grid.resolution_m)
    curv = pp.curvature(dem, grid.resolution_m)
    twi = pp.twi(dem, grid.resolution_m)
    worldcover = pp.reproject_to_grid(_first(raw, "esa-worldcover"), grid, "nearest")

    osm = raw / "osm" / "water.osm.json"
    pw_path = build_permanent_water(
        grid,
        _first(raw, "jrc-gsw"),
        osm if osm.exists() else None,
        cfg.labels_s1.permanent_water_occurrence_pct,
        feat_dir / "permanent_water.tif",
    )
    import rasterio

    with rasterio.open(pw_path) as src:
        permanent = src.read(1).astype(bool)
    dist_water = pp.distance_to_water(permanent, grid.resolution_m)

    bands = {
        "dem": dem,
        "slope": slope,
        "aspect": aspect,
        "curvature": curv,
        "twi": twi,
        "worldcover": worldcover,
        "dist_to_water": dist_water,
    }
    stack = pp.write_feature_stack(bands, grid, feat_dir / "stack.tif")
    specs = pp.cut_tiles(stack, out / "tiles", cfg.tile_size_px, cfg.tile_overlap_px)
    assignment = pp.assign_geographic_splits(
        specs, cfg.splits.train, cfg.splits.val, cfg.splits.test
    )
    pp.write_index_parquet(
        specs,
        assignment,
        grid,
        cfg.tile_size_px,
        [ev.event_id for ev in cfg.events],
        out / "index.parquet",
    )
    return out


def _s1_db_composite(scene_paths, grid, out_path) -> Path:
    """Репроецировать GRD VV-сцены на сетку, перевести в dB, взять медиану. → GeoTIFF."""
    import numpy as np
    import rasterio

    from floodrisk.data import preprocess as pp
    from floodrisk.data.labels_s1 import amplitude_to_db

    stack = []
    for sp in scene_paths:
        amp = pp.reproject_to_grid(sp, grid, "bilinear")
        stack.append(amplitude_to_db(amp))
    # nanmedian по стеку; all-NaN срезы (пиксели вне всех сцен) → NaN, это норма
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        composite = np.nanmedian(np.stack(stack, axis=0), axis=0).astype("float32")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
        "nodata": float("nan"),
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(composite, 1)
    return out_path


def run_label(cfg: DataConfig | None = None) -> dict[str, Path]:
    """FR-3: маски затопления S1 (GRD→dB композиты, Otsu+CD) по событиям, нарезка на тайлы."""
    from floodrisk.data import preprocess as pp
    from floodrisk.data.labels_s1 import build_flood_mask

    cfg = cfg or load_data_config()
    grid = TargetGrid.from_config(cfg)
    raw = fetch.RAW_DIR
    out = processed_dir(cfg)
    feat_dir = out / "features"
    pw_path = feat_dir / "permanent_water.tif"
    slope_path = feat_dir / "slope.tif"  # опционально, если выгружен отдельно

    results: dict[str, Path] = {}
    for ev in cfg.events:
        s1_dir = raw / "sentinel1" / ev.event_id
        flood_vv = sorted((s1_dir / "flood").glob("*_vv.tif"))
        dry_vv = sorted((s1_dir / "dry").glob("*_vv.tif"))
        if not flood_vv:
            raise FileNotFoundError(f"нет S1 VV (flood) для {ev.event_id} — сначала `data fetch`")

        lab_dir = out / "labels" / ev.event_id
        flood_db = _s1_db_composite(flood_vv, grid, lab_dir / "flood_vv_db.tif")
        dry_db = _s1_db_composite(dry_vv, grid, lab_dir / "dry_vv_db.tif") if dry_vv else None

        mask_path, _thr = build_flood_mask(
            flood_db,
            lab_dir / "flood_mask.tif",
            dry_db_path=dry_db,
            permanent_water_path=pw_path if pw_path.exists() else None,
            slope_path=slope_path if slope_path.exists() else None,
            max_slope_deg=cfg.labels_s1.slope_mask_deg,
            min_blob_px=cfg.labels_s1.min_blob_px,
        )
        pp.cut_tiles(mask_path, lab_dir, cfg.tile_size_px, 0)
        results[ev.event_id] = mask_path
    return results


def run_manifest(cfg: DataConfig | None = None) -> Path:
    """FR-4: собрать data/manifest.yaml из обработанных тайлов и провенанса."""
    cfg = cfg or load_data_config()
    out = processed_dir(cfg)
    provenance = fetch.load_provenance()
    manifest.build_manifest(cfg, out, sources_provenance=provenance)
    return manifest.DEFAULT_MANIFEST_PATH


def run_verify(manifest_path: Path | None = None) -> int:
    """FR-4: сверить файлы с манифестом. Exit-code: 0 ок, 1 расхождения."""
    problems = manifest.verify_manifest(manifest_path or manifest.DEFAULT_MANIFEST_PATH)
    if problems:
        for p in problems:
            print(f"[verify-data] РАСХОЖДЕНИЕ: {p}")
        return 1
    print("[verify-data] OK: файлы совпадают с манифестом")
    return 0
