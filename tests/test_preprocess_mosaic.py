"""Регресс на баг «препроцесс брал ОДИН тайл вместо мозаики».

Источники (DEM/WorldCover/JRC) покрываются несколькими 1°-тайлами; раньше `_first`
давал только один → признаки рельефа пусты вне его покрытия. mosaic_source_to_file
должен склеивать ВСЕ пересекающие сетку тайлы и отбрасывать непересекающие.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from floodrisk.data import preprocess as pp
from floodrisk.data.grid import TargetGrid


def _wgs_tile(path, lon0, lat0, val, size=1.0, px=60):
    res = size / px
    transform = from_origin(lon0, lat0 + size, res, res)  # верхний-левый угол
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=px,
        width=px,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.full((px, px), val, "float32"), 1)


def test_mosaic_fills_multiple_tiles(tmp_path):
    # два соседних тайла по долготе: E99 (val 10), E100 (val 20) + далёкий (не пересекает)
    _wgs_tile(tmp_path / "a.tif", 99.0, 54.0, 10.0)
    _wgs_tile(tmp_path / "b.tif", 100.0, 54.0, 20.0)
    _wgs_tile(tmp_path / "far.tif", 10.0, 10.0, 99.0)
    grid = TargetGrid.from_bbox([99.4, 54.2, 100.6, 54.8], 300.0, "EPSG:32647")

    merged = pp.mosaic_source_to_file(
        [tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "far.tif"], grid, tmp_path / "m.tif"
    )
    arr = pp.reproject_to_grid(merged, grid, "nearest")
    vals = set(np.unique(arr).tolist())
    # ОБЕ константы присутствуют → мозаика обоих тайлов (а не один по _first)
    assert 10.0 in vals
    assert 20.0 in vals
    # непересекающий тайл не попал
    assert 99.0 not in vals
    # сетка реально заполнена (не пусто)
    assert float((arr != 0).mean()) > 0.9


def test_tiles_overlapping_grid_filters(tmp_path):
    _wgs_tile(tmp_path / "a.tif", 99.0, 54.0, 1.0)
    _wgs_tile(tmp_path / "far.tif", 10.0, 10.0, 1.0)
    grid = TargetGrid.from_bbox([99.2, 54.2, 99.8, 54.8], 300.0, "EPSG:32647")
    keep = pp.tiles_overlapping_grid([tmp_path / "a.tif", tmp_path / "far.tif"], grid)
    assert [p.name for p in keep] == ["a.tif"]


def test_mosaic_raises_when_no_overlap(tmp_path):
    _wgs_tile(tmp_path / "far.tif", 10.0, 10.0, 1.0)
    grid = TargetGrid.from_bbox([99.2, 54.2, 99.8, 54.8], 300.0, "EPSG:32647")
    try:
        pp.mosaic_source_to_file([tmp_path / "far.tif"], grid, tmp_path / "m.tif")
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
