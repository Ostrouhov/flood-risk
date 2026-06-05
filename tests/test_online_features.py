"""Тесты онлайн-инференса по произвольному bbox (глобальный режим).

Чистая логика (UTM-зона, кап размера, сетка, селектор источника) — на хосте без сети.
Реальная сборка признаков (загрузка COG через PC STAC) — под @requires_network, вне CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from floodrisk.data.grid import TargetGrid, utm_epsg_for_bbox
from floodrisk.inference import online_features as online
from floodrisk.inference import service
from floodrisk.inference.features import InvalidBBox

# bbox в зоне покрытия пилота (Тулун) и вне её (Москва).
IN_COVERAGE = [100.64, 54.54, 100.89, 54.71]
OUT_COVERAGE = [37.5, 55.6, 37.7, 55.8]

# Тесты «в покрытии» требуют локальной мозаики признаков (data/processed/v1, gitignored).
# В CI её нет → пропускаем (логика селектора вне покрытия проверяется отдельными тестами).
_HAS_TULUN_MOSAIC = service.feature_stack_path("v1").exists()
_requires_mosaic = pytest.mark.skipif(
    not _HAS_TULUN_MOSAIC, reason="нет локальной мозаики data/processed/v1 (например в CI)"
)


# ──────────────── UTM-зона (чистая функция) ────────────────


@pytest.mark.parametrize(
    "bbox,expected",
    [
        ([99.9, 54.05, 101.5, 55.15], "EPSG:32647"),  # Тулун — совпадает с manifest target_crs
        ([-0.2, 51.4, 0.1, 51.6], "EPSG:32630"),  # Лондон, зона 30N
        ([151.0, -34.0, 151.3, -33.8], "EPSG:32756"),  # Сидней, зона 56S
    ],
)
def test_utm_epsg_for_bbox(bbox, expected):
    assert utm_epsg_for_bbox(bbox) == expected


def test_grid_from_bbox_sane():
    grid = TargetGrid.from_bbox([100.6, 54.5, 100.7, 54.6], 30.0, "EPSG:32647")
    assert grid.width > 0 and grid.height > 0
    assert grid.crs == "EPSG:32647"
    assert grid.resolution_m == 30.0


# ──────────────── валидация bbox онлайн-режима ────────────────


def test_build_feature_window_rejects_inverted_bbox():
    with pytest.raises(InvalidBBox):
        online.build_feature_window([10.0, 50.0, 9.0, 49.0], cache_dir="/tmp/none")


def test_build_feature_window_caps_huge_bbox():
    # ~5° по стороне ≫ MAX_BBOX_KM → BBoxTooLarge ДО любых сетевых вызовов
    with pytest.raises(online.BBoxTooLarge):
        online.build_feature_window([10.0, 50.0, 15.0, 55.0], cache_dir="/tmp/none")


def test_bbox_sides_km_positive():
    w, h = online._bbox_sides_km(IN_COVERAGE)
    assert 5 < w < 40 and 5 < h < 40


def test_buffer_expands_bbox():
    buf = online._buffer_bbox([100.0, 54.0, 100.1, 54.1], 2000.0)
    assert buf[0] < 100.0 and buf[1] < 54.0 and buf[2] > 100.1 and buf[3] > 54.1


# ──────────────── селектор источника ────────────────


@_requires_mosaic
def test_resolve_features_mosaic_in_coverage():
    raw, _t, _crs, used, region = service._resolve_features(IN_COVERAGE, "v1", "mosaic")
    assert used == "mosaic"
    assert region == "v1"  # bbox внутри тулунской мозаики
    assert raw.shape[0] == 7


def test_resolve_features_auto_falls_back_to_online(monkeypatch):
    called = {}

    def fake_build(bbox, *, cache_dir, catalog=None):
        called["bbox"] = bbox
        return np.zeros((7, 40, 40), dtype="float32"), "TRANSFORM", "EPSG:32637"

    monkeypatch.setattr(online, "build_feature_window", fake_build)
    _raw, _t, crs, used, region = service._resolve_features(OUT_COVERAGE, "v1", "auto")
    assert used == "online"
    assert region is None  # онлайн-сборка — не региональная мозаика
    assert crs == "EPSG:32637"
    assert called["bbox"] == OUT_COVERAGE


@_requires_mosaic
def test_resolve_features_auto_uses_mosaic_in_coverage(monkeypatch):
    # в покрытии auto НЕ должен трогать онлайн-сборку
    def boom(*a, **k):
        raise AssertionError("online не должен вызываться в покрытии")

    monkeypatch.setattr(online, "build_feature_window", boom)
    _raw, _t, _crs, used, _region = service._resolve_features(IN_COVERAGE, "v1", "auto")
    assert used == "mosaic"


# ──────────────── склейка кэша по bbox (без сети, регрессия зависания) ────────────────


def _write_tile(path, bounds, *, size=16, val=1.0):
    """Крошечный GeoTIFF EPSG:4326 на заданных границах [W,S,E,N]."""
    import rasterio
    from rasterio.transform import from_bounds as tfb

    w, s, e, n = bounds
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=tfb(w, s, e, n, size, size),
    ) as dst:
        dst.write(np.full((size, size), val, dtype="float32"), 1)


def test_tiles_intersecting_filters_other_regions(tmp_path):
    coll = tmp_path / "cop-dem-glo-30"
    _write_tile(coll / "moscow.tif", [37.4, 55.5, 37.8, 55.9])
    _write_tile(coll / "krasnoyarsk.tif", [92.7, 55.9, 93.1, 56.2])
    inter = online._tiles_intersecting(sorted(coll.glob("*.tif")), [37.5, 55.6, 37.7, 55.8])
    assert [p.name for p in inter] == ["moscow.tif"]


def test_merge_tiles_bounded_by_bbox(tmp_path):
    # Регрессия: кэш с тайлами разных регионов не должен давать глобальную склейку
    # (иначе reproject захлёбывается и онлайн-режим зависает).
    import rasterio

    coll = tmp_path / "cop-dem-glo-30"
    _write_tile(coll / "moscow.tif", [37.4, 55.5, 37.8, 55.9])
    _write_tile(coll / "krasnoyarsk.tif", [92.7, 55.9, 93.1, 56.2])
    out = online._merge_tiles_to_file(coll, tmp_path / "merged.tif", [37.5, 55.6, 37.7, 55.8])
    with rasterio.open(out) as src:
        b = src.bounds
    # склейка ограничена окном Москвы, не тянется до Красноярска (~92°E)
    assert b.left >= 37.3 and b.right <= 37.9
    assert b.bottom >= 55.4 and b.top <= 56.0


# ──────────────── реальная сборка (сеть) ────────────────


@pytest.mark.requires_network
def test_build_feature_window_live(tmp_path):
    raw, _transform, crs = online.build_feature_window(OUT_COVERAGE, cache_dir=tmp_path)
    assert raw.shape[0] == 7
    assert raw.shape[1] >= 32 and raw.shape[2] >= 32
    assert np.isfinite(raw).all()
    assert crs.startswith("EPSG:327") or crs.startswith("EPSG:326")
