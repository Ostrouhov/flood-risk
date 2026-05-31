"""Признаки и тайлинг для инференса по произвольному bbox. См. SRS §6.4, §11.3.

Источник признаков — готовый мозаичный стек `data/processed/v1/features/stack.tif`
(7 каналов, EPSG:32647). Зона покрытия = границы стека. bbox вне покрытия → OutOfCoverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds, from_bounds

WGS84 = "EPSG:4326"


class OutOfCoverage(Exception):
    """bbox не помещается в зону покрытия признакового стека."""

    def __init__(self, coverage_wgs84: list[float]):
        self.coverage_wgs84 = coverage_wgs84
        super().__init__("bbox_out_of_coverage")


class InvalidBBox(Exception):
    """Геометрически некорректный bbox."""


def coverage_wgs84(stack_path: str | Path) -> list[float]:
    """Границы покрытия стека в EPSG:4326 [W, S, E, N]."""
    with rasterio.open(stack_path) as src:
        w, s, e, n = transform_bounds(src.crs, WGS84, *src.bounds)
    return [float(w), float(s), float(e), float(n)]


def read_feature_window(bbox_wgs84: list[float], stack_path: str | Path):
    """Читает окно признаков [7, H, W] под bbox (WGS84 [W,S,E,N]).

    Возвращает (features, transform, crs, bounds_native). Бросает InvalidBBox / OutOfCoverage.
    """
    w, s, e, n = bbox_wgs84
    if not (e > w and n > s):
        raise InvalidBBox(f"ожидается W<E и S<N, получено {bbox_wgs84}")

    with rasterio.open(stack_path) as src:
        left, bottom, right, top = transform_bounds(WGS84, src.crs, w, s, e, n)
        sb = src.bounds
        eps = src.res[0]  # допуск ~1 пиксель
        inside = (
            left >= sb.left - eps
            and bottom >= sb.bottom - eps
            and right <= sb.right + eps
            and top <= sb.top + eps
        )
        if not inside:
            cov = transform_bounds(src.crs, WGS84, *sb)
            raise OutOfCoverage([float(c) for c in cov])

        win = from_bounds(left, bottom, right, top, src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width < 32 or win.height < 32:
            raise InvalidBBox("bbox слишком мал (<32 пикселей по стороне)")

        data = src.read(window=win).astype("float32")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        transform = src.window_transform(win)
        native_bounds = bounds(win, src.transform)
        crs = src.crs
    return data, transform, crs, native_bounds


def _hann_taper(size: int, overlap: int) -> np.ndarray:
    """1D вес для блендинга: плавный подъём/спад на overlap, плато в центре."""
    w = np.ones(size, dtype="float32")
    if overlap > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(1, overlap + 1) / (overlap + 1))))
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return np.clip(w, 1e-3, 1.0)


def predict_sliding(
    features: np.ndarray,
    predict_fn,
    *,
    tile: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    """Прогон по скользящему окну tile×tile с перекрытием overlap и weighted blending.

    ``predict_fn(tile_chw) -> prob_hw`` — функция инференса одного тайла (вход [C,tile,tile]).
    Возвращает карту вероятностей [H, W] исходного размера признаков.
    """
    _c, h, w = features.shape
    stride = tile - overlap

    def grid_len(length: int) -> int:
        if length <= tile:
            return tile
        n = int(np.ceil((length - tile) / stride))
        return tile + n * stride

    hp, wp = grid_len(h), grid_len(w)
    if hp > h or wp > w:
        features = np.pad(features, ((0, 0), (0, hp - h), (0, wp - w)), mode="reflect")

    taper = np.outer(_hann_taper(tile, overlap), _hann_taper(tile, overlap)).astype("float32")
    acc = np.zeros((hp, wp), dtype="float32")
    wsum = np.zeros((hp, wp), dtype="float32")

    for y in range(0, hp - tile + 1, stride):
        for x in range(0, wp - tile + 1, stride):
            chunk = features[:, y : y + tile, x : x + tile]
            prob = predict_fn(chunk).astype("float32")
            acc[y : y + tile, x : x + tile] += prob * taper
            wsum[y : y + tile, x : x + tile] += taper

    out = acc / np.maximum(wsum, 1e-6)
    return out[:h, :w]
