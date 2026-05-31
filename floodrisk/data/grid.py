"""Целевая растровая сетка датасета (SRS §6.4).

Все растры пайплайна (признаки, тайлы, лейблы, маска воды) приводятся к одной
сетке: целевой CRS, разрешение 30 м, общий affine-transform. ``TargetGrid``
вычисляется из bbox пилота и параметров ``configs/data.yaml``.

Зависит только от pyproj/affine (идут с rasterio) — тестируется на хосте.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from affine import Affine
from pyproj import Transformer

from floodrisk.data.config import DataConfig


@dataclass(frozen=True)
class TargetGrid:
    crs: str
    transform: Affine
    width: int
    height: int
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols) — как у numpy/rasterio."""
        return (self.height, self.width)

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) в целевом CRS."""
        xmin = self.transform.c
        ymax = self.transform.f
        xmax = xmin + self.width * self.resolution_m
        ymin = ymax - self.height * self.resolution_m
        return (xmin, ymin, xmax, ymax)

    @classmethod
    def from_config(cls, cfg: DataConfig) -> TargetGrid:
        w, s, e, n = cfg.pilot_bbox_wgs84
        res = cfg.tile_resolution_m
        transformer = Transformer.from_crs("EPSG:4326", cfg.target_crs, always_xy=True)

        # проецируем все 4 угла bbox и берём охватывающий прямоугольник
        corners = [(w, s), (w, n), (e, s), (e, n)]
        xs, ys = zip(*(transformer.transform(lon, lat) for lon, lat in corners), strict=True)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # привязываем к кратности разрешения, расширяя наружу
        xmin = math.floor(xmin / res) * res
        ymin = math.floor(ymin / res) * res
        xmax = math.ceil(xmax / res) * res
        ymax = math.ceil(ymax / res) * res

        width = round((xmax - xmin) / res)
        height = round((ymax - ymin) / res)
        # north-up affine: origin в верхнем-левом углу, y убывает вниз
        transform = Affine(res, 0.0, xmin, 0.0, -res, ymax)
        return cls(
            crs=cfg.target_crs,
            transform=transform,
            width=width,
            height=height,
            resolution_m=res,
        )
