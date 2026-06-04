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
    def from_bbox(cls, bbox_wgs84: list[float], res_m: float, crs: str) -> TargetGrid:
        """Сетка для произвольного bbox (WGS84 [W,S,E,N]) в заданном CRS.

        Проецирует углы bbox в целевой CRS, берёт охватывающий прямоугольник,
        привязывает к кратности разрешения. Основа и для пилота (from_config),
        и для онлайн-инференса по произвольной зоне.
        """
        w, s, e, n = bbox_wgs84
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        corners = [(w, s), (w, n), (e, s), (e, n)]
        xs, ys = zip(*(transformer.transform(lon, lat) for lon, lat in corners), strict=True)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # привязываем к кратности разрешения, расширяя наружу
        xmin = math.floor(xmin / res_m) * res_m
        ymin = math.floor(ymin / res_m) * res_m
        xmax = math.ceil(xmax / res_m) * res_m
        ymax = math.ceil(ymax / res_m) * res_m

        width = round((xmax - xmin) / res_m)
        height = round((ymax - ymin) / res_m)
        # north-up affine: origin в верхнем-левом углу, y убывает вниз
        transform = Affine(res_m, 0.0, xmin, 0.0, -res_m, ymax)
        return cls(crs=crs, transform=transform, width=width, height=height, resolution_m=res_m)

    @classmethod
    def from_config(cls, cfg: DataConfig) -> TargetGrid:
        return cls.from_bbox(cfg.pilot_bbox_wgs84, cfg.tile_resolution_m, cfg.target_crs)


def utm_epsg_for_bbox(bbox_wgs84: list[float]) -> str:
    """EPSG UTM-зоны по центроиду bbox. Север → 326xx, юг → 327xx.

    Чистая функция (без I/O). Зона = floor((lon+180)/6)+1, клип к 1..60.
    Для онлайн-инференса по произвольной зоне (целевая проекция метрическая).
    """
    w, s, e, n = bbox_wgs84
    lon_c = (w + e) / 2.0
    lat_c = (s + n) / 2.0
    zone = math.floor((lon_c + 180.0) / 6.0) + 1
    zone = min(60, max(1, zone))
    hemisphere = 600 if lat_c >= 0 else 700  # 326xx (N) / 327xx (S)
    return f"EPSG:32{hemisphere + zone}"
