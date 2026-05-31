"""Разметка затопления по Sentinel-1 (FR-3, OQ-2).

Метод ``otsu_change_detection`` (закреплён в configs/data.yaml):
вода → зеркальное отражение → низкое обратное рассеяние. Порог по VV(dB)
подбирается Otsu автоматически; change detection (сухое окно той же орбиты)
отсекает перманентно тёмные не-водные поверхности (асфальт, песок). Затем —
маска уклона, морфочистка и вычитание постоянной воды (см. permanent_water).

Ядро (Otsu, change detection, морфология, IoU) — чистый numpy/scipy,
тестируется на хосте. ``build_flood_mask`` — гео-оркестрация (rasterio).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def amplitude_to_db(amplitude: np.ndarray) -> np.ndarray:
    """Линейная амплитуда GRD (uint16, nodata=0) → dB: 10·log10(amp²).

    Нули/отрицательные (nodata после варпа) → NaN. Калибровка не применяется: для
    Otsu и change detection важна форма распределения и относительный сдвиг, а не
    абсолютный σ0 (см. OQ-2: GRD без террейн-флэттенинга).
    """
    amp = np.asarray(amplitude, dtype="float64")
    out = np.full(amp.shape, np.nan, dtype="float32")
    valid = amp > 0
    out[valid] = (10.0 * np.log10(amp[valid] ** 2)).astype("float32")
    return out


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Порог Otsu по одномерному распределению значений (только конечные)."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("нет конечных значений для Otsu")
    hist, edges = np.histogram(v, bins=nbins)
    total = hist.sum()
    if total == 0:
        raise ValueError("пустая гистограмма")
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = hist.astype("float64") / total
    omega = np.cumsum(p)  # вес класса «ниже порога»
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return float(centers[int(np.argmax(sigma_b))])


def detect_flood(
    flood_db: np.ndarray,
    dry_db: np.ndarray | None = None,
    min_drop_db: float = 2.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Детекция воды по VV(dB): Otsu + (опц.) change detection.

    Вода — пиксели с обратным рассеянием НИЖЕ порога Otsu. Если задан ``dry_db``,
    дополнительно требуем падение сигнала ≥ ``min_drop_db`` относительно сухого
    окна — так перманентно тёмные не-водные поверхности не попадают в разлив.

    Returns:
        (water_mask, threshold_db)
    """
    flood = np.asarray(flood_db, dtype="float32")
    finite = np.isfinite(flood)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)

    thr = otsu_threshold(flood[finite])
    water = (flood < thr) & finite
    if dry_db is not None:
        dry = np.asarray(dry_db, dtype="float32")
        drop = dry - flood
        water &= np.isfinite(drop) & (drop >= min_drop_db)
    return water, thr


def apply_slope_mask(water: np.ndarray, slope_deg: np.ndarray, max_slope_deg: float) -> np.ndarray:
    """Убрать воду на склонах круче порога (разлив там маловероятен)."""
    slope = np.asarray(slope_deg, dtype="float32")
    return np.asarray(water, dtype=bool) & (slope <= max_slope_deg)


def clean_speckle(water: np.ndarray, min_blob_px: int) -> np.ndarray:
    """Удалить связные области воды мельче ``min_blob_px`` (морфочистка)."""
    from scipy import ndimage

    mask = np.asarray(water, dtype=bool)
    if min_blob_px <= 1:
        return mask
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = {i + 1 for i, s in enumerate(sizes) if s >= min_blob_px}
    return np.isin(labels, list(keep)) if keep else np.zeros_like(mask)


def subtract_permanent(flood: np.ndarray, permanent_water: np.ndarray) -> np.ndarray:
    """Оставить только паводок: detected ∧ ¬permanent."""
    return np.asarray(flood, dtype=bool) & ~np.asarray(permanent_water, dtype=bool)


def iou(pred: np.ndarray, reference: np.ndarray) -> float:
    """Intersection-over-Union двух бинарных масок (метрика валидации FR-3)."""
    a = np.asarray(pred, dtype=bool)
    b = np.asarray(reference, dtype=bool)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(inter / union) if union else 1.0


def build_flood_mask(
    flood_db_path: str | Path,
    out_path: str | Path,
    dry_db_path: str | Path | None = None,
    slope_path: str | Path | None = None,
    permanent_water_path: str | Path | None = None,
    *,
    min_drop_db: float = 2.0,
    max_slope_deg: float = 5.0,
    min_blob_px: int = 8,
) -> tuple[Path, float]:
    """Построить полносеточную бинарную маску затопления и записать GeoTIFF.

    Все входы должны быть на одной целевой сетке (см. grid.TargetGrid).
    Возвращает (путь, подобранный порог Otsu в dB).
    """
    import rasterio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(flood_db_path) as src:
        flood = src.read(1).astype("float32")
        profile = src.profile

    def _read(path):
        with rasterio.open(path) as s:
            return s.read(1).astype("float32")

    dry = _read(dry_db_path) if dry_db_path else None
    water, thr = detect_flood(flood, dry, min_drop_db=min_drop_db)

    if slope_path:
        water = apply_slope_mask(water, _read(slope_path), max_slope_deg)
    water = clean_speckle(water, min_blob_px)
    if permanent_water_path:
        water = subtract_permanent(water, _read(permanent_water_path).astype(bool))

    profile.update(driver="GTiff", dtype="uint8", count=1, nodata=255, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(water.astype("uint8"), 1)
    return out_path, thr
