"""Научная валидация на карте: реальная маска затопления Sentinel-1 vs предсказание.

Реальная (ground-truth) бинарная маска разлива собрана на Этапе 1 из Sentinel-1
(см. data/labels_s1) и лежит мозаикой `data/processed/<ds>/labels/<event>/flood_mask.tif`
в нативной проекции датасета (EPSG:32647 для Тулуна). Здесь мы:

  * выравниваем эту маску на грид конкретного предсказания (`runs/<id>/prediction.tif`);
  * считаем согласие предсказания с реальностью на видимой зоне (IoU/F1/precision/recall);
  * рендерим маску как полупрозрачный overlay в EPSG:4326 с теми же bounds, что и
    `prediction.png` (для совмещения в шторке «реальность ↔ предсказание»).

Реальная маска есть ТОЛЬКО для размеченных событий (пилот — Тулун). Для зон вне
покрытия / онлайн-режима пересечения нет → `ground_truth_for_run` возвращает None.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from floodrisk.settings import settings

WGS84 = "EPSG:4326"
# Цвет реальной маски на карте (красный) — контрастен к Viridis-предсказанию.
MASK_RGB = (239, 68, 68)
MASK_ALPHA = 165


# Регионы с валидированными лейблами (эталон/проверка). Канск — демо: признаки корректны,
# но S1-маска не валидирована эталоном (пере-детекция) → помечаем experimental.
VALIDATED_REGIONS = {"v1"}


def label_mask_paths(dataset_version: str | None = None) -> list[Path]:
    """Пути ко ВСЕМ мозаикам реальных масок затопления (мультирегион).

    Сканирует `data/processed/*/labels/*/flood_mask.tif` (Тулун, Канск, …). Маска
    выбирается по пересечению с гридом рана в `ground_truth_for_run`. Патчируется в тестах.
    """
    base = settings.project_root / "data" / "processed"
    if not base.exists():
        return []
    return sorted(base.glob("*/labels/*/flood_mask.tif"))


def _region_of_mask(mask_path: Path) -> str:
    """`processed/<region>/labels/<event>/flood_mask.tif` → <region>."""
    parents = mask_path.parents
    return parents[2].name if len(parents) >= 3 else ""


def compute_agreement(
    pred_prob: np.ndarray,
    truth_bin: np.ndarray,
    valid: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Согласие бинаризованного предсказания (p≥threshold) с реальной маской.

    Считается ТОЛЬКО по пикселям ``valid`` (где реальная маска покрывает зону).
    Деления защищены от нуля → None при пустом знаменателе.
    """
    valid = valid.astype(bool)
    pred_bin = (pred_prob >= threshold) & valid
    truth = truth_bin.astype(bool) & valid

    tp = int(np.count_nonzero(pred_bin & truth))
    fp = int(np.count_nonzero(pred_bin & ~truth))
    fn = int(np.count_nonzero(~pred_bin & truth))

    def _ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den > 0 else None

    n_valid = int(np.count_nonzero(valid))
    n_truth = int(np.count_nonzero(truth))
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_valid_px": n_valid,
        "n_truth_px": n_truth,
        "truth_fraction": round(n_truth / n_valid, 4) if n_valid > 0 else None,
    }


def _reproject_mask_to_grid(mask_path: Path, dst_crs, dst_transform, shape):
    """Репроекция маски на грид предсказания → (mask01[H,W], valid[H,W]).

    ``valid`` помечает пиксели, реально покрытые источником (отличает «нет воды»
    от «вне маски»), чтобы метрики/overlay считались только по покрытой области.
    """
    from rasterio.warp import Resampling, reproject

    h, w = shape
    with rasterio.open(mask_path) as src:
        src_data = (src.read(1) > 0).astype("uint8")
        src_valid = np.ones((src.height, src.width), dtype="uint8")
        src_crs = src.crs
        src_transform = src.transform

    mask01 = np.zeros((h, w), dtype="uint8")
    valid = np.zeros((h, w), dtype="uint8")
    common = dict(
        src_crs=src_crs,
        src_transform=src_transform,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        resampling=Resampling.nearest,
    )
    reproject(source=src_data, destination=mask01, src_nodata=0, dst_nodata=0, **common)
    reproject(source=src_valid, destination=valid, src_nodata=0, dst_nodata=0, **common)
    return mask01, valid


def _write_mask_png_wgs84(path: Path, mask01: np.ndarray, transform, crs) -> list[float]:
    """Бинарная маска → PNG (EPSG:4326), сплошной красный с alpha; вне воды — прозрачно.

    Bounds считаются той же математикой, что ``service._write_png_wgs84`` →
    совпадают с prediction.png (нужно для шторки). Возвращает [S, W, N, E].
    """
    from PIL import Image
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    h, w = mask01.shape
    left, bottom, right, top = array_bounds(h, w, transform)
    dt, dw, dh = calculate_default_transform(crs, WGS84, w, h, left, bottom, right, top)
    dst = np.zeros((dh, dw), dtype="float32")
    reproject(
        source=mask01.astype("float32"),
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dt,
        dst_crs=WGS84,
        resampling=Resampling.nearest,
    )
    flooded = dst > 0.5
    rgba = np.zeros((dh, dw, 4), dtype="uint8")
    rgba[..., 0] = MASK_RGB[0]
    rgba[..., 1] = MASK_RGB[1]
    rgba[..., 2] = MASK_RGB[2]
    rgba[..., 3] = np.where(flooded, MASK_ALPHA, 0).astype("uint8")
    Image.fromarray(rgba, "RGBA").save(path)

    w4, s4, e4, n4 = array_bounds(dh, dw, dt)
    return [float(s4), float(w4), float(n4), float(e4)]


def ground_truth_for_run(run_id: str, dataset_version: str, threshold: float = 0.5) -> dict | None:
    """Реальная маска + метрики согласия для запуска. None, если нет пересечения.

    Открывает `runs/<id>/prediction.tif`, выравнивает все доступные маски события
    на его грид, считает метрики и рисует `runs/<id>/groundtruth.png`.
    Бросает FileNotFoundError, если растра запуска нет.
    """
    run_dir = settings.runs_dir / run_id
    pred_path = run_dir / "prediction.tif"
    if not pred_path.exists():
        raise FileNotFoundError(run_id)

    paths = label_mask_paths(dataset_version)
    if not paths:
        return None

    with rasterio.open(pred_path) as src:
        pred_prob = src.read(1).astype("float32")
        dst_crs = src.crs
        dst_transform = src.transform
        shape = (src.height, src.width)

    mask01 = np.zeros(shape, dtype="uint8")
    valid = np.zeros(shape, dtype="uint8")
    regions: list[str] = []
    for mp in paths:
        m, v = _reproject_mask_to_grid(mp, dst_crs, dst_transform, shape)
        if int(np.count_nonzero(v)) == 0:
            continue  # маска этого региона не пересекает грид рана
        mask01 |= m
        valid |= v
        regions.append(_region_of_mask(mp))

    if int(np.count_nonzero(valid)) == 0:
        return None  # зона вне размеченной области (онлайн/Москва/вне регионов)

    truth_bin = (mask01 > 0) & (valid > 0)
    metrics = compute_agreement(pred_prob, truth_bin, valid, threshold)

    png_path = run_dir / "groundtruth.png"
    bounds_wgs84 = _write_mask_png_wgs84(png_path, mask01 * (valid > 0), dst_transform, dst_crs)

    # experimental, если хоть один пересёкший регион не валидирован эталоном (напр. Канск).
    experimental = any(r not in VALIDATED_REGIONS for r in regions)
    return {
        "png_url": f"/runs/{run_id}/groundtruth.png",
        "bounds_wgs84": bounds_wgs84,
        "metrics": metrics,
        "threshold": metrics["threshold"],
        "regions": regions,
        "experimental": experimental,
    }
