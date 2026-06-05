"""Объяснимость: Integrated Gradients (U-Net) / feature_importances_ (baseline). См. SRS §7.7, FR-10.

Окно 256×256 вокруг точки клика → атрибуции по 18 каналам входа → агрегация обратно в 7
семантических признаков (feature_transform.feature_groups). Только core-зависимости (captum, torch).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from sqlmodel import Session

from floodrisk import feature_transform as ft
from floodrisk.db.models import Explanation, ModelVersion
from floodrisk.db.repositories import get_run
from floodrisk.inference import online_features as online
from floodrisk.inference.service import _load_norm_stats
from floodrisk.settings import settings

WGS84 = "EPSG:4326"
TILE = 256


class RunNotFound(Exception):
    pass


class PointOutOfCoverage(Exception):
    pass


def _window_around(lat: float, lon: float, stack_path: Path):
    """Окно TILE×TILE сырых признаков вокруг точки. Бросает PointOutOfCoverage."""
    with rasterio.open(stack_path) as src:
        try:
            xs, ys = warp_transform(WGS84, src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
        except Exception as exc:  # точка вне домена проекции
            raise PointOutOfCoverage("точка вне зоны покрытия") from exc
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise PointOutOfCoverage("точка вне зоны покрытия")
        c0 = int(np.clip(col - TILE // 2, 0, max(0, src.width - TILE)))
        r0 = int(np.clip(row - TILE // 2, 0, max(0, src.height - TILE)))
        win = Window(c0, r0, min(TILE, src.width), min(TILE, src.height))
        raw = np.nan_to_num(src.read(window=win).astype("float32"))
        transform = src.window_transform(win)
        crs = src.crs
    return raw, transform, crs


def _crop_to_multiple_of_32(raw: np.ndarray, transform):
    """Центральная обрезка окна до кратности 32 по сторонам (вход энкодера U-Net resnet34)."""
    _c, h, w = raw.shape
    nh = max(32, (h // 32) * 32)
    nw = max(32, (w // 32) * 32)
    r0, c0 = (h - nh) // 2, (w - nw) // 2
    win = Window(c0, r0, nw, nh)
    cropped = np.ascontiguousarray(raw[:, r0 : r0 + nh, c0 : c0 + nw])
    return cropped, window_transform(win, transform)


def _window_for_explain(lat: float, lon: float, dataset_version: str):
    """Окно признаков вокруг точки: мозаика в покрытии, иначе онлайн-сборка (как в predict).

    Возвращает (raw[7,H,W], transform, crs). Размер кратен 32 (требование U-Net).
    """
    from floodrisk.inference.service import _mosaic_stacks

    # Мультирегион: пробуем все региональные мозаики (Тулун, Канск); первая, что покрывает точку.
    for stack in _mosaic_stacks(dataset_version):
        try:
            return _window_around(lat, lon, stack)
        except PointOutOfCoverage:
            continue
    # вне всех мозаик → собрать окно ~TILE*res вокруг точки онлайн (experimental)
    half_m = (TILE * online.RES_M) / 2.0
    dlat = half_m / 111320.0
    dlon = half_m / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    bbox = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]
    raw, transform, crs = online.build_feature_window(bbox, cache_dir=settings.online_cache_dir)
    cropped, t2 = _crop_to_multiple_of_32(raw, transform)
    return cropped, t2, crs


def _unet_attributions(checkpoint_path: str, fnorm: np.ndarray, n_steps: int = 16) -> np.ndarray:
    """Integrated Gradients по TorchScript-модели. Возвращает атрибуции [C,H,W]."""
    import torch
    from captum.attr import IntegratedGradients

    module = torch.jit.load(checkpoint_path, map_location="cpu")
    module.eval()
    inp = torch.from_numpy(np.ascontiguousarray(fnorm[None], dtype="float32")).requires_grad_(True)

    def forward(x):
        # Скаляр на батч = суммарная «улика затопления» по окну (логиты).
        return module(x).flatten(1).sum(1)

    ig = IntegratedGradients(forward)
    attr = ig.attribute(inp, n_steps=n_steps)
    return attr.detach()[0].numpy()


def _to_wgs84_png(arr: np.ndarray, transform, crs, path: Path, cmap: str) -> list[float]:
    """Знаковую карту атрибуции → PNG в EPSG:4326 (нормировка по max|·|). Возвращает [S,W,N,E]."""
    import matplotlib

    matplotlib.use("Agg")
    from PIL import Image
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    h, w = arr.shape
    left, bottom, right, top = array_bounds(h, w, transform)
    dt, dw, dh = calculate_default_transform(crs, WGS84, w, h, left, bottom, right, top)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(
        source=arr.astype("float32"),
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dt,
        dst_crs=WGS84,
        resampling=Resampling.bilinear,
        dst_nodata=float("nan"),
    )
    valid = ~np.isnan(dst)
    m = float(np.nanmax(np.abs(dst))) or 1.0
    norm = np.clip(0.5 + 0.5 * np.nan_to_num(dst) / m, 0.0, 1.0)  # знак → [0,1]
    rgba = (matplotlib.colormaps[cmap](norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(valid, 200, 0).astype("uint8")
    Image.fromarray(rgba, "RGBA").save(path)
    w4, s4, e4, n4 = array_bounds(dh, dw, dt)
    return [float(s4), float(w4), float(n4), float(e4)]


def explain(session: Session, run_id: str, lat: float, lon: float, *, top_n: int = 5) -> dict:
    """Объяснение для запуска в точке (lat,lon). См. SRS §8.1. Бросает RunNotFound/PointOutOfCoverage."""
    run = get_run(session, run_id)
    if run is None:
        raise RunNotFound(run_id)
    rec = session.get(ModelVersion, run.model_version_id)

    raw, transform, crs = _window_for_explain(lat, lon, rec.dataset_version)
    mean, std = _load_norm_stats(rec.config_path)
    fnorm = ft.transform(raw, mean, std)
    groups = ft.feature_groups()

    per_feature_map: dict[str, np.ndarray] = {}
    if Path(rec.checkpoint_path).suffix == ".pt":
        attr = _unet_attributions(rec.checkpoint_path, fnorm)  # [C,H,W]
        importance = {name: float(np.abs(attr[idxs]).sum()) for name, idxs in groups.items()}
        per_feature_map = {name: attr[idxs].sum(axis=0) for name, idxs in groups.items()}
    else:  # baseline RF: глобальные feature_importances_
        from floodrisk.ml.baseline import load_baseline

        fi = load_baseline(rec.checkpoint_path).feature_importances_
        importance = {name: float(fi[idxs].sum()) for name, idxs in groups.items()}

    total = sum(importance.values()) or 1.0
    ranking = sorted(
        (
            {"feature": n, "label": ft.FEATURE_LABELS[n], "importance": round(v / total, 4)}
            for n, v in importance.items()
        ),
        key=lambda r: r["importance"],
        reverse=True,
    )[:top_n]

    attr_dir = settings.runs_dir / run_id / "attribution"
    attribution_layers = []
    attr_paths = []
    if per_feature_map:
        attr_dir.mkdir(parents=True, exist_ok=True)
        for r in ranking:
            name = r["feature"]
            png = attr_dir / f"{name}.png"
            bounds = _to_wgs84_png(per_feature_map[name], transform, crs, png, "magma")
            url = f"/runs/{run_id}/attribution/{name}.png"
            attr_paths.append(url)
            attribution_layers.append(
                {"feature": name, "label": r["label"], "png_url": url, "bounds_wgs84": bounds}
            )

    exp = Explanation(
        run_id=run_id,
        lat=lat,
        lon=lon,
        importance_json=json.dumps(ranking, ensure_ascii=False),
        attribution_tif_paths_json=json.dumps(attr_paths),
    )
    session.add(exp)
    session.commit()
    session.refresh(exp)

    return {"explanation_id": exp.id, "ranking": ranking, "attribution_layers": attribution_layers}
