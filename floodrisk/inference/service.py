"""Inference service: bbox+сценарий+модель → растр подверженности и агрегаты.

См. SRS §6.5, §8.1, FR-8/FR-9/FR-12/FR-13. Признаки берём из готового стека
(см. features.py), модель — резидентная (registry.py), сценарий — logit-сдвиг (scenario.py).
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from sqlmodel import Session

from floodrisk.db.models import Run
from floodrisk.db.repositories import (
    get_model_version,
    get_scenario,
    list_model_versions,
    list_scenarios,
)
from floodrisk.inference import features as feat
from floodrisk.inference.registry import load_model
from floodrisk.inference.scenario import apply_scenario
from floodrisk.settings import settings

# Реэкспорт ошибок покрытия/геометрии для маппинга в API.
OutOfCoverage = feat.OutOfCoverage
InvalidBBox = feat.InvalidBBox


class UnknownScenario(Exception):
    def __init__(self, available: list[str]):
        self.available = available
        super().__init__("unknown_scenario")


class UnknownModel(Exception):
    def __init__(self, available: list[str]):
        self.available = available
        super().__init__("unknown_model")


def feature_stack_path(dataset_version: str) -> Path:
    return settings.project_root / "data" / "processed" / dataset_version / "features" / "stack.tif"


def _load_norm_stats(config_path: str) -> tuple[np.ndarray, np.ndarray]:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    stats_path = settings.project_root / cfg["data"]["norm_stats"]
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype="float32").reshape(-1, 1, 1)
    std = np.asarray(stats["std"], dtype="float32").reshape(-1, 1, 1)
    return mean, std


def _resolve_model(session: Session, model_version: str):
    """'unet-v1' → запись ModelVersion. Бросает UnknownModel со списком доступных."""
    available = [f"{m.name}-{m.version}" for m in list_model_versions(session)]
    if "-" not in model_version:
        raise UnknownModel(available)
    name, version = model_version.rsplit("-", 1)
    rec = get_model_version(session, name, version)
    if rec is None:
        raise UnknownModel(available)
    return rec


def _write_geotiff(path: Path, prob: np.ndarray, transform, crs) -> None:
    h, w = prob.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(prob.astype("float32"), 1)


def _write_png_wgs84(path: Path, prob: np.ndarray, transform, crs) -> list[float]:
    """Репроекция вероятностей в EPSG:4326, Viridis+alpha. Возвращает bounds [S,W,N,E]."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm
    from PIL import Image
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    h, w = prob.shape
    left, bottom, right, top = array_bounds(h, w, transform)
    dst_crs = "EPSG:4326"
    dt, dw, dh = calculate_default_transform(crs, dst_crs, w, h, left, bottom, right, top)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(
        source=prob.astype("float32"),
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dt,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=float("nan"),
    )
    valid = ~np.isnan(dst)
    rgba = (cm.viridis(np.clip(np.nan_to_num(dst), 0.0, 1.0)) * 255).astype("uint8")
    rgba[..., 3] = np.where(valid, 180, 0).astype("uint8")
    Image.fromarray(rgba, "RGBA").save(path)

    w4, s4, e4, n4 = array_bounds(dh, dw, dt)
    return [float(s4), float(w4), float(n4), float(e4)]


def _aggregates(prob: np.ndarray, res_m: float) -> dict:
    px_km2 = (res_m * res_m) / 1e6
    return {
        "area_km2": round(float(prob.size) * px_km2, 3),
        "fraction_p_gt_0_5": round(float((prob > 0.5).mean()), 4),
        "fraction_p_gt_0_8": round(float((prob > 0.8).mean()), 4),
        "mean_p": round(float(prob.mean()), 4),
    }


def predict(
    session: Session,
    bbox: list[float],
    scenario_id: str,
    model_version: str,
    *,
    run_dir: Path | None = None,
    run_id: str | None = None,
    persist: bool = True,
) -> dict:
    """Полный инференс. Возвращает dict для API/CLI. Бросает Unknown*/OutOfCoverage/InvalidBBox."""
    t0 = time.perf_counter()

    if len(bbox) != 4 or not (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
        raise InvalidBBox(f"ожидается [W,S,E,N] с W<E и S<N, получено {bbox}")

    scenario = get_scenario(session, scenario_id)
    if scenario is None:
        raise UnknownScenario([s.scenario_id for s in list_scenarios(session)])
    rec = _resolve_model(session, model_version)

    stack = feature_stack_path(rec.dataset_version)
    fdata, transform, crs, _native_bounds = feat.read_feature_window(bbox, stack)

    mean, std = _load_norm_stats(rec.config_path)
    fnorm = ((fdata - mean) / std).astype("float32")

    model = load_model(rec.name, rec.version, rec.checkpoint_path)
    base_prob = feat.predict_sliding(fnorm, model.predict_tile)

    params = json.loads(scenario.params_json)
    prob = apply_scenario(base_prob, params)

    run_id = run_id or uuid.uuid4().hex
    run_dir = run_dir or (settings.runs_dir / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    tif_path = run_dir / "prediction.tif"
    png_path = run_dir / "prediction.png"

    _write_geotiff(tif_path, prob, transform, crs)
    bounds_wgs84 = _write_png_wgs84(png_path, prob, transform, crs)
    res_m = abs(transform.a)
    aggregates = _aggregates(prob, res_m)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    metadata = {
        "model_version": model_version,
        "dataset_version": rec.dataset_version,
        "scenario_id": scenario_id,
        "bbox": bbox,
        "run_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "latency_ms": latency_ms,
    }
    (run_dir / "aggregates.json").write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    png_url = f"/runs/{run_id}/prediction.png"
    tif_url = f"/runs/{run_id}/prediction.tif"

    if persist:
        session.add(
            Run(
                id=run_id,
                bbox_w=bbox[0],
                bbox_s=bbox[1],
                bbox_e=bbox[2],
                bbox_n=bbox[3],
                scenario_id=scenario_id,
                model_version_id=rec.id,
                prediction_tif_path=tif_url,
                prediction_png_path=png_url,
                aggregates_json=json.dumps(aggregates, ensure_ascii=False),
                latency_ms=latency_ms,
                status="ok",
            )
        )
        session.commit()

    return {
        "run_id": run_id,
        "prediction_png_url": png_url,
        "prediction_tif_url": tif_url,
        "bounds_wgs84": bounds_wgs84,
        "aggregates": aggregates,
        "metadata": metadata,
    }
