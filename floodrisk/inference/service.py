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

from floodrisk import feature_transform as ft
from floodrisk.db.models import Run
from floodrisk.db.repositories import (
    get_model_version,
    get_scenario,
    list_model_versions,
    list_scenarios,
)
from floodrisk.inference import features as feat
from floodrisk.inference import online_features as online
from floodrisk.inference.registry import load_model
from floodrisk.inference.scenario import apply_scenario
from floodrisk.settings import settings

# Реэкспорт ошибок покрытия/геометрии для маппинга в API.
OutOfCoverage = feat.OutOfCoverage
InvalidBBox = feat.InvalidBBox
BBoxTooLarge = online.BBoxTooLarge
OnlineFetchError = online.OnlineFetchError


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


# Регионы с офлайн-мозаикой признаков (валидированные/быстрые). Человекочитаемые подписи и
# отметка «валидирован» (для UI/валидации): Тулун обучен и проверен; Канск — демо (признаки
# корректны, но S1-лейблы не валидированы эталоном).
REGION_LABELS = {"v1": "Тулун", "kansk": "Канск"}
VALIDATED_REGIONS = {"v1"}


def _region_stacks() -> list[Path]:
    """Все региональные мозаики признаков `data/processed/<region>/features/stack.tif`.

    Исключает объединённые датасеты (v2) — у них нет своего stack.tif. Мультирегион:
    инференс перебирает эти стеки, выбирая покрывающий bbox (Тулун, Канск, …).
    """
    base = settings.project_root / "data" / "processed"
    if not base.exists():
        return []
    return sorted(p for p in base.glob("*/features/stack.tif"))


def _mosaic_stacks(dataset_version: str) -> list[Path]:
    """Список стеков-мозаик для перебора: основной (по dataset_version) + прочие регионы."""
    primary = feature_stack_path(dataset_version)
    out: list[Path] = []
    seen: set = set()
    for sp in [primary, *_region_stacks()]:
        try:
            key = sp.resolve()
        except OSError:
            key = sp
        if key not in seen and sp.exists():
            seen.add(key)
            out.append(sp)
    return out


def mosaic_coverages() -> list[dict]:
    """Покрытия всех региональных мозаик для UI: [{region, label, bbox_wgs84}].

    Порядок: валидированные регионы (Тулун) первыми — первый становится дефолтной зоной в UI.
    """
    out: list[dict] = []
    stacks = sorted(
        _region_stacks(),
        key=lambda p: (p.parent.parent.name not in VALIDATED_REGIONS, p.parent.parent.name),
    )
    for sp in stacks:
        region = sp.parent.parent.name
        try:
            out.append(
                {
                    "region": region,
                    "label": REGION_LABELS.get(region, region),
                    "bbox": feat.coverage_wgs84(sp),
                }
            )
        except Exception:
            continue
    return out


def _load_norm_stats(config_path: str) -> tuple[np.ndarray, np.ndarray]:
    import yaml

    # config_path в БД хранится относительным (например 'configs/unet_v1.yaml') →
    # резолвим от корня проекта, чтобы работать независимо от cwd процесса (Docker WORKDIR,
    # запуск не из корня и т.п.), а не только когда cwd == корень.
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = settings.project_root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return ft.load_stats(settings.project_root / cfg["data"]["norm_stats"])


def _resolve_features(bbox: list[float], dataset_version: str, source: str):
    """Источник признаков → (raw[7,H,W], transform, crs, used_source).

    source: 'mosaic' (готовая мозаика, валидированный регион), 'online' (сборка на
    лету для произвольного bbox, экспериментально) или 'auto' (мозаика в покрытии,
    иначе онлайн). См. SRS-расширение «глобальный режим».
    """
    stacks = _mosaic_stacks(dataset_version)
    if source == "online":
        fdata, transform, crs = online.build_feature_window(
            bbox, cache_dir=settings.online_cache_dir
        )
        return fdata, transform, crs, "online"

    # mosaic/auto: перебираем региональные мозаики (Тулун, Канск, …); первая покрывающая bbox.
    last_coverage: list[float] | None = None
    for stack in stacks:
        try:
            fdata, transform, crs, _ = feat.read_feature_window(bbox, stack)
            return fdata, transform, crs, "mosaic"
        except feat.OutOfCoverage as exc:
            last_coverage = exc.coverage_wgs84
            continue
    # ни одна мозаика не покрыла bbox
    if source == "mosaic":
        raise feat.OutOfCoverage(last_coverage or [])
    # auto → онлайн-сборка
    fdata, transform, crs = online.build_feature_window(bbox, cache_dir=settings.online_cache_dir)
    return fdata, transform, crs, "online"


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


def sample_point(run_id: str, lat: float, lon: float) -> float | None:
    """Вероятность затопления в точке (lat, lon, EPSG:4326) из растра запуска.

    Открывает ``runs/<run_id>/prediction.tif``, репроецирует точку в CRS растра и
    сэмплирует значение. Возвращает float ∈[0,1] или None (точка вне границ растра
    либо nodata/NaN). Бросает FileNotFoundError, если растра запуска нет.
    """
    from rasterio.warp import transform as warp_transform

    tif_path = settings.runs_dir / run_id / "prediction.tif"
    if not tif_path.exists():
        raise FileNotFoundError(run_id)

    with rasterio.open(tif_path) as src:
        try:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        except Exception:
            # Точка вне домена проекции растра (напр. далеко от UTM-зоны) → вне зоны.
            return None
        x, y = xs[0], ys[0]
        left, bottom, right, top = src.bounds
        if not (left <= x <= right and bottom <= y <= top):
            return None
        value = next(src.sample([(x, y)]))[0]

    if value is None or np.isnan(value):
        return None
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def predict(
    session: Session,
    bbox: list[float],
    scenario_id: str,
    model_version: str,
    *,
    source: str = "auto",
    run_dir: Path | None = None,
    run_id: str | None = None,
    persist: bool = True,
) -> dict:
    """Полный инференс. Возвращает dict для API/CLI.

    source: 'auto' (мозаика в покрытии, иначе онлайн), 'mosaic', 'online'.
    Бросает Unknown*/OutOfCoverage/InvalidBBox/BBoxTooLarge/OnlineFetchError.
    """
    t0 = time.perf_counter()

    if len(bbox) != 4 or not (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
        raise InvalidBBox(f"ожидается [W,S,E,N] с W<E и S<N, получено {bbox}")

    scenario = get_scenario(session, scenario_id)
    if scenario is None:
        raise UnknownScenario([s.scenario_id for s in list_scenarios(session)])
    rec = _resolve_model(session, model_version)

    fdata, transform, crs, used_source = _resolve_features(bbox, rec.dataset_version, source)

    mean, std = _load_norm_stats(rec.config_path)
    fnorm = ft.transform(fdata, mean, std)

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
        "source": used_source,
        "experimental": used_source == "online",
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
