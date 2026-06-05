"""Горячие точки риска: связные компоненты по растру предсказания + эндпоинт.

Без сети и без extras [ml]: чистый file-IO на синтетических растрах (реальные
runs/<id>/prediction.tif в CI отсутствуют — gitignore).
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from floodrisk.inference import hotspots as hs
from floodrisk.settings import settings

CRS = "EPSG:32647"
RES = 30.0
ORIGIN = (500000.0, 6000000.0)


def _write_prediction(path, arr):
    transform = from_origin(ORIGIN[0], ORIGIN[1], RES, RES)
    h, w = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def _make_run(tmp_path, monkeypatch):
    """Растр 64×64: большой кластер A (20×20, p0.9), меньший B (8×8, p0.7),
    спекл из 2 px (p0.9, должен отсеяться min_area_px=4). Фон 0.1."""
    monkeypatch.setattr(settings, "project_root", tmp_path)
    run_dir = settings.runs_dir / "run123"
    run_dir.mkdir(parents=True)

    pred = np.full((64, 64), 0.1, dtype="float32")
    pred[0:20, 0:20] = 0.9  # A: 400 px
    pred[40:48, 40:48] = 0.7  # B: 64 px
    pred[62, 0:2] = 0.9  # спекл: 2 px < min_area_px
    _write_prediction(run_dir / "prediction.tif", pred)
    return "run123"


def test_find_hotspots_ranking_and_filter(tmp_path, monkeypatch):
    run_id = _make_run(tmp_path, monkeypatch)
    res = hs.find_hotspots(run_id, threshold=0.5, top_n=5, min_area_px=4)

    # спекл (2 px) отфильтрован → ровно 2 кластера, A и B
    assert res["count"] == 2
    assert len(res["hotspots"]) == 2
    a, b = res["hotspots"]
    # A крупнее и плотнее → rank 1; ранги проставлены
    assert a["rank"] == 1 and b["rank"] == 2
    assert a["area_km2"] > b["area_km2"]
    # A: 400 px * (30*30/1e6) = 0.36 км²
    assert a["area_km2"] == pytest.approx(0.36, abs=1e-6)
    assert a["mean_p"] == pytest.approx(0.9, abs=1e-4)
    assert a["max_p"] == pytest.approx(0.9, abs=1e-4)
    # bounds [S,W,N,E] корректной ориентации
    s, w, n, e = a["bounds_wgs84"]
    assert n > s and e > w
    # центроид A — на северо-западе зоны (большие lat, меньшие lon относительно B)
    assert a["lat"] > b["lat"] and a["lon"] < b["lon"]


def test_find_hotspots_top_n_limits_list_not_count(tmp_path, monkeypatch):
    run_id = _make_run(tmp_path, monkeypatch)
    res = hs.find_hotspots(run_id, top_n=1)
    assert res["count"] == 2  # всего кластеров
    assert len(res["hotspots"]) == 1  # но в списке — только топ-1
    assert res["hotspots"][0]["rank"] == 1


def test_find_hotspots_high_threshold_drops_weaker(tmp_path, monkeypatch):
    run_id = _make_run(tmp_path, monkeypatch)
    # порог 0.8 отсекает кластер B (p0.7), остаётся только A
    res = hs.find_hotspots(run_id, threshold=0.8, min_area_px=4)
    assert res["count"] == 1
    assert res["hotspots"][0]["mean_p"] == pytest.approx(0.9, abs=1e-4)


def test_find_hotspots_missing_run_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", tmp_path)
    with pytest.raises(FileNotFoundError):
        hs.find_hotspots("nope")


# ── эндпоинт ──────────────────────────────────────────────────────────────────
def test_endpoint_unknown_run_404(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", tmp_path)
    r = client.get("/api/runs/nope/hotspots")
    assert r.status_code == 404


def test_endpoint_returns_hotspots(client: TestClient, tmp_path, monkeypatch):
    run_id = _make_run(tmp_path, monkeypatch)
    r = client.get(f"/api/runs/{run_id}/hotspots")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["count"] == 2
    assert body["hotspots"][0]["rank"] == 1
