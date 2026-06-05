"""Научная валидация: метрики согласия + реальная маска S1 vs предсказание.

Без сети и без extras [ml]: чистая функция метрик + file-IO на синтетических растрах
(реальный data/processed/.../flood_mask.tif в CI отсутствует — он gitignore).
"""

from __future__ import annotations

import json

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from floodrisk.inference import service as svc
from floodrisk.inference import validation
from floodrisk.settings import settings

CRS = "EPSG:32647"
RES = 30.0
ORIGIN = (500000.0, 6000000.0)


# ── compute_agreement (чистая функция) ────────────────────────────────────────
def test_agreement_perfect_match():
    pred = np.array([[0.9, 0.9], [0.1, 0.1]], dtype="float32")
    truth = np.array([[1, 1], [0, 0]], dtype="uint8")
    valid = np.ones((2, 2), dtype="uint8")
    m = validation.compute_agreement(pred, truth, valid, threshold=0.5)
    assert m["iou"] == 1.0
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_agreement_partial_known_counts():
    pred = np.array([[0.9, 0.9], [0.1, 0.1]], dtype="float32")
    truth = np.array([[1, 0], [1, 0]], dtype="uint8")
    valid = np.ones((2, 2), dtype="uint8")
    m = validation.compute_agreement(pred, truth, valid)
    # TP=1, FP=1, FN=1 → IoU=1/3, F1=0.5, precision=recall=0.5
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert m["iou"] == 0.3333
    assert m["f1"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5


def test_agreement_disjoint_is_zero():
    pred = np.array([[1.0, 0.0], [0.0, 0.0]], dtype="float32")
    truth = np.array([[0, 0], [0, 1]], dtype="uint8")
    valid = np.ones((2, 2), dtype="uint8")
    m = validation.compute_agreement(pred, truth, valid)
    assert m["iou"] == 0.0
    assert m["f1"] == 0.0


def test_agreement_empty_denominator_is_none():
    pred = np.zeros((2, 2), dtype="float32")
    truth = np.zeros((2, 2), dtype="uint8")
    valid = np.ones((2, 2), dtype="uint8")
    m = validation.compute_agreement(pred, truth, valid)
    assert m["iou"] is None and m["f1"] is None
    assert m["precision"] is None and m["recall"] is None


def test_agreement_respects_valid_mask():
    # Несовпадение лежит вне valid → не учитывается.
    pred = np.array([[0.9, 0.9], [0.9, 0.1]], dtype="float32")
    truth = np.array([[1, 1], [0, 0]], dtype="uint8")
    valid = np.array([[1, 1], [0, 0]], dtype="uint8")  # нижняя строка вне покрытия
    m = validation.compute_agreement(pred, truth, valid)
    assert m["n_valid_px"] == 2
    assert m["iou"] == 1.0


# ── bounds: реальная маска совпадает с prediction.png ─────────────────────────
def test_mask_png_bounds_match_prediction(tmp_path):
    transform = from_origin(ORIGIN[0], ORIGIN[1], RES, RES)
    prob = np.random.default_rng(0).random((48, 48)).astype("float32")
    mask = (prob > 0.5).astype("uint8")
    pred_bounds = svc._write_png_wgs84(tmp_path / "p.png", prob, transform, CRS)
    mask_bounds = validation._write_mask_png_wgs84(tmp_path / "m.png", mask, transform, CRS)
    assert np.allclose(pred_bounds, mask_bounds)
    assert (tmp_path / "m.png").exists()


# ── ground_truth_for_run (file-IO, без модели/БД) ─────────────────────────────
def _write_raster(path, arr, count=1, dtype="float32"):
    transform = from_origin(ORIGIN[0], ORIGIN[1], RES, RES)
    h, w = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=dtype,
        crs=CRS,
        transform=transform,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def _make_run(tmp_path, monkeypatch, *, source="mosaic", with_mask=True):
    # runs_dir = project_root/"runs" (property) → патчим базовый project_root.
    monkeypatch.setattr(settings, "project_root", tmp_path)
    run_dir = settings.runs_dir / "run123"
    run_dir.mkdir(parents=True)

    pred = np.full((64, 64), 0.1, dtype="float32")
    pred[:32, :32] = 0.9  # «затопленный» квадрант
    _write_raster(run_dir / "prediction.tif", pred)
    (run_dir / "metadata.json").write_text(
        json.dumps({"dataset_version": "v1", "source": source}), encoding="utf-8"
    )

    if with_mask:
        mask = np.zeros((64, 64), dtype="uint8")
        mask[:32, :32] = 1  # реальная маска совпадает с предсказанным квадрантом
        mask_path = tmp_path / "flood_mask.tif"
        _write_raster(mask_path, mask, dtype="uint8")
        monkeypatch.setattr(validation, "label_mask_paths", lambda _ds: [mask_path])
    else:
        monkeypatch.setattr(validation, "label_mask_paths", lambda _ds: [])
    return "run123"


def test_ground_truth_for_run_perfect(tmp_path, monkeypatch):
    run_id = _make_run(tmp_path, monkeypatch)
    res = validation.ground_truth_for_run(run_id, "v1", threshold=0.5)
    assert res is not None
    assert res["metrics"]["iou"] == 1.0
    assert res["metrics"]["f1"] == 1.0
    assert (settings.runs_dir / run_id / "groundtruth.png").exists()
    s, w, n, e = res["bounds_wgs84"]
    assert n > s and e > w


def test_ground_truth_none_without_mask(tmp_path, monkeypatch):
    run_id = _make_run(tmp_path, monkeypatch, with_mask=False)
    assert validation.ground_truth_for_run(run_id, "v1") is None


# ── эндпоинт ──────────────────────────────────────────────────────────────────
def test_endpoint_unknown_run_404(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", tmp_path)
    r = client.get("/api/runs/nope/groundtruth")
    assert r.status_code == 404


def test_endpoint_available_with_metrics(client: TestClient, tmp_path, monkeypatch):
    run_id = _make_run(tmp_path, monkeypatch)
    r = client.get(f"/api/runs/{run_id}/groundtruth")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["metrics"]["iou"] == 1.0
    assert body["png_url"].endswith("groundtruth.png")


def test_endpoint_online_source_unavailable(client: TestClient, tmp_path, monkeypatch):
    run_id = _make_run(tmp_path, monkeypatch, source="online")
    r = client.get(f"/api/runs/{run_id}/groundtruth")
    assert r.status_code == 200
    assert r.json()["available"] is False
