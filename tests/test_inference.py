"""Smoke + ошибки инференса (FR-9, FR-12). Фикстурный стек + крошечная TorchScript-модель.

Не требует extras [ml] и реальных чекпойнтов → гоняется в core-CI.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from sqlmodel import Session, delete

from floodrisk.db.models import Explanation, Export, ModelVersion, Run, Scenario
from floodrisk.db.session import create_db_and_tables, engine
from floodrisk.inference import registry
from floodrisk.inference import service as svc
from floodrisk.inference.features import coverage_wgs84


def _clear_db(session: Session) -> None:
    # in-memory SQLite из conftest общий на сессию (StaticPool) → чистим за собой.
    # Порядок учитывает FK: сначала зависимые (Export/Explanation → Run), потом Run.
    for model in (Export, Explanation, Run, ModelVersion, Scenario):
        session.exec(delete(model))
    session.commit()


@pytest.fixture
def inference_env(tmp_path, monkeypatch):
    import torch

    # Крошечный признаковый стек: 7 каналов, EPSG:32647, 384×384, 30 м.
    crs = "EPSG:32647"
    res = 30.0
    w_px = h_px = 384
    transform = from_origin(500000.0, 6000000.0, res, res)
    rng = np.random.default_rng(0)
    data = rng.random((7, h_px, w_px)).astype("float32")
    stack = tmp_path / "stack.tif"
    with rasterio.open(
        stack,
        "w",
        driver="GTiff",
        height=h_px,
        width=w_px,
        count=7,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    from floodrisk.feature_transform import CONTINUOUS, out_channels

    n_cont = len(CONTINUOUS)
    norm = tmp_path / "norm_stats.json"
    norm.write_text(json.dumps({"mean": [0.0] * n_cont, "std": [1.0] * n_cont}))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"data:\n  norm_stats: {norm.as_posix()}\n", encoding="utf-8")

    # Крошечная TorchScript-модель C→1 (вход = feature_transform.out_channels()), чистый torch.
    c_in = out_channels()
    net = torch.nn.Conv2d(c_in, 1, kernel_size=1)
    traced = torch.jit.trace(net, torch.zeros(1, c_in, 256, 256))
    model_path = tmp_path / "model.ts.pt"
    traced.save(str(model_path))

    monkeypatch.setattr(svc, "feature_stack_path", lambda _ds: stack)
    registry.clear_cache()

    create_db_and_tables()
    with Session(engine) as session:
        _clear_db(session)
        session.add(
            Scenario(scenario_id="p100y", name="100y", params_json='{"return_period_years": 100}')
        )
        session.add(
            ModelVersion(
                name="unet",
                version="test",
                checkpoint_path=str(model_path),
                config_path=str(cfg),
                dataset_version="testds",
            )
        )
        session.commit()

    cov = coverage_wgs84(stack)  # [W,S,E,N]
    yield {"coverage": cov}

    with Session(engine) as session:
        _clear_db(session)
    registry.clear_cache()


def _inner_bbox(cov: list[float]) -> list[float]:
    w, s, e, n = cov
    return [w + (e - w) * 0.3, s + (n - s) * 0.3, w + (e - w) * 0.7, s + (n - s) * 0.7]


def test_predict_ok_returns_valid_geotiff(client, inference_env):
    bbox = _inner_bbox(inference_env["coverage"])
    r = client.post(
        "/api/predict", json={"bbox": bbox, "scenario_id": "p100y", "model_version": "unet-test"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"]
    assert body["prediction_tif_url"].endswith("prediction.tif")
    assert len(body["bounds_wgs84"]) == 4
    assert set(body["aggregates"]) == {
        "area_km2",
        "fraction_p_gt_0_5",
        "fraction_p_gt_0_8",
        "mean_p",
    }
    # Открыть сохранённый GeoTIFF и проверить, что он валиден.
    from floodrisk.settings import settings

    tif = settings.runs_dir / body["run_id"] / "prediction.tif"
    with rasterio.open(tif) as src:
        assert src.count == 1
        assert src.crs.to_epsg() == 32647


def test_predict_out_of_coverage_422(client, inference_env):
    w, s, _e, _n = inference_env["coverage"]
    bbox = [w + 20, s + 5, w + 21, s + 6]  # заведомо вне покрытия
    r = client.post(
        "/api/predict", json={"bbox": bbox, "scenario_id": "p100y", "model_version": "unet-test"}
    )
    assert r.status_code == 422
    assert r.json()["error"] == "bbox_out_of_coverage"
    assert len(r.json()["coverage_wgs84"]) == 4


def test_predict_unknown_scenario_404(client, inference_env):
    bbox = _inner_bbox(inference_env["coverage"])
    r = client.post(
        "/api/predict", json={"bbox": bbox, "scenario_id": "nope", "model_version": "unet-test"}
    )
    assert r.status_code == 404
    assert r.json()["error"] == "unknown_scenario"


def test_predict_unknown_model_404(client, inference_env):
    bbox = _inner_bbox(inference_env["coverage"])
    r = client.post(
        "/api/predict", json={"bbox": bbox, "scenario_id": "p100y", "model_version": "ghost-v9"}
    )
    assert r.status_code == 404
    assert r.json()["error"] == "unknown_model"


def test_scenario_modulation_monotonic():
    from floodrisk.inference.scenario import apply_scenario

    base = np.full((8, 8), 0.2, dtype="float32")
    p1 = apply_scenario(base, {"return_period_years": 1}).mean()
    p10 = apply_scenario(base, {"return_period_years": 10}).mean()
    p100 = apply_scenario(base, {"return_period_years": 100}).mean()
    assert p1 < p10 < p100  # выше повторяемость → выше вероятность
    assert abs(p10 - 0.2) < 1e-5  # p10y — якорь (β=0)


def test_predict_invalid_bbox_400(client, inference_env):
    # E<W — геометрически некорректный bbox.
    r = client.post(
        "/api/predict",
        json={"bbox": [10.0, 5.0, 9.0, 6.0], "scenario_id": "p100y", "model_version": "unet-test"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_bbox"


# ── Этап 4: HTML-роуты и экспорт ──────────────────────────────────────────────


def test_ui_predict_returns_fragment(client, inference_env):
    bbox = ",".join(str(x) for x in _inner_bbox(inference_env["coverage"]))
    r = client.post(
        "/ui/predict",
        data={"scenario_id": "p100y", "model_version": "unet-test", "bbox": bbox},
    )
    assert r.status_code == 200
    assert 'id="prediction-data"' in r.text
    assert "data-png-url" in r.text
    assert "Площадь" in r.text  # OOB-фрагмент агрегатов


def test_ui_predict_out_of_coverage_message(client, inference_env):
    w, s, _e, _n = inference_env["coverage"]
    bbox = f"{w + 20},{s + 5},{w + 21},{s + 6}"
    r = client.post(
        "/ui/predict",
        data={"scenario_id": "p100y", "model_version": "unet-test", "bbox": bbox},
    )
    assert r.status_code == 200
    assert "вне зоны покрытия" in r.text


def test_run_meta_and_export_zip(client, inference_env):
    import zipfile

    from floodrisk.settings import settings

    bbox = _inner_bbox(inference_env["coverage"])
    rid = client.post(
        "/api/predict", json={"bbox": bbox, "scenario_id": "p100y", "model_version": "unet-test"}
    ).json()["run_id"]

    meta = client.get(f"/api/runs/{rid}")
    assert meta.status_code == 200
    assert meta.json()["scenario_id"] == "p100y"

    ex = client.post(f"/api/runs/{rid}/export")
    assert ex.status_code == 200
    assert ex.json()["export_url"].endswith(f"{rid}.zip")

    zp = settings.exports_dir / f"{rid}.zip"
    assert zp.exists()
    with zipfile.ZipFile(zp) as zf:
        names = set(zf.namelist())
    assert {"prediction.tif", "aggregates.geojson", "report.pdf"} <= names


def test_ui_export_triggers_download(client, inference_env):
    bbox = _inner_bbox(inference_env["coverage"])
    rid = client.post(
        "/api/predict", json={"bbox": bbox, "scenario_id": "p100y", "model_version": "unet-test"}
    ).json()["run_id"]
    r = client.post("/ui/export", data={"run_id": rid})
    assert r.status_code == 200
    assert "Скачать ZIP" in r.text
    assert "download-ready" in r.headers.get("HX-Trigger", "")


# ── Этап 5: объяснимость (FR-10) ──────────────────────────────────────────────


def _center(cov: list[float]) -> tuple[float, float]:
    w, s, e, n = cov
    return (s + n) / 2, (w + e) / 2  # lat, lon


def _make_run(client, cov) -> str:
    return client.post(
        "/api/predict",
        json={"bbox": _inner_bbox(cov), "scenario_id": "p100y", "model_version": "unet-test"},
    ).json()["run_id"]


def test_explain_returns_ranking_and_attributions(client, inference_env):
    from floodrisk.settings import settings

    rid = _make_run(client, inference_env["coverage"])
    lat, lon = _center(inference_env["coverage"])
    r = client.post("/api/explain", json={"run_id": rid, "lat": lat, "lon": lon})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["explanation_id"], int)
    assert len(body["ranking"]) == 5  # top-5 из 7 признаков
    assert all({"feature", "label", "importance"} <= set(item) for item in body["ranking"])
    assert len(body["attribution_layers"]) == 5  # U-Net → растры атрибуции
    top = body["attribution_layers"][0]
    assert top["png_url"].endswith(".png")
    assert (settings.runs_dir / rid / "attribution" / f"{top['feature']}.png").exists()


def test_explain_unknown_run_404(client, inference_env):
    lat, lon = _center(inference_env["coverage"])
    r = client.post("/api/explain", json={"run_id": "ghost", "lat": lat, "lon": lon})
    assert r.status_code == 404


def test_explain_point_out_of_coverage_422(client, inference_env):
    rid = _make_run(client, inference_env["coverage"])
    r = client.post("/api/explain", json={"run_id": rid, "lat": 0.0, "lon": 0.0})
    assert r.status_code == 422


def test_ui_explain_returns_panel(client, inference_env):
    rid = _make_run(client, inference_env["coverage"])
    lat, lon = _center(inference_env["coverage"])
    r = client.post("/ui/explain", data={"run_id": rid, "lat": lat, "lon": lon})
    assert r.status_code == 200
    assert "Важность признаков" in r.text
    assert 'id="explanation-data"' in r.text
