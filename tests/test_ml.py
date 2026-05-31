"""Лёгкие тесты ML-слоя Этапа 2 (без обучения моделей). См. SRS §7, FR-5..FR-7."""

from __future__ import annotations

import numpy as np
import pytest

# config/metrics зависят только от core (yaml/numpy/sklearn) → импортируем на верхнем уровне.
# data.py тянет pandas (extras [data]/[ml]), которого в core-CI нет → импорт ленивый,
# а сам тест помечен ml_smoke (CI его деселектит).
from floodrisk.ml import config as cfgmod
from floodrisk.ml import metrics as M


def test_config_type_detection():
    assert cfgmod.is_unet(cfgmod.load_yaml("configs/unet_v1.yaml"))
    assert cfgmod.is_baseline(cfgmod.load_yaml("configs/baseline_v1.yaml"))
    assert len(cfgmod.CHANNEL_NAMES) == 7


def test_tracking_uri_windows_safe():
    # file://./x, file:./x, plain — всё → абсолютный file:/// URI.
    for raw in ("file://./mlruns", "file:./mlruns", "mlruns", "./mlruns"):
        uri = cfgmod.tracking_uri({"mlflow": {"tracking_uri": raw}})
        assert uri.startswith("file:///")
        assert "mlruns" in uri
    # remote/db backends пропускаются как есть.
    assert cfgmod.tracking_uri({"mlflow": {"tracking_uri": "sqlite:///x.db"}}) == "sqlite:///x.db"


def test_iou_and_f1_perfect_and_zero():
    y = np.array([0, 0, 1, 1], dtype="float32")
    assert M.iou_score(y, y) == pytest.approx(1.0, abs=1e-6)
    assert M.f1_score_bin(y, y) == pytest.approx(1.0, abs=1e-6)
    inv = 1.0 - y
    assert M.iou_score(y, inv) < 0.01
    assert M.f1_score_bin(y, inv) < 0.01


def test_metrics_handle_single_class():
    y = np.zeros(10, dtype="float32")
    p = np.random.default_rng(0).random(10).astype("float32")
    res = M.compute_all(y, p)
    assert np.isnan(res["ROC-AUC"])  # AUC не определён на одном классе
    assert not np.isnan(res["Brier"])  # Brier определён


def test_bootstrap_ci_ordered_and_overlap():
    rng = np.random.default_rng(1)
    tiles_t = [rng.integers(0, 2, 200).astype("float32") for _ in range(8)]
    tiles_p = [t * 0.9 + 0.05 for t in tiles_t]  # почти идеальные предсказания
    lo, hi = M.bootstrap_ci(tiles_t, tiles_p, "IoU", n_boot=100, seed=42)
    assert 0.0 <= lo <= hi <= 1.0
    assert M.ci_overlap((0.1, 0.3), (0.2, 0.4)) is True
    assert M.ci_overlap((0.1, 0.2), (0.3, 0.4)) is False
    assert M.ci_overlap((float("nan"), 0.2), (0.3, 0.4)) is True  # nan → консервативно


@pytest.mark.ml_smoke
def test_augment_preserves_shape_and_alignment():
    from floodrisk.ml.data import _augment  # тянет pandas/torch — только в [ml]-окружении

    rng = np.random.default_rng(0)
    x = rng.random((7, 16, 16)).astype("float32")
    y = (rng.random((16, 16)) > 0.5).astype("float32")
    # Канал 0 == лейбл → после любых геом. преобразований выравнивание сохраняется.
    x[0] = y
    xa, ya = _augment(x.copy(), y.copy(), ["rot90", "hflip", "vflip"], rng)
    assert xa.shape == (7, 16, 16)
    assert ya.shape == (16, 16)
    assert np.allclose(xa[0], ya)
