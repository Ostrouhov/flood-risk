"""Platt-калибровка: чистые функции на numpy + sklearn, без [ml]/сети."""

from __future__ import annotations

import numpy as np

from floodrisk.ml.calibrate import (
    apply_platt,
    calibrate_probs,
    fit_platt,
    prob_to_logit,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def test_prob_to_logit_inverse_and_clip():
    # обратимость в середине диапазона
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    recovered = _sigmoid(prob_to_logit(p))
    assert np.allclose(recovered, p, atol=1e-6)
    # края не дают inf/nan благодаря клипу
    edge = prob_to_logit(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(edge))


def test_apply_platt_identity_params():
    # a=1, b=0 → sigmoid(logit) возвращает исходную вероятность
    p = np.array([0.2, 0.5, 0.8])
    out = apply_platt(prob_to_logit(p), 1.0, 0.0)
    assert np.allclose(out, p, atol=1e-6)


def test_fit_platt_reduces_brier_on_overconfident():
    # Синтетика: истинная частота ~ sigmoid(true_logit), но модель ЗАВЫШАЕТ —
    # умножает логит на 3 (как U-Net с pos_weight). Platt должен это сжать.
    rng = np.random.default_rng(0)
    n = 20000
    true_logit = rng.normal(-2.0, 1.5, size=n)  # редкий класс
    y = (rng.uniform(size=n) < _sigmoid(true_logit)).astype("int64")
    overconf_logit = true_logit * 3.0  # завышение уверенности
    raw_p = _sigmoid(overconf_logit)

    a, b = fit_platt(overconf_logit, y)
    cal_p = apply_platt(overconf_logit, a, b)

    assert _brier(y, cal_p) < _brier(y, raw_p)
    # калибровка монотонна → порядок вероятностей сохраняется
    order_raw = np.argsort(raw_p)
    order_cal = np.argsort(cal_p)
    assert np.array_equal(order_raw, order_cal)


def test_calibrate_probs_wrapper_val_to_test():
    rng = np.random.default_rng(1)
    n = 10000
    tl_val = rng.normal(-2.0, 1.5, size=n)
    y_val = (rng.uniform(size=n) < _sigmoid(tl_val)).astype("int64")
    val_p = _sigmoid(tl_val * 3.0)

    tl_test = rng.normal(-2.0, 1.5, size=n)
    y_test = (rng.uniform(size=n) < _sigmoid(tl_test)).astype("int64")
    test_p = _sigmoid(tl_test * 3.0)

    cal_p = calibrate_probs(val_p, y_val, test_p)
    assert _brier(y_test, cal_p) < _brier(y_test, test_p)
    assert np.all((cal_p >= 0.0) & (cal_p <= 1.0))
