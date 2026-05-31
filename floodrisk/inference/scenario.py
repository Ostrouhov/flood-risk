"""Модуляция карты подверженности метео-сценарием. См. SRS §7.1, решение Этапа 2/3.

Модель терреновая (7 каналов, без метео) → сценарий применяется ПОСЛЕ инференса
монотонным logit-сдвигом. Сдвиг привязан к повторяемости (return period), якорь —
базовый p10y (β=0). Это документированная эвристика, а НЕ обученная зависимость:
p100y усиливает, p1y ослабляет вероятности, ранжирование пикселей сохраняется.
"""

from __future__ import annotations

import math

import numpy as np

# Усиление сдвига на декаду повторяемости (в логитах). β = GAIN * log10(RP / 10).
DEFAULT_GAIN = 1.0


def scenario_logit_shift(params: dict, gain: float = DEFAULT_GAIN) -> float:
    """β для сценария: GAIN * log10(return_period_years / 10). p10y → 0."""
    rp = float(params.get("return_period_years", 10) or 10)
    rp = max(rp, 1e-6)
    return gain * math.log10(rp / 10.0)


def apply_scenario(prob: np.ndarray, params: dict, gain: float = DEFAULT_GAIN) -> np.ndarray:
    """p_сцен = sigmoid(logit(p_база) + β(сценарий)). Возвращает массив в [0,1]."""
    beta = scenario_logit_shift(params, gain)
    if beta == 0.0:
        return prob.astype("float32")
    eps = 1e-6
    p = np.clip(prob.astype("float64"), eps, 1 - eps)
    logit = np.log(p / (1 - p)) + beta
    return (1.0 / (1.0 + np.exp(-logit))).astype("float32")
