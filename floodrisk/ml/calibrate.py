"""Калибровка вероятностей методом Platt (логистическая регрессия по логиту).

ТОЛЬКО для отчёта (`ml/evaluate.py`): U-Net обучен с ``pos_weight=20`` на редком
классе (~1.2%) и систематически ЗАВЫШАЕТ p (видно в reliability — ``mean_pred`` ≫
``obs_freq``). Platt-калибровка чинит это завышение, и здесь мы измеряем эффект
(Brier до/после). Преобразование монотонно → ранжирование (ROC-AUC/PR-AUC) НЕ меняется.

ВАЖНО: в serving/карту/хотспоты калибровку НЕ вносим — сжатие p к базовой частоте
убило бы визуал карты и порог хотспотов p≥0.5. Параметры (a, b) подбираются на VAL,
применяются на TEST. См. SRS §7.4 и итоги stageB-and-roadmap.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def prob_to_logit(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Восстановить логит из вероятности: ``log(p/(1-p))`` с клипом p в ``[eps, 1-eps]``.

    Probs U-Net приходят из sigmoid, baseline — из ``predict_proba``; для обоих логит —
    монотонная функция p, так что Platt поверх восстановленного логита корректен.
    """
    p = np.clip(np.asarray(p, dtype="float64").reshape(-1), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_platt(logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Подобрать параметры Platt (a, b) на признаке-логите.

    Логистическая регрессия с почти нулевой регуляризацией (``C=1e9``) на единственном
    признаке. Возвращает ``(a, b)`` для ``sigmoid(a*logit + b)``. Подбирать на VAL.
    """
    x = np.asarray(logit, dtype="float64").reshape(-1, 1)
    yt = (np.asarray(y).reshape(-1) > 0.5).astype("int64")
    clf = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    clf.fit(x, yt)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def apply_platt(logit: np.ndarray, a: float, b: float) -> np.ndarray:
    """Применить Platt: ``sigmoid(a*logit + b)``."""
    z = a * np.asarray(logit, dtype="float64").reshape(-1) + b
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_probs(
    val_prob: np.ndarray,
    val_true: np.ndarray,
    test_prob: np.ndarray,
) -> np.ndarray:
    """Откалибровать test-вероятности: fit Platt на VAL, apply на TEST.

    Удобная обёртка для ``run_eval``: ``val_prob`` и ``val_true`` — пул VAL,
    ``test_prob`` — пул TEST; возврат — калиброванные TEST-вероятности.
    """
    a, b = fit_platt(prob_to_logit(val_prob), val_true)
    return apply_platt(prob_to_logit(test_prob), a, b)
