"""Метрики сегментации и bootstrap-CI. См. SRS §7.4, §7.5, OQ-6.

Все функции работают на пуле пикселей test-split. Bootstrap-CI ресэмплит
по ТАЙЛАМ (а не пикселям): пиксели внутри тайла пространственно коррелированы,
тайл — честная независимая единица.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

EPS = 1e-9


def iou_score(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> float:
    pred = prob >= thr
    t = y_true > 0.5
    inter = np.logical_and(pred, t).sum()
    union = np.logical_or(pred, t).sum()
    return float(inter / (union + EPS))


def f1_score_bin(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> float:
    pred = prob >= thr
    t = y_true > 0.5
    tp = np.logical_and(pred, t).sum()
    fp = np.logical_and(pred, ~t).sum()
    fn = np.logical_and(~pred, t).sum()
    return float(2 * tp / (2 * tp + fp + fn + EPS))


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def _safe_ap(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, prob))


METRIC_FNS = {
    "IoU": iou_score,
    "F1": f1_score_bin,
    "ROC-AUC": lambda yt, p: _safe_auc(yt, p),
    "PR-AUC": lambda yt, p: _safe_ap(yt, p),
    "Brier": lambda yt, p: (
        float(brier_score_loss(yt, p)) if len(np.unique(yt)) > 0 else float("nan")
    ),
}


def compute_all(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {name: fn(y_true, prob) for name, fn in METRIC_FNS.items()}


def reliability_table(
    y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, float]]:
    """Таблица калибровки (reliability): по бинам предсказанной вероятности.

    Делит [0,1] на ``n_bins`` равных бинов; для каждого непустого бина — доля пула,
    средняя предсказанная вероятность и наблюдаемая частота положительного класса.
    Идеальная калибровка: mean_pred ≈ obs_freq. Дополняет Brier (агрегат) разбивкой.

    Возвращает список dict: {bin_lo, bin_hi, count, frac, mean_pred, obs_freq}.
    """
    y = np.asarray(y_true).reshape(-1) > 0.5
    p = np.asarray(prob, dtype="float64").reshape(-1)
    total = p.shape[0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = int(np.count_nonzero(sel))
        if count == 0:
            continue
        rows.append(
            {
                "bin_lo": round(float(lo), 2),
                "bin_hi": round(float(hi), 2),
                "count": count,
                "frac": round(count / total, 4) if total else 0.0,
                "mean_pred": round(float(p[sel].mean()), 4),
                "obs_freq": round(float(y[sel].mean()), 4),
            }
        )
    return rows


def bootstrap_ci(
    per_tile_true: list[np.ndarray],
    per_tile_prob: list[np.ndarray],
    metric: str,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    max_pixels: int = 80_000,
) -> tuple[float, float]:
    """95%-CI метрики ресэмплингом тайлов (с возвращением). Возвращает (lo, hi).

    Внутри каждого ресэмпла пул пикселей подвыбирается до ``max_pixels`` —
    иначе сортировки ROC-AUC/PR-AUC на ~2М пикселей ×1000 неприемлемо медленны.
    Точечные оценки (compute_all) считаются на полном пуле, без подвыборки.
    """
    fn = METRIC_FNS[metric]
    rng = np.random.default_rng(seed)
    n = len(per_tile_true)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = np.concatenate([per_tile_true[i] for i in idx])
        pp = np.concatenate([per_tile_prob[i] for i in idx])
        if yt.shape[0] > max_pixels:
            sub = rng.integers(0, yt.shape[0], size=max_pixels)
            yt = yt[sub]
            pp = pp[sub]
        v = fn(yt, pp)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def ci_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """True, если интервалы пересекаются."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    if any(np.isnan([lo_a, hi_a, lo_b, hi_b])):
        return True  # неопределённость трактуем как пересечение (консервативно)
    return not (hi_a < lo_b or hi_b < lo_a)
