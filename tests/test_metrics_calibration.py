"""reliability_table (калибровка): чистая функция на numpy, без [ml]/сети."""

from __future__ import annotations

import numpy as np

from floodrisk.ml.metrics import reliability_table


def test_reliability_perfectly_calibrated():
    # Низкие p → класс 0, высокие p → класс 1: mean_pred и obs_freq согласованы по краям.
    prob = np.array([0.05, 0.05, 0.95, 0.95], dtype="float32")
    y = np.array([0, 0, 1, 1], dtype="uint8")
    rows = reliability_table(y, prob, n_bins=10)
    # два непустых бина (первый и последний)
    assert len(rows) == 2
    lo_bin, hi_bin = rows[0], rows[1]
    assert lo_bin["obs_freq"] == 0.0 and lo_bin["mean_pred"] < 0.1
    assert hi_bin["obs_freq"] == 1.0 and hi_bin["mean_pred"] > 0.9
    # доли пула суммируются к 1, count покрывает весь пул
    assert sum(r["count"] for r in rows) == 4


def test_reliability_skips_empty_bins_and_counts_total():
    prob = np.full(20, 0.5, dtype="float32")  # всё в один бин [0.5,0.6)
    y = (np.arange(20) % 2).astype("uint8")  # половина положительных
    rows = reliability_table(y, prob, n_bins=10)
    assert len(rows) == 1
    r = rows[0]
    assert r["count"] == 20 and r["frac"] == 1.0
    assert r["obs_freq"] == 0.5
    assert r["bin_lo"] == 0.5
