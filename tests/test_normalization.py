"""Единый модуль нормировки (core) — общий для обучения и инференса."""

from __future__ import annotations

import json

import numpy as np

from floodrisk.normalization import load_stats, normalize


def test_load_stats_shapes(tmp_path):
    p = tmp_path / "norm_stats.json"
    p.write_text(json.dumps({"mean": [1.0, 2.0, 3.0], "std": [2.0, 4.0, 0.5]}))
    mean, std = load_stats(p)
    assert mean.shape == (3, 1, 1)
    assert std.shape == (3, 1, 1)
    assert mean.dtype == np.float32


def test_normalize_applies_zscore(tmp_path):
    mean = np.array([1.0, 10.0]).reshape(2, 1, 1)
    std = np.array([2.0, 5.0]).reshape(2, 1, 1)
    x = np.stack([np.full((4, 4), 3.0), np.full((4, 4), 20.0)]).astype("float32")  # [2,4,4]
    out = normalize(x, mean, std)
    assert out.dtype == np.float32
    assert np.allclose(out[0], 1.0)  # (3-1)/2
    assert np.allclose(out[1], 2.0)  # (20-10)/5
