"""Табличный бейзлайн: RandomForest попиксельно. См. SRS §7.2.2, FR-6."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from floodrisk.ml.data import _read_label, _read_tile, load_norm_stats, tile_paths


def _sample_pixels(cfg: dict, split: str, *, per_tile: int, stratify: bool, seed: int):
    """Собирает (X[N,7], y[N]) подвыборкой пикселей с нормировкой."""
    mean, std = load_norm_stats(cfg)
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _tid, tpath, lpath in tile_paths(cfg, split):
        x = ((_read_tile(tpath) - mean) / std).reshape(7, -1).T  # [HW, 7]
        y = _read_label(lpath).reshape(-1)  # [HW]
        n_pix = y.shape[0]
        if stratify:
            pos = np.flatnonzero(y > 0.5)
            neg = np.flatnonzero(y <= 0.5)
            k = max(1, per_tile // 2)
            sel_pos = pos if len(pos) <= k else rng.choice(pos, k, replace=False)
            n_neg = min(len(neg), per_tile - len(sel_pos))
            sel_neg = rng.choice(neg, n_neg, replace=False) if n_neg > 0 else np.array([], int)
            sel = np.concatenate([sel_pos, sel_neg])
        else:
            sel = rng.choice(n_pix, min(per_tile, n_pix), replace=False)
        xs.append(x[sel])
        ys.append(y[sel])
    return np.concatenate(xs), np.concatenate(ys)


def train_baseline(cfg: dict) -> Path:
    d = cfg["data"]
    X, y = _sample_pixels(
        cfg, "train",
        per_tile=int(d.get("pixel_subsample_per_tile", 500)),
        stratify=bool(d.get("stratify_by_class", True)),
        seed=int(d.get("seed", 42)),
    )
    params = dict(cfg["model"].get("params", {}))
    clf = RandomForestClassifier(**params)
    clf.fit(X, y.astype(int))

    out = Path(cfg["artifact_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"model": clf, "channels": 7}, f)
    print(f"[baseline] обучено на {X.shape[0]} пикселях; модель → {out}")
    return out


def load_baseline(path: str | Path) -> RandomForestClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)["model"]
