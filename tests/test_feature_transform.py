"""Фича-трансформ сырых каналов во вход модели (core). См. floodrisk/feature_transform.py."""

from __future__ import annotations

import numpy as np

from floodrisk import feature_transform as ft


def _raw(aspect_deg=90.0, wc=80):
    """Сырой тайл [7,H,W] с заданными aspect и worldcover."""
    h = w = 4
    raw = np.zeros((7, h, w), dtype="float32")
    raw[ft._idx("aspect")] = aspect_deg
    raw[ft._idx("worldcover")] = wc
    return raw


def test_out_channels():
    assert ft.out_channels() == len(ft.CONTINUOUS) + 2 + len(ft.WC_CLASSES) == 18


def test_transform_shape_and_aspect_sincos():
    mean = np.zeros((len(ft.CONTINUOUS), 1, 1), dtype="float32")
    std = np.ones((len(ft.CONTINUOUS), 1, 1), dtype="float32")
    out = ft.transform(_raw(aspect_deg=90.0), mean, std)
    assert out.shape == (18, 4, 4)
    # sin/cos идут сразу после непрерывных (индексы 5 и 6).
    s = len(ft.CONTINUOUS)
    assert np.allclose(out[s], 1.0, atol=1e-6)  # sin(90°)=1
    assert np.allclose(out[s + 1], 0.0, atol=1e-6)  # cos(90°)=0


def test_transform_worldcover_onehot():
    mean = np.zeros((len(ft.CONTINUOUS), 1, 1), dtype="float32")
    std = np.ones((len(ft.CONTINUOUS), 1, 1), dtype="float32")
    out = ft.transform(_raw(wc=80), mean, std)
    onehot = out[len(ft.CONTINUOUS) + 2 :]
    assert onehot.shape[0] == len(ft.WC_CLASSES)
    assert np.allclose(onehot.sum(axis=0), 1.0)  # ровно один класс активен
    active = ft.WC_CLASSES.index(80)
    assert np.allclose(onehot[active], 1.0)


def test_transform_unknown_worldcover_all_zero():
    mean = np.zeros((len(ft.CONTINUOUS), 1, 1), dtype="float32")
    std = np.ones((len(ft.CONTINUOUS), 1, 1), dtype="float32")
    out = ft.transform(_raw(wc=999), mean, std)  # класс не из набора
    onehot = out[len(ft.CONTINUOUS) + 2 :]
    assert np.allclose(onehot.sum(axis=0), 0.0)
