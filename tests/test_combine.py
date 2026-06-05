"""Объединение мультирегиональных датасетов (combine) + пер-строчный tile_paths.

Чистая логика на синтетических index.parquet (без растров/сети/[ml]).
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from floodrisk.ml import combine
from floodrisk.ml.data import tile_paths
from floodrisk.settings import settings

# combine работает через parquet (pandas to_parquet/read_parquet). Движок (pyarrow/
# fastparquet) идёт с extras [data]/[ml] и отсутствует в CI (core+[dev]) → там пропускаем.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None,
    reason="нет parquet-движка (pyarrow/fastparquet); тесты combine требуют extras [data]/[ml]",
)


def _make_region(root, rv, event, tile_ids, splits):
    pdir = root / "data" / "processed" / rv
    (pdir / "tiles").mkdir(parents=True)
    (pdir / "labels" / event).mkdir(parents=True)
    df = pd.DataFrame(
        {
            "tile_id": tile_ids,
            "split": splits,
            "event_ids": event,
            "min_lon": 0.0,
            "min_lat": 0.0,
            "max_lon": 1.0,
            "max_lat": 1.0,
        }
    )
    df.to_parquet(pdir / "index.parquet", index=False)


def test_build_combined_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", tmp_path)
    _make_region(tmp_path, "v1", "2019-tulun-iya", ["t_r000c000", "t_r000c001"], ["train", "val"])
    _make_region(
        tmp_path, "kansk", "2019-kansk-kan", ["t_r000c000", "t_r001c000"], ["train", "test"]
    )

    out = combine.build_combined_index(["v1", "kansk"], "v2")
    assert out == tmp_path / "data" / "processed" / "v2" / "index.parquet"
    df = pd.read_parquet(out)

    assert len(df) == 4  # 2 + 2
    assert {"tile_path", "label_path", "region"} <= set(df.columns)
    assert set(df["region"]) == {"v1", "kansk"}
    # tile_id префиксован регионом → уникален между регионами
    assert "v1/t_r000c000" in set(df["tile_id"])
    assert "kansk/t_r000c000" in set(df["tile_id"])
    # пути собраны по исходному tile_id и указывают на нужный регион/событие
    row = df[df["tile_id"] == "kansk/t_r001c000"].iloc[0]
    assert row["tile_path"] == "data/processed/kansk/tiles/t_r001c000.tif"
    assert row["label_path"] == "data/processed/kansk/labels/2019-kansk-kan/t_r001c000.tif"
    # split-метки сохранены при конкатенации
    assert sorted(df["split"]) == ["test", "train", "train", "val"]


def test_tile_paths_uses_per_row_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", tmp_path)
    _make_region(tmp_path, "v1", "2019-tulun-iya", ["t_r000c000", "t_r000c001"], ["train", "val"])
    _make_region(tmp_path, "kansk", "2019-kansk-kan", ["t_r000c000"], ["train"])
    out = combine.build_combined_index(["v1", "kansk"], "v2")

    cfg = {"data": {"index": str(out)}}  # tile_dir/label_dir намеренно отсутствуют
    train = tile_paths(cfg, "train")
    assert len(train) == 2  # v1/t_r000c000 + kansk/t_r000c000
    ids = {t[0] for t in train}
    assert ids == {"v1/t_r000c000", "kansk/t_r000c000"}
    # абсолютный резолв относительно project_root
    by_id = {t[0]: t for t in train}
    _tid, tpath, lpath = by_id["kansk/t_r000c000"]
    assert tpath == tmp_path / "data" / "processed" / "kansk" / "tiles" / "t_r000c000.tif"
    assert lpath.name == "t_r000c000.tif"
    assert "2019-kansk-kan" in str(lpath)


def test_tile_paths_backcompat_single_region(tmp_path):
    # Без колонок tile_path/label_path → старое поведение (tile_dir/label_dir + tile_id).
    df = pd.DataFrame({"tile_id": ["t_r000c000"], "split": ["train"]})
    idx = tmp_path / "index.parquet"
    df.to_parquet(idx, index=False)
    cfg = {
        "data": {
            "index": str(idx),
            "tile_dir": "/data/tiles",
            "label_dir": "/data/labels/ev",
        }
    }
    out = tile_paths(cfg, "train")
    assert len(out) == 1
    _tid, tpath, lpath = out[0]
    assert tpath.as_posix().endswith("data/tiles/t_r000c000.tif")
    assert lpath.as_posix().endswith("data/labels/ev/t_r000c000.tif")
