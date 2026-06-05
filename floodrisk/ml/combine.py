"""Объединение per-region датасетов в один обучающий index (мультирегион, атака на M-2).

SCAFFOLDING под обучение v2 на 2+ событиях: модуль покрыт тестами (tests/test_combine.py)
и понятен ``ml.data.tile_paths``, но в текущем релизе НЕ задействован (v1 = одно событие,
Тулун). Оставлен намеренно как фундамент для расширения датасета (см. reports/acceptance.md).

Каждый регион обработан отдельно своим прогоном пайплайна (своя проекция/сетка):
``data/processed/<rv>/{tiles, labels/<event>, index.parquet}``. Единый грид на разные регионы
не натянуть, поэтому объединяем на уровне ОБУЧЕНИЯ: конкатенируем index'ы регионов в
``data/processed/<out>/index.parquet``, проставляя пер-строчные ``tile_path``/``label_path``
(относительно project_root) и колонку ``region``. ``ml.data.tile_paths`` понимает эти колонки.

Split train/val/test уже назначен гео-методом ВНУТРИ каждого региона
(``preprocess.assign_geographic_splits``) → простой конкат сохраняет непересекаемость по регионам.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from floodrisk.settings import settings


def _processed(version: str) -> Path:
    return settings.project_root / "data" / "processed" / version


def _resolve_event(df: pd.DataFrame, pdir: Path) -> str:
    """Имя каталога события лейблов (``labels/<event>/``) для региона."""
    if "event_ids" in df.columns:
        for raw in df["event_ids"].astype(str):
            first = raw.split(",")[0].strip()
            if first and first != "nan" and (pdir / "labels" / first).exists():
                return first
    labels = pdir / "labels"
    if labels.exists():
        for p in sorted(labels.iterdir()):
            if p.is_dir() and (p / "flood_mask.tif").exists():
                return p.name
    raise FileNotFoundError(f"не найдено событие лейблов в {labels}")


def build_combined_index(region_versions: list[str], out_version: str = "v2") -> Path:
    """Объединить index'ы регионов → ``processed/<out_version>/index.parquet``.

    Возвращает путь к записанному index. Пути тайлов/лейблов — относительные от project_root.
    """
    frames: list[pd.DataFrame] = []
    for rv in region_versions:
        pdir = _processed(rv)
        idx_path = pdir / "index.parquet"
        if not idx_path.exists():
            raise FileNotFoundError(f"нет index.parquet для региона {rv}: {idx_path}")
        df = pd.read_parquet(idx_path).copy()
        event = _resolve_event(df, pdir)
        rel = Path("data") / "processed" / rv
        # Пути считаем по ИСХОДНОМУ tile_id (файлы названы t_rRRRcCCC.tif), затем префиксуем id.
        df["tile_path"] = df["tile_id"].map(lambda t, r=rel: (r / "tiles" / f"{t}.tif").as_posix())
        df["label_path"] = df["tile_id"].map(
            lambda t, r=rel, e=event: (r / "labels" / e / f"{t}.tif").as_posix()
        )
        df["region"] = rv
        df["tile_id"] = rv + "/" + df["tile_id"].astype(str)  # уникальность id между регионами
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    out_dir = _processed(out_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path
