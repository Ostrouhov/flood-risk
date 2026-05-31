"""Манифест датасета и контрольные суммы (FR-4, SRS §6.3).

``build_manifest`` собирает ``data/manifest.yaml`` из обработанных тайлов и
провенанса выгрузки; ``verify_manifest`` пересчитывает sha256 и сообщает
расхождения. CLI ``data verify`` → exit 0 при совпадении, иначе ≠0.

Чистая логика (хеши над файлами) — тестируется без сети и GDAL.
"""

from __future__ import annotations

import contextlib
import hashlib
from datetime import date
from pathlib import Path

import yaml

from floodrisk.data.config import DataConfig
from floodrisk.settings import settings

DATA_DIR = settings.project_root / "data"
DEFAULT_MANIFEST_PATH = DATA_DIR / "manifest.yaml"

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: str | Path) -> str:
    """Потоковый sha256 файла (hex)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path, base: Path) -> str:
    """POSIX-относительный путь от base (для переносимости манифеста)."""
    return path.resolve().relative_to(base.resolve()).as_posix()


def build_manifest(
    cfg: DataConfig,
    processed_dir: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    sources_provenance: dict | None = None,
    created_at: date | None = None,
) -> dict:
    """Собрать манифест и записать его в ``manifest_path``.

    ``sources_provenance`` — словарь ``{source: {url, fetched_at, path, sha256}}``,
    который пишет :mod:`floodrisk.data.fetch`. Пути в манифесте хранятся
    относительно его расположения, чтобы verify работал из любого CWD.
    """
    processed_dir = Path(processed_dir)
    manifest_path = Path(manifest_path)
    base = manifest_path.resolve().parent

    tiles_dir = processed_dir / "tiles"
    tile_checksums = []
    for tif in sorted(tiles_dir.glob("*.tif")):
        tile_checksums.append(
            {
                "tile_id": tif.stem,
                "path": _rel(tif, base),
                "sha256": sha256_file(tif),
            }
        )

    sources: dict[str, dict] = {}
    for name, prov in (sources_provenance or {}).items():
        entry = dict(prov)
        # путь к файлу-источнику тоже делаем относительным, если он внутри base
        if "path" in entry:
            p = Path(entry["path"])
            if p.exists():
                with contextlib.suppress(ValueError):  # вне data/ — путь оставляем как есть
                    entry["path"] = _rel(p, base)
                entry.setdefault("sha256", sha256_file(p))
        sources[name] = entry

    manifest = {
        "dataset_version": cfg.dataset_version,
        "created_at": (created_at or date.today()).isoformat(),
        "pilot_bbox_wgs84": cfg.pilot_bbox_wgs84,
        "target_crs": cfg.target_crs,
        "tile_size_px": cfg.tile_size_px,
        "tile_resolution_m": cfg.tile_resolution_m,
        "sources": sources,
        "splits": {"seed": cfg.splits.seed, "method": cfg.splits.method},
        "events": [
            {
                "event_id": ev.event_id,
                "date_window": [ev.flood_window[0].isoformat(), ev.flood_window[1].isoformat()],
                "reference": ev.reference,
            }
            for ev in cfg.events
        ],
        # метод и пороги разметки S1 — требование FR-3 Done (OQ-2)
        "labels_s1": cfg.labels_s1.model_dump(),
        "tile_checksums": tile_checksums,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return manifest


def verify_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> list[str]:
    """Сверить файлы с манифестом. Возвращает список расхождений (пуст → всё ок)."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return [f"манифест не найден: {manifest_path}"]

    base = manifest_path.resolve().parent
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    problems: list[str] = []

    entries = list(manifest.get("tile_checksums", []))
    for name, src in (manifest.get("sources") or {}).items():
        if isinstance(src, dict) and src.get("path") and src.get("sha256"):
            entries.append({"tile_id": f"source:{name}", **src})

    if not entries:
        problems.append("в манифесте нет ни контрольных сумм тайлов, ни источников")

    for entry in entries:
        rel = entry.get("path")
        if not rel:
            problems.append(f"{entry.get('tile_id')}: нет поля path")
            continue
        fpath = (base / rel).resolve()
        if not fpath.exists():
            problems.append(f"{entry['tile_id']}: файл отсутствует ({rel})")
            continue
        actual = sha256_file(fpath)
        if actual != entry.get("sha256"):
            problems.append(
                f"{entry['tile_id']}: sha256 не совпадает "
                f"(ожидалось {entry.get('sha256', '')[:12]}…, получено {actual[:12]}…)"
            )
    return problems
