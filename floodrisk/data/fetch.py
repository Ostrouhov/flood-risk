"""Загрузка сырых геоданных по bbox + временному окну (FR-1, SRS §6.1, §10.1).

Идемпотентность: файл скачивается, только если его ещё нет; для каждого источника
пишется провенанс (url, fetched_at, sha256) в ``data/raw/_provenance.yaml``,
который затем подхватывает :func:`floodrisk.data.manifest.build_manifest`.

Растровые источники берём единообразно через STAC Microsoft Planetary Computer
(анонимно): Copernicus DEM GLO-30 (``cop-dem-glo-30``), ESA WorldCover 2021
(``esa-worldcover``), JRC Global Surface Water (``jrc-gsw``), Sentinel-1 GRD
(``sentinel-1-grd``). Все COG читаем по окну bbox через /vsicurl и НЕ качаем
целиком. DEM/WorldCover/JRC геокодированы аффинно; Sentinel-1 GRD привязан GCP —
геокодируем по тай-поинтам через WarpedVRT. OSM — Overpass. ERA5-Land — CDS API.

Сетевые клиенты импортируются лениво. Тяжёлые выгрузки — под ``make data`` (§14.5).
"""

from __future__ import annotations

import math
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import yaml

from floodrisk.data.config import DataConfig
from floodrisk.data.manifest import DATA_DIR, sha256_file

RAW_DIR = DATA_DIR / "raw"
PROVENANCE_PATH = RAW_DIR / "_provenance.yaml"

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
S1_COLLECTION = "sentinel-1-grd"  # глобальный GRD; RTC Сибирь не покрывает
MAX_DRY_SCENES = 6  # для медианного сухого фона (change detection) хватает нескольких


# ──────────────── низкоуровневые помощники ────────────────


def _download(url: str, dest: Path, *, timeout: int = 300) -> Path:
    """Скачать url в dest целиком, если файла ещё нет (идемпотентно)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def _bbox_int_tiles(bbox: list[float], step: int = 1) -> list[tuple[int, int]]:
    """Целочисленные (lat, lon) левые-нижние углы тайлов step°, покрывающих bbox."""
    w, s, e, n = bbox
    lats = range(math.floor(s / step) * step, math.floor(n / step) * step + 1, step)
    lons = range(math.floor(w / step) * step, math.floor(e / step) * step + 1, step)
    return [(lat, lon) for lat in lats for lon in lons]


def _prov_entry(url: str, path: Path) -> dict:
    return {
        "url": url.split("?")[0],  # без SAS-токена/query
        "fetched_at": date.today().isoformat(),
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _pad_clamp(win, width: int, height: int, pad_px: int):
    """Окружить окно паддингом и подрезать к границам растра."""
    from rasterio.windows import Window

    win = win.round_offsets().round_lengths()
    return Window(
        max(0, win.col_off - pad_px),
        max(0, win.row_off - pad_px),
        min(width, win.width + 2 * pad_px),
        min(height, win.height + 2 * pad_px),
    )


def _write_window(reader, win, dest: Path) -> Path:
    """Прочитать окно из открытого растра/VRT и записать маленький GeoTIFF."""
    import rasterio

    data = reader.read(window=win)
    profile = reader.profile.copy()
    profile.update(
        driver="GTiff",
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        transform=reader.window_transform(win),
        compress="deflate",
    )
    tmp = dest.with_suffix(dest.suffix + ".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
    tmp.replace(dest)
    return dest


def _clip_remote_cog(href: str, bbox: list[float], dest: Path, *, pad_px: int = 8) -> Path | None:
    """Вырезать окно bbox(WGS84) из удалённого аффинно-геокодированного COG.

    Для DEM/WorldCover/JRC. Возвращает None, если растр без CRS (не наш случай).
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    w, s, e, n = bbox
    with rasterio.open(href) as src:
        if src.crs is None or src.transform.is_identity:
            return None
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, w, s, e, n)
        win = _pad_clamp(
            from_bounds(left, bottom, right, top, src.transform), src.width, src.height, pad_px
        )
        return _write_window(src, win, dest)


def _clip_remote_s1(href: str, bbox: list[float], dest: Path, *, pad_px: int = 8) -> Path | None:
    """Вырезать окно bbox из GRD-сцены S1, геокодируя по GCP через WarpedVRT.

    Sentinel-1 GRD в Planetary Computer привязан тай-поинтами (GCP, crs=None),
    а не аффинной геопривязкой. WarpedVRT варпит сцену в gcp_crs (EPSG:4326),
    затем читаем окно bbox. Возвращает None, если у сцены нет GCP.
    """
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    w, s, e, n = bbox
    with rasterio.open(href) as src:
        gcps, gcp_crs = src.gcps
        if not gcps:
            return None
        with WarpedVRT(src, src_crs=gcp_crs) as vrt:
            win = _pad_clamp(from_bounds(w, s, e, n, vrt.transform), vrt.width, vrt.height, pad_px)
            return _write_window(vrt, win, dest)


def _pc_catalog():
    """Открыть каталог Planetary Computer с подписью ассетов (ленивый импорт)."""
    import planetary_computer as pc
    import pystac_client

    return pystac_client.Client.open(PC_STAC_URL, modifier=pc.sign_inplace)


def _stac_items(catalog, *, collections, bbox, dt=None, retries: int = 6):
    """STAC-поиск с ретраями: PC иногда отдаёт пустую страницу при троттлинге."""
    import time

    last: list = []
    for attempt in range(retries):
        kwargs = {"collections": collections, "bbox": bbox}
        if dt is not None:
            kwargs["datetime"] = dt
        items = list(catalog.search(**kwargs).items())
        if items:
            return items
        last = items
        time.sleep(2 * (attempt + 1))
    return last


def _bbox_tag(bbox: list[float]) -> str:
    """Короткий тег bbox для имени файла — чтобы клипы ОДНОГО STAC-тайла под РАЗНЫЕ регионы
    (общий ``raw/``) не коллизили по имени (напр. 10°-тайл JRC GSW покрывает 2 региона)."""
    w, s, e, n = bbox
    return f"{w:.1f}_{s:.1f}_{e:.1f}_{n:.1f}".replace("-", "m").replace(".", "p")


def _fetch_pc_asset(
    catalog, collection: str, asset_key: str, bbox: list[float], out_dir: Path, *, dt=None
) -> dict:
    """Скачать (обрезкой по bbox) один named-ассет у всех сцен коллекции в bbox."""
    entries: dict = {}
    items = _stac_items(catalog, collections=[collection], bbox=bbox, dt=dt)
    if not items:
        raise RuntimeError(f"PC: нет сцен {collection} для bbox={bbox} (dt={dt})")
    tag = _bbox_tag(bbox)
    for item in items:
        if asset_key not in item.assets:
            continue
        href = item.assets[asset_key].href
        # bbox-тег в имени: один и тот же тайл, обрезанный под разные регионы, → разные файлы.
        dest = out_dir / collection / f"{item.id}__{tag}.tif"
        if _clip_remote_cog(href, bbox, dest) is None:
            continue
        entries[f"{collection}:{item.id}"] = _prov_entry(href, dest)
    return entries


# ──────────────── источники ────────────────


def fetch_dem(bbox: list[float], out_dir: Path, catalog=None) -> dict:
    """Copernicus DEM GLO-30 через Planetary Computer (asset ``data``)."""
    return _fetch_pc_asset(catalog or _pc_catalog(), "cop-dem-glo-30", "data", bbox, out_dir)


def fetch_worldcover(bbox: list[float], out_dir: Path, catalog=None) -> dict:
    """ESA WorldCover 2021 через Planetary Computer (asset ``map``)."""
    return _fetch_pc_asset(
        catalog or _pc_catalog(), "esa-worldcover", "map", bbox, out_dir, dt="2021-01-01/2021-12-31"
    )


def fetch_jrc_gsw(bbox: list[float], out_dir: Path, catalog=None) -> dict:
    """JRC Global Surface Water, слой occurrence, через Planetary Computer."""
    return _fetch_pc_asset(catalog or _pc_catalog(), "jrc-gsw", "occurrence", bbox, out_dir)


def fetch_osm_water(bbox: list[float], out_dir: Path) -> dict:
    """Водные объекты OSM через Overpass API → Overpass JSON (без токена)."""
    w, s, e, n = bbox
    query = (
        "[out:json][timeout:180];("
        f'way["natural"="water"]({s},{w},{n},{e});'
        f'relation["natural"="water"]({s},{w},{n},{e});'
        f'way["waterway"]({s},{w},{n},{e});'
        f'way["landuse"="reservoir"]({s},{w},{n},{e});'
        ");out geom;"
    )
    dest = out_dir / "osm" / "water.osm.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 0):
        data = urllib.parse.urlencode({"data": query}).encode()
        # Overpass отклоняет запросы без User-Agent (HTTP 406)
        req = urllib.request.Request(
            OVERPASS_URL, data=data, headers={"User-Agent": "floodrisk/0.1 (research prototype)"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            dest.write_bytes(resp.read())
    return {"osm_water": _prov_entry(OVERPASS_URL, dest)}


def fetch_sentinel1(bbox: list[float], event, out_dir: Path, catalog=None) -> dict:
    """Sentinel-1 GRD из Planetary Computer по STAC (анонимно, SAS).

    Скачивает (обрезкой по bbox) VV/VH ассеты сцен в окнах события (flood/dry),
    раскладывая по ``sentinel1/<event_id>/{flood,dry}/``. GRD привязан GCP —
    геокодируем по тай-поинтам в EPSG:4326. uint16 амплитуда; перевод в dB и
    репроекция на целевую сетку — на этапе разметки (labels_s1/pipeline).
    """
    from rasterio.errors import RasterioIOError

    catalog = catalog or _pc_catalog()
    entries: dict = {}
    # flood — все сцены (нужен пик разлива); dry — достаточно нескольких для медианы
    windows = {"flood": (event.flood_window, None), "dry": (event.dry_window, MAX_DRY_SCENES)}
    for tag, ((start, end), cap) in windows.items():
        items = _stac_items(
            catalog,
            collections=[S1_COLLECTION],
            bbox=bbox,
            dt=f"{start.isoformat()}/{end.isoformat()}",
        )
        if cap is not None:
            items = items[:cap]
        for item in items:
            for pol in ("vv", "vh"):
                if pol not in item.assets:
                    continue
                href = item.assets[pol].href
                dest = out_dir / "sentinel1" / event.event_id / tag / f"{item.id}_{pol}.tif"
                try:
                    if _clip_remote_s1(href, bbox, dest) is None:
                        continue  # сцена без GCP — пропускаем
                except RasterioIOError as exc:
                    # обрыв чтения тайла с Azure после ретраев GDAL — пропускаем сцену,
                    # не валим весь fetch (остальных сцен хватает для композита)
                    print(f"[fetch] пропуск {item.id}_{pol}: {exc}", file=sys.stderr)
                    dest.unlink(missing_ok=True)
                    continue
                entries[f"s1_{tag}_{item.id}_{pol}"] = _prov_entry(href, dest)
    return entries


def fetch_era5(bbox: list[float], event, out_dir: Path) -> dict:
    """ERA5-Land total_precipitation через CDS API (нужен CDS_API_KEY)."""
    import cdsapi

    from floodrisk.settings import settings

    if not settings.cds_api_key:
        raise RuntimeError("CDS_API_KEY не задан в .env — выгрузка ERA5 невозможна")
    w, s, e, n = bbox
    dest = out_dir / "era5" / f"{event.event_id}_tp.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 0):
        client = cdsapi.Client()
        start, end = event.flood_window
        client.retrieve(
            "reanalysis-era5-land",
            {
                "variable": "total_precipitation",
                "area": [n, w, s, e],  # CDS-формат: N, W, S, E
                "date": f"{start.isoformat()}/{end.isoformat()}",
                "time": [f"{h:02d}:00" for h in range(24)],
                "format": "netcdf",
            },
            str(dest),
        )
    return {f"era5_{event.event_id}": _prov_entry("cds:reanalysis-era5-land", dest)}


def fetch_all(cfg: DataConfig, raw_dir: Path = RAW_DIR, *, skip_era5: bool = False) -> dict:
    """Скачать все источники для пилота и записать провенанс. Возвращает провенанс."""
    bbox = cfg.pilot_bbox_wgs84
    catalog = _pc_catalog()
    provenance: dict = {}
    provenance.update(fetch_dem(bbox, raw_dir, catalog))
    provenance.update(fetch_worldcover(bbox, raw_dir, catalog))
    provenance.update(fetch_jrc_gsw(bbox, raw_dir, catalog))
    provenance.update(fetch_osm_water(bbox, raw_dir))
    for event in cfg.events:
        provenance.update(fetch_sentinel1(bbox, event, raw_dir, catalog))
        if not skip_era5:
            provenance.update(fetch_era5(bbox, event, raw_dir))

    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.write_text(
        yaml.safe_dump(provenance, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    return provenance


def load_provenance(path: Path = PROVENANCE_PATH) -> dict:
    """Прочитать провенанс выгрузки (для manifest.build). {} если нет."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
