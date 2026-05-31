"""Smoke-тесты data-пайплайна (Этап 1, FR-1…FR-4).

Чистая логика на синтетике — без сети и без тяжёлого GDAL. Реальные сетевые
выгрузки помечаются ``@pytest.mark.requires_network`` и в CI не гоняются (OQ-12).
"""

from __future__ import annotations

import numpy as np
import pytest

from floodrisk.data.config import DataConfig, load_data_config

# ──────────────── config.py (загрузка configs/data.yaml) ────────────────


def test_load_real_data_config() -> None:
    """Реальный configs/data.yaml парсится и проходит валидацию."""
    cfg = load_data_config()
    assert cfg.dataset_version == "v1"
    assert cfg.target_crs.startswith("EPSG:")
    assert len(cfg.pilot_bbox_wgs84) == 4
    # пилот закрыт по OQ-9 — Тулун, есть хотя бы одно событие с лейблами
    assert cfg.events, "ожидается хотя бы одно событие-источник лейблов"
    assert cfg.event("2019-tulun-iya").reference == "copernicus_ems"


def test_bbox_validation_rejects_inverted() -> None:
    with pytest.raises(ValueError, match="W<E"):
        DataConfig(
            pilot_bbox_wgs84=[38.0, 55.0, 37.0, 54.0],  # W>E
            target_crs="EPSG:32647",
        )


def test_target_crs_must_be_epsg() -> None:
    with pytest.raises(ValueError, match="EPSG"):
        DataConfig(pilot_bbox_wgs84=[100.4, 54.4, 100.95, 54.8], target_crs="utm47n")


def test_splits_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="суммироваться"):
        load_data_config_from_dict(
            {
                "pilot_bbox_wgs84": [100.4, 54.4, 100.95, 54.8],
                "target_crs": "EPSG:32647",
                "splits": {"train": 0.6, "val": 0.1, "test": 0.1},
            }
        )


def load_data_config_from_dict(d: dict) -> DataConfig:
    return DataConfig.model_validate(d)


# ──────────────── manifest.py (FR-4: build + verify) ────────────────


def _tiny_cfg() -> DataConfig:
    return DataConfig.model_validate(
        {
            "pilot_bbox_wgs84": [100.4, 54.4, 100.95, 54.8],
            "target_crs": "EPSG:32647",
            "events": [
                {
                    "event_id": "2019-tulun-iya",
                    "flood_window": ["2019-06-28", "2019-07-05"],
                    "dry_window": ["2019-05-01", "2019-06-10"],
                    "reference": "copernicus_ems",
                }
            ],
        }
    )


def _make_processed(tmp_path, n_tiles: int = 3):
    """Создать data/processed/v1/tiles/*.tif как непрозрачные байт-файлы."""
    from datetime import date

    from floodrisk.data.manifest import build_manifest

    processed = tmp_path / "processed" / "v1"
    tiles = processed / "tiles"
    tiles.mkdir(parents=True)
    for i in range(n_tiles):
        (tiles / f"t_{i:04d}.tif").write_bytes(f"fake-geotiff-{i}".encode())
    manifest_path = tmp_path / "manifest.yaml"
    build_manifest(_tiny_cfg(), processed, manifest_path, created_at=date(2026, 5, 29))
    return manifest_path, tiles


def test_build_manifest_then_verify_ok(tmp_path) -> None:
    from floodrisk.data.manifest import verify_manifest

    manifest_path, _ = _make_processed(tmp_path)
    assert verify_manifest(manifest_path) == [], "свежесобранный манифест должен сходиться"


def test_manifest_records_method_and_thresholds(tmp_path) -> None:
    """FR-3 Done: метод и пороги разметки S1 зафиксированы в манифесте (OQ-2)."""
    import yaml

    manifest_path, _ = _make_processed(tmp_path)
    m = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert m["labels_s1"]["method"] == "otsu_change_detection"
    assert m["labels_s1"]["validation"]["target"] == 0.50
    assert m["target_crs"] == "EPSG:32647"


def test_verify_fails_when_tile_mutated(tmp_path) -> None:
    """FR-4 Done: меняем один тайл → verify сообщает расхождение."""
    from floodrisk.data.manifest import verify_manifest

    manifest_path, tiles = _make_processed(tmp_path)
    (tiles / "t_0001.tif").write_bytes(b"tampered")
    problems = verify_manifest(manifest_path)
    assert problems, "verify должен поймать изменённый тайл"
    assert any("t_0001" in p for p in problems)


def test_verify_fails_when_tile_missing(tmp_path) -> None:
    from floodrisk.data.manifest import verify_manifest

    manifest_path, tiles = _make_processed(tmp_path)
    (tiles / "t_0002.tif").unlink()
    problems = verify_manifest(manifest_path)
    assert any("отсутствует" in p for p in problems)


# ──────────────── grid.py (целевая сетка, SRS §6.4) ────────────────


def test_target_grid_from_config_shape_and_crs() -> None:
    from floodrisk.data.grid import TargetGrid

    grid = TargetGrid.from_config(_tiny_cfg())
    assert grid.crs == "EPSG:32647"
    assert grid.resolution_m == 30.0
    assert grid.width > 0 and grid.height > 0
    # bbox ~50×40 км при 30 м → порядка 1000–2000 пикселей по стороне
    assert 500 < grid.width < 4000
    assert 500 < grid.height < 4000
    # transform north-up: пиксель по Y отрицательный
    assert grid.transform.a == 30.0
    assert grid.transform.e == -30.0


# ──────────────── permanent_water.py (FR-3, OQ-2) ────────────────


def test_combine_permanent_water_threshold_and_osm() -> None:
    from floodrisk.data.permanent_water import combine_permanent_water

    occ = np.array([[10.0, 60.0], [49.0, 90.0]], dtype="float32")
    mask = combine_permanent_water(occ, threshold_pct=50)
    assert mask.tolist() == [[False, True], [False, True]]

    osm = np.array([[True, False], [False, False]], dtype=bool)
    mask2 = combine_permanent_water(occ, threshold_pct=50, osm_water=osm)
    assert mask2.tolist() == [[True, True], [False, True]]


def test_combine_permanent_water_respects_nodata() -> None:
    from floodrisk.data.permanent_water import combine_permanent_water

    occ = np.array([[255.0, 80.0]], dtype="float32")  # 255 = nodata, не вода
    mask = combine_permanent_water(occ, threshold_pct=50, nodata=255.0)
    assert mask.tolist() == [[False, True]]


# ──────────────── labels_s1.py (FR-3: Otsu + change detection) ────────────────


def test_otsu_separates_bimodal() -> None:
    from floodrisk.data.labels_s1 import otsu_threshold

    rng = np.random.default_rng(0)
    water = rng.normal(-22, 1.0, 2000)  # тёмная вода
    land = rng.normal(-8, 1.5, 2000)  # яркая суша
    thr = otsu_threshold(np.concatenate([water, land]))
    assert -22 < thr < -8, f"порог должен лечь между модами, получено {thr}"


def test_detect_flood_low_backscatter_is_water() -> None:
    from floodrisk.data.labels_s1 import detect_flood

    rng = np.random.default_rng(1)
    flood = rng.normal(-7.0, 1.0, (64, 64)).astype("float32")  # суша
    flood[16:48, 16:48] = rng.normal(-22.0, 1.0, (32, 32))  # затопленный блок
    water, thr = detect_flood(flood)
    assert -22 < thr < -7
    # внутренность блока — вода, дальний угол — суша
    assert water[32, 32]
    assert not water[0, 0]
    # большинство пикселей блока распознано как вода
    assert water[16:48, 16:48].mean() > 0.9


def test_change_detection_removes_persistent_dark() -> None:
    """Перманентно тёмная не-водная поверхность (дорога) не попадает в разлив."""
    from floodrisk.data.labels_s1 import detect_flood

    rng = np.random.default_rng(2)
    flood = rng.normal(-7.0, 1.0, (64, 64)).astype("float32")
    flood[16:48, 16:48] = rng.normal(-22.0, 1.0, (32, 32))  # тёмный блок в паводок
    dry = rng.normal(-7.0, 1.0, (64, 64)).astype("float32")
    # левая половина блока была тёмной и в сухое окно (дорога), правая — посветлела
    dry[16:48, 16:32] = flood[16:48, 16:32]
    water, _ = detect_flood(flood, dry_db=dry, min_drop_db=3.0)
    # правая половина (реально залило) — вода, левая (дорога) — отброшена
    assert water[16:48, 32:48].mean() > 0.8
    assert water[16:48, 16:32].mean() < 0.2


def test_clean_speckle_removes_small_blobs() -> None:
    from floodrisk.data.labels_s1 import clean_speckle

    m = np.zeros((10, 10), dtype=bool)
    m[0, 0] = True  # одиночный спекл
    m[5:9, 5:9] = True  # блоб 16 px
    cleaned = clean_speckle(m, min_blob_px=8)
    assert not cleaned[0, 0]
    assert cleaned[5:9, 5:9].all()


def test_subtract_permanent_and_iou() -> None:
    from floodrisk.data.labels_s1 import iou, subtract_permanent

    detected = np.array([[1, 1], [1, 0]], dtype=bool)
    permanent = np.array([[1, 0], [0, 0]], dtype=bool)
    flood = subtract_permanent(detected, permanent)
    assert flood.tolist() == [[False, True], [True, False]]
    assert iou(flood, flood) == 1.0
    assert iou(flood, np.zeros_like(flood)) == 0.0


# ──────────────── preprocess.py (FR-2: деривативы, тайлинг, split) ────────────────


def test_slope_on_constant_plane() -> None:
    from floodrisk.data.preprocess import slope_aspect_deg

    res = 30.0
    cols = np.arange(20)
    dem = np.tile(cols * res, (20, 1)).astype("float32")  # подъём 30 м/пиксель по X
    slope, aspect = slope_aspect_deg(dem, res)
    # tan(slope)=Δz/res=1 → 45°; проверяем внутренность (без краевых эффектов)
    assert np.allclose(slope[5:15, 5:15], 45.0, atol=0.5)
    assert slope.shape == dem.shape and aspect.shape == dem.shape


def test_twi_higher_in_valley_than_ridge() -> None:
    from floodrisk.data.preprocess import twi

    # V-образная долина: высота растёт от центрального столбца к краям
    cols = np.abs(np.arange(40) - 20).astype("float64")
    dem = np.tile(cols * 5.0, (40, 1))  # ложбина по центру (col 20)
    t = twi(dem, res_m=30.0)
    assert t.shape == dem.shape
    # в днище долины TWI выше, чем на склоне у края
    assert t[20, 20] > t[20, 5]


def test_flow_accumulation_increases_downstream() -> None:
    from floodrisk.data.preprocess import flow_accumulation_d8

    # равномерный склон вниз по строкам → накопление растёт сверху вниз
    dem = np.tile(np.arange(30, 0, -1).reshape(-1, 1).astype("float64"), (1, 10))
    acc = flow_accumulation_d8(dem, res_m=30.0)
    assert acc[25, 5] > acc[2, 5]  # ниже по склону площадь сбора больше


def test_distance_to_water_zero_on_water() -> None:
    from floodrisk.data.preprocess import distance_to_water

    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    dist = distance_to_water(mask, res_m=30.0)
    assert dist[0, 0] == 0.0
    assert dist[0, 1] == 30.0


def test_tile_windows_full_tiles_only() -> None:
    from floodrisk.data.preprocess import tile_windows

    specs = tile_windows(width=800, height=600, tile_px=256, overlap_px=0)
    # cols: 0,256,512 (3) × rows: 0,256 (2) = 6 полных тайлов; остаток отброшен
    assert len(specs) == 6
    assert {s.col_idx for s in specs} == {0, 1, 2}
    assert all(s.row_off + 256 <= 600 and s.col_off + 256 <= 800 for s in specs)


def test_geographic_split_is_disjoint() -> None:
    """FR-2 Done: train/val/test не пересекаются по bbox (полосам easting)."""
    from floodrisk.data.preprocess import (
        assign_geographic_splits,
        splits_are_disjoint,
        tile_windows,
    )

    specs = tile_windows(width=256 * 10, height=512, tile_px=256, overlap_px=0)
    assignment = assign_geographic_splits(specs, train=0.7, val=0.15, test=0.15)
    assert set(assignment.values()) == {"train", "val", "test"}
    assert splits_are_disjoint(specs, assignment)


# ──────────────── fetch.py (FR-1) ────────────────


def test_bbox_int_tiles_covers_pilot() -> None:
    from floodrisk.data.fetch import _bbox_int_tiles

    # bbox Тулуна целиком в одном 1°-тайле DEM (lat 54, lon 100)
    assert _bbox_int_tiles([100.4, 54.4, 100.95, 54.8], step=1) == [(54, 100)]
    # bbox через границу долготы → два тайла
    assert _bbox_int_tiles([99.5, 54.4, 100.5, 54.8], step=1) == [(54, 99), (54, 100)]


def test_load_provenance_absent_returns_empty(tmp_path) -> None:
    from floodrisk.data.fetch import load_provenance

    assert load_provenance(tmp_path / "nope.yaml") == {}


@pytest.mark.requires_network
def test_planetary_computer_s1_search() -> None:
    """Smoke (за сетью): по bbox+окну Тулуна находятся сцены Sentinel-1 RTC."""
    pytest.importorskip("pystac_client")
    pytest.importorskip("planetary_computer")
    import planetary_computer as pc
    import pystac_client

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=[100.40, 54.40, 100.95, 54.80],
        datetime="2019-06-28/2019-07-05",
    )
    assert len(list(search.items())) > 0, "ожидаются сцены S1 RTC на пик паводка"
