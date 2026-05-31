# SRS: Прототип ИНС-оценки подверженности затоплению

> **Software Requirements Specification — финальный документ для разработки.**
> Исходные документы (history of decisions): [archive/discovery.md](archive/discovery.md), [archive/concept.md](archive/concept.md), [archive/prd.md](archive/prd.md).
> Все требования здесь — атомарные, тестируемые. При расхождении с архивными документами приоритет — за SRS.
>
> **Версия:** 1.1
> **Дата:** 2026-05-29
> **Статус:** утверждено, можно начинать разработку.

---

## 1. Введение

### 1.1 Назначение

Прототип — веб-сервис, который по bbox городского района и выбранному метео-сценарию возвращает растровую карту вероятности затопления, построенную обученной U-Net моделью на открытых геоданных. Демонстрируется на защите диссертации.

### 1.2 Объём работы (scope)

SRS покрывает **весь end-to-end**:
- Пайплайн загрузки и подготовки открытых геоданных (DEM, Sentinel-1, ERA5, OSM, WorldCover).
- Обучение двух моделей (U-Net и табличный бейзлайн) с фиксированными метриками на geographic hold-out.
- REST API инференса и объяснимости.
- Веб-интерфейс: одна HTML-страница + HTMX + Alpine.js + Tailwind + Leaflet (без SPA-фреймворка и сборщика).
- CLI для вторичной персоны.
- Воспроизводимая сборка через Docker и Makefile.

### 1.3 Что НЕ входит (anti-goals)

Перечислены явно, чтобы исполнитель не расширял скоуп (полный список — [archive/prd.md §3.2](archive/prd.md)):

- ❌ Оперативное оповещение, real-time API, push-уведомления.
- ❌ Прогноз метеоусловий (сценарии — фиксированные пресеты).
- ❌ Гидродинамика во времени (распространение волны паводка).
- ❌ Оценка ущерба в денежном выражении.
- ❌ Подмена государственной экспертизы.
- ❌ Мобильное приложение, оффлайн-режим, PWA, мультитенантность, аутентификация, RBAC, биллинг.
- ❌ Прод-деплой, нагрузочное тестирование, SLA-мониторинг.
- ❌ Дообучение под новый регион через UI.
- ❌ Ансамбли моделей, калибровка неопределённости на пиксель.
- ❌ Закрытые ведомственные данные, платные коммерческие спутники, real-time API сторонних сервисов.
- ❌ React/Vue/любой SPA-фреймворк, отдельный сервер для фронта, сборщик JS-бандла, npm/node_modules.
- ❌ PostgreSQL (включая PostGIS), отдельный сервер БД. Только SQLite.

### 1.4 Терминология

| Термин | Значение |
|---|---|
| **Пилотная территория** | Городской район или агломерация площадью порядка нескольких тысяч км², на которой собирается датасет и обучается модель. Параметризуется через bbox. |
| **Тайл** | Квадратный фрагмент растра фиксированного размера (по умолчанию 256×256 пикселей при разрешении 30 м). |
| **Сценарий** | Преднастроенный набор метео-параметров (`p1y`, `p10y`, `p100y`), который подаётся на вход модели как дополнительные каналы. |
| **Geographic hold-out** | Разделение train/val/test по непересекающимся bbox, а не случайно по пикселям. |
| **Run** | Один запуск инференса: bbox + scenario_id + model_version → растр предсказания + агрегаты. |

---

## 2. Архитектура

### 2.1 Слои системы

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (single HTML + HTMX + Alpine.js + Tailwind + Leaflet)   │
│  все JS-зависимости через CDN, без сборщика                       │
│  - форма выбора сценария / bbox (HTMX → серверный рендер)         │
│  - Leaflet с растровым overlay + панель агрегатов                 │
│  - Alpine.js для реактивных микро-состояний (режим, спиннеры)     │
└──────────────────────────┬───────────────────────────────────────┘
                           │  HTTP (HTMX form posts, JSON API)
┌──────────────────────────▼───────────────────────────────────────┐
│  FastAPI + Uvicorn  (app extras: только web + inference deps)     │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐  │
│  │ HTML routes      │  │ JSON API routes                       │  │
│  │ /, /ui/predict   │  │ /api/predict, /api/explain,           │  │
│  │ /ui/explain      │  │ /api/scenarios, /api/runs/{id}/export │  │
│  └──────────────────┘  └──────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ Inference service: резидентная U-Net + baseline в памяти   │   │
│  └───────────────────────────────────────────────────────────┘   │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐  │
│  │ SQLModel ORM     │  │ Geo-IO: rasterio, geopandas, shapely  │  │
│  └────────┬─────────┘  └──────────────────────────────────────┘  │
└───────────┼──────────────────────────────────────────────────────┘
            │
   ┌────────▼────────┐    ┌──────────────────────────────────────┐
   │  SQLite         │    │  Файловое хранилище                  │
   │  (только        │    │  data/, models/, runs/, exports/     │
   │   метаданные,   │    │  (GeoTIFF, PNG-overlay, GeoJSON,     │
   │   ~5 таблиц)    │    │   PDF, .ckpt, .pkl, manifest.yaml)   │
   └─────────────────┘    └──────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────┐
   │  Off-process: ML training pipeline (extras: [ml])        │
   │  PyTorch Lightning + segmentation-models-pytorch         │
   │  MLflow tracking → mlruns/                               │
   │  Запускается через `make train`. В отдельном окружении   │
   │  (локальная GPU / Colab / RunPod — см. §14.5).           │
   └──────────────────────────────────────────────────────────┘
```

### 2.2 Принципы

- **Воспроизводимость > удобства.** Любой результат пересобирается из исходников и зафиксированных данных. Сиды, версии, контрольные суммы — в манифестах.
- **Минимальный, но честный end-to-end.** Тонкий слой через всю систему — приоритет над глубокой проработкой одной части.
- **Без SPA, без сборщика.** Одна HTML-страница, серверный рендеринг фрагментов через HTMX, Alpine.js для микро-реактивности, Leaflet — обычная JS-библиотека через CDN. Никаких npm/node_modules/webpack/vite.
- **БД хранит только метаданные.** Геоданные (растры, векторы) живут как файлы; БД содержит ссылки и метки запусков.
- **Разделение зависимостей через extras.** Тяжёлые ML-зависимости (PyTorch Lightning, MLflow, captum) ставятся опционально для тренировки; web/inference контейнер их не тянет.
- **Open source стек.** Никаких платных или закрытых компонентов.

---

## 3. Технологический стек

### 3.1 Backend (core, обязательные)

| Компонент | Версия | Назначение |
|---|---|---|
| Python | 3.11 | Основной язык |
| FastAPI | ≥ 0.110 | HTTP-сервер, OpenAPI |
| Uvicorn | ≥ 0.29 | ASGI-сервер |
| SQLModel | ≥ 0.0.16 | ORM + pydantic в одной модели (поверх SQLAlchemy 2.x) |
| pydantic | 2.x | Валидация запросов/ответов |
| pydantic-settings | ≥ 2.2 | Загрузка конфига из env |
| Jinja2 | ≥ 3.1 | Серверные HTML-шаблоны (для HTMX-фрагментов) |
| PyYAML | ≥ 6.0 | Конфиги сценариев, моделей, манифест |

> **Без Alembic.** Для прототипа с 5 таблицами и быстрой итерацией схемы используется `SQLModel.metadata.create_all()` + явный `make db-reset`. Метаданные регенерируются из конфигов (scenarios) и из файлов в `runs/` (runs/exports). Потеря БД не катастрофична. Если схема стабилизируется к моменту защиты — Alembic добавляется отдельной миграцией (это решение OQ-нового — см. §16).

### 3.2 Геоданные и инференс (core, обязательные)

> Эти зависимости нужны для веб-сервиса и инференса. Ставятся всегда.

| Компонент | Версия | Назначение |
|---|---|---|
| torch | ≥ 2.2 (CPU-wheel в app-образе) | Загрузка чекпоинта U-Net и инференс |
| scikit-learn | ≥ 1.4 | Загрузка бейзлайна, инференс |
| rasterio | ≥ 1.3 | Чтение/запись GeoTIFF |
| geopandas | ≥ 0.14 | Векторные слои, GeoJSON-экспорт |
| shapely | ≥ 2.0 | Геометрия, валидация bbox |
| numpy, scipy | актуальные | Численные операции |
| matplotlib | ≥ 3.8 | Колормапы для PNG-overlay |
| Pillow | ≥ 10.0 | PNG для overlay карты |
| ReportLab | ≥ 4.0 | Генерация PDF-отчёта |
| captum | ≥ 0.7 | Integrated Gradients для `/api/explain` в рантайме |

### 3.2.1 ML training (extras: `[ml]`, опциональные)

> Ставятся только в окружении для тренировки: `uv pip install -e ".[ml]"`. App/inference Docker-образ их не содержит — это снижает размер контейнера на ~2-3 ГБ.

| Компонент | Версия | Назначение |
|---|---|---|
| pytorch-lightning | ≥ 2.2 | Тренировочный цикл, чекпоинты |
| segmentation-models-pytorch | ≥ 0.3 | Готовая U-Net |
| albumentations | ≥ 1.4 | Аугментации |
| mlflow | ≥ 2.10 | Трекинг экспериментов |
| xarray, rioxarray | ≥ 2024.x | Многоканальные растры, ERA5 (нужны на этапе data prep) |
| cdsapi | ≥ 0.7 | Загрузка ERA5/ERA5-Land с Copernicus CDS |
| torch (CUDA wheel) | ≥ 2.2 | GPU-вариант для тренировки (ставится отдельно из CUDA-индекса) |

### 3.3 Frontend

| Компонент | Канал доставки | Назначение |
|---|---|---|
| Tailwind CSS | CDN (`<script src="https://cdn.tailwindcss.com">`) | Стили |
| HTMX | CDN (`htmx.org@1.9.x`) | Серверный рендеринг HTML-фрагментов, OOB swaps |
| Alpine.js | CDN (`alpinejs@3.x`, defer) | Микро-реактивность: режимы UI (просмотр / объяснение), спиннеры, локальное состояние панелей. Дешёвая декларативная альтернатива vanilla JS. |
| Leaflet | CDN (`leaflet@1.9.x` JS + CSS) | Интерактивная карта с растровым overlay (`L.imageOverlay`) |
| OSM tiles | tile.openstreetmap.org | Базовая подложка |

**Никаких node_modules, package.json, сборщиков.** Всё через CDN, отдаётся как статика из FastAPI.

> **Разделение ответственности:** HTMX рулит серверным обменом (формы → HTML-фрагменты). Alpine.js рулит локальной реактивностью (показать/скрыть, переключить режим). Vanilla JS в `static/app.js` отвечает только за императивный мост к Leaflet (он управляется не декларативно). Такое разделение удерживает каждую часть в ~50-100 строк.

### 3.4 Хранилища

| Хранилище | Что хранит |
|---|---|
| SQLite (`app.db` в корне проекта) | Метаданные: runs, scenarios, model_versions, exports, explanations. Только реляционные поля, без геометрии. Единственная СУБД. |
| Файловая система | `data/raw/`, `data/processed/`, `models/`, `runs/`, `exports/`, `mlruns/` (GeoTIFF, PNG, GeoJSON, PDF, .ckpt, .pkl, manifest.yaml) |

> **Почему только SQLite, без Postgres.** Один пользователь, демо на ноутбуке, никаких параллельных пишущих, никакой PostGIS-специфики. SQLite через WAL-режим даёт более чем достаточно для прототипа. Profile с Postgres в docker-compose был бы чистым maintenance overhead без выигрыша. Если в дальнейшем потребуется (LATER) — миграция на Postgres через SQLModel/SQLAlchemy = смена URL, без изменений в коде.

### 3.5 Инструменты разработки

| Инструмент | Назначение |
|---|---|
| **uv** (≥ 0.4) | Менеджер зависимостей и venv (от Astral, авторов ruff). Заменяет pip + pip-tools + venv. Используется для `uv venv`, `uv pip compile`, `uv pip sync`. |
| pytest, pytest-asyncio | Тесты |
| httpx | Async HTTP-клиент для тестов API |
| pytest-cov | Покрытие |
| ruff | Линт + автоформат (`ruff check`, `ruff format`) |
| pre-commit | Hooks: ruff check + ruff format при коммите |
| make (GNU Make) | Универсальный entry point |
| Docker | Образ для приложения; для геоданных и тренировки на Windows — основной путь (см. §14.5) |

> **Почему uv:** под критерий M-1 («≤ 1 день на чистой машине») установка зависимостей через uv в 10-100× быстрее pip. Lockfile нативный, поддержка `[project.optional-dependencies]` из PEP 621.

---

## 4. Структура проекта

```
.
├── Makefile
├── README.md
├── pyproject.toml             # PEP 621: метаданные + deps + extras [ml] + ruff + pytest config
├── requirements.lock          # зафиксированные версии core (uv pip compile)
├── requirements-ml.lock       # зафиксированные версии core + [ml] (для training env)
├── Dockerfile                 # app/inference образ (без ML extras)
├── .dockerignore
├── .env.example               # CDS_API_KEY, COPERNICUS_TOKEN, LOG_LEVEL
├── .pre-commit-config.yaml
├── docs/
│   ├── srs.md                 # этот документ
│   ├── api.md                 # автогенерируемое описание API (опционально)
│   └── archive/               # discovery.md, concept.md, prd.md
├── configs/
│   ├── unet_v1.yaml
│   ├── baseline_v1.yaml
│   └── scenarios.yaml         # параметры p1y, p10y, p100y
├── floodrisk/
│   ├── __init__.py
│   ├── app.py                 # FastAPI app factory
│   ├── settings.py            # pydantic-settings (env vars)
│   ├── db/
│   │   ├── models.py          # SQLModel (одновременно ORM и pydantic)
│   │   ├── session.py         # engine, get_session(), create_db_and_tables()
│   │   └── repositories.py
│   ├── api/
│   │   ├── routes_html.py     # HTMX-фрагменты, Jinja2
│   │   ├── routes_api.py      # JSON API
│   │   └── schemas.py         # pydantic
│   ├── inference/
│   │   ├── service.py         # резидентная модель, predict(), explain()
│   │   ├── raster_to_png.py   # GeoTIFF + colormap → PNG для Leaflet
│   │   └── aggregates.py      # доли площади по порогам
│   ├── data/
│   │   ├── fetch.py           # загрузка сырых источников
│   │   ├── preprocess.py      # производные, проекция, нарезка тайлов
│   │   ├── labels_s1.py       # разметка Sentinel-1
│   │   ├── manifest.py        # генерация и верификация manifest.yaml
│   │   └── permanent_water.py # маска OSM + JRC GSW
│   ├── ml/
│   │   ├── datamodule.py      # LightningDataModule, tile loader
│   │   ├── unet.py            # LightningModule
│   │   ├── baseline.py        # sklearn pipeline
│   │   ├── train.py           # entry для `make train`
│   │   ├── evaluate.py        # отчёт сравнения U-Net vs baseline
│   │   ├── explain.py         # integrated gradients / feature importance
│   │   └── losses.py
│   ├── cli/
│   │   └── __main__.py        # `python -m floodrisk` CLI
│   └── templates/             # Jinja2
│       ├── base.html
│       ├── index.html
│       └── fragments/
│           ├── map_layer.html
│           ├── aggregates.html
│           └── explanation_panel.html
├── static/
│   ├── app.js                 # ~50-100 строк: императивный мост HTMX → Leaflet
│   └── app.css                # минимум, основное — через Tailwind CDN
├── data/
│   ├── raw/                   # .gitignore
│   ├── processed/
│   │   └── v1/
│   │       ├── features/
│   │       ├── tiles/
│   │       ├── labels/
│   │       └── index.parquet
│   └── manifest.yaml          # коммитится
├── models/
│   ├── unet/<run_id>/best.ckpt
│   └── baseline/<run_id>/model.pkl
├── runs/                      # инференс: <run_id>/prediction.tif, .png, aggregates.json
├── exports/                   # ZIP-архивы для скачивания
├── mlruns/                    # MLflow tracking (.gitignore по умолчанию)
├── reports/
│   └── comparison_v1.md       # таблица U-Net vs baseline
├── notebooks/                 # вторичная персона: исследовательские
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_inference.py
    ├── test_data_pipeline.py
    ├── test_ml_smoke.py
    └── fixtures/
        └── mini_dataset/      # маленький детерминированный датасет для CI
```

---

## 5. Модель данных (БД)

Все таблицы — реляционные, без геометрии. Геоданные хранятся как файлы; БД содержит только пути и метки.

### 5.1 Сущности

#### `model_versions`
| Поле | Тип | Описание |
|---|---|---|
| `id` | INTEGER PK | |
| `name` | TEXT NOT NULL | например `unet`, `baseline` |
| `version` | TEXT NOT NULL | например `v1` |
| `checkpoint_path` | TEXT NOT NULL | путь к `.ckpt`/`.pkl` |
| `config_path` | TEXT NOT NULL | путь к YAML конфигу |
| `metrics_json` | TEXT | сериализованные финальные метрики |
| `dataset_version` | TEXT NOT NULL | например `v1` |
| `created_at` | DATETIME | |

UNIQUE (`name`, `version`).

#### `scenarios`
| Поле | Тип | Описание |
|---|---|---|
| `id` | INTEGER PK | |
| `scenario_id` | TEXT UNIQUE NOT NULL | `p1y`, `p10y`, `p100y` |
| `name` | TEXT NOT NULL | человекочитаемое название |
| `description` | TEXT | |
| `params_json` | TEXT NOT NULL | параметры из `configs/scenarios.yaml`, скопированы при сидинге |

Сидинг (`make seed`) читает `configs/scenarios.yaml` и наполняет таблицу.

#### `runs`
| Поле | Тип | Описание |
|---|---|---|
| `id` | TEXT PK (UUID) | run_id |
| `bbox_w`, `bbox_s`, `bbox_e`, `bbox_n` | REAL NOT NULL | bbox в EPSG:4326 |
| `scenario_id` | TEXT FK → scenarios.scenario_id | |
| `model_version_id` | INTEGER FK → model_versions.id | |
| `prediction_tif_path` | TEXT | путь к GeoTIFF |
| `prediction_png_path` | TEXT | путь к PNG для Leaflet overlay |
| `aggregates_json` | TEXT | доли площади, средняя вероятность |
| `latency_ms` | INTEGER | |
| `status` | TEXT | `ok` / `error` |
| `error_message` | TEXT NULL | |
| `created_at` | DATETIME | |

#### `exports`
| Поле | Тип | Описание |
|---|---|---|
| `id` | INTEGER PK | |
| `run_id` | TEXT FK → runs.id | |
| `zip_path` | TEXT NOT NULL | путь к скачиваемому ZIP |
| `created_at` | DATETIME | |

#### `explanations`
| Поле | Тип | Описание |
|---|---|---|
| `id` | INTEGER PK | |
| `run_id` | TEXT FK → runs.id | связан с инференсом, на котором кликнули |
| `lat`, `lon` | REAL NOT NULL | |
| `importance_json` | TEXT NOT NULL | ранжирование признаков |
| `attribution_tif_paths_json` | TEXT NOT NULL | массив путей к растрам важности по каждому из топ-N признаков |
| `created_at` | DATETIME | |

### 5.2 Управление схемой

- Без Alembic. Используется `SQLModel.metadata.create_all(engine)` при старте приложения (idempotent).
- `make db-reset` — удаляет `app.db` и пересоздаёт схему + запускает сидинг.
- `make seed` — наполняет `scenarios` из `configs/scenarios.yaml` и регистрирует `model_versions` из `models/`.
- При изменении схемы в `db/models.py` исполнитель обязан запустить `make db-reset`. Данные в БД некритичны: scenarios регенерируются из конфига, runs/exports — из файлов на диске (`runs/<id>/` остаются, но строки в БД пропадают; в README — указание не полагаться на БД-историю запусков).
- **Если** к фазе подготовки к защите схема стабилизируется и не хочется терять историю запусков — добавляется Alembic (одна initial-миграция, дальше autogenerate). Это решение **OQ-14** (см. §16), не блокирует разработку.

### 5.3 Выбор СУБД

- SQLite, файл `app.db` в корне проекта. URL: `sqlite:///./app.db`.
- Включён WAL-режим (через `PRAGMA journal_mode=WAL`) для безопасности при параллельных read во время инференса.
- Дефолт зашит, переопределяется через `DATABASE_URL` (для тестов используется `sqlite:///:memory:`).

---

## 6. Геоданные и файловое хранилище

### 6.1 Сырые данные (`data/raw/`)

Скачиваются скриптом `floodrisk.data.fetch` по bbox + временному окну. См. §8.

| Источник | Формат | Путь |
|---|---|---|
| Copernicus DEM GLO-30 | GeoTIFF | `data/raw/dem/` |
| ESA WorldCover 2021 | GeoTIFF | `data/raw/worldcover/` |
| OSM (Geofabrik) | PBF / GPKG | `data/raw/osm/` |
| GHSL | GeoTIFF | `data/raw/ghsl/` |
| SoilGrids | GeoTIFF | `data/raw/soilgrids/` |
| ERA5 / ERA5-Land | NetCDF | `data/raw/era5/` |
| Sentinel-1 GRD | GeoTIFF | `data/raw/sentinel1/<event_id>/` |
| JRC GSW | GeoTIFF | `data/raw/jrc_gsw/` |

`data/raw/` — в `.gitignore`. Воспроизводимость — через манифест + sha256 контрольные суммы.

### 6.2 Обработанные данные (`data/processed/v1/`)

| Папка/файл | Содержание |
|---|---|
| `features/` | Производные растры: уклон, экспозиция, кривизна, TWI, расстояние до водотока. Единая проекция (см. §6.4), разрешение 30 м. |
| `tiles/<tile_id>.tif` | Многоканальные GeoTIFF, размер 256×256 (по умолчанию, переопределимо в конфиге). |
| `labels/<event_id>/<tile_id>.tif` | Бинарные маски затоплено/нет, по тайлам. |
| `index.parquet` | Таблица: `tile_id`, `bbox_wgs84`, `event_ids`, `split` (`train`/`val`/`test`). |

### 6.3 Манифест датасета (`data/manifest.yaml`)

Коммитится в репозиторий. Структура:

```yaml
dataset_version: v1
created_at: 2026-MM-DD
pilot_bbox_wgs84: [w, s, e, n]
target_crs: EPSG:32637            # см. §6.4
tile_size_px: 256
tile_resolution_m: 30
sources:
  dem:
    url: https://...
    fetched_at: 2026-MM-DD
    sha256: ...
  worldcover: { ... }
  # ...
splits:
  seed: 42
  method: geographic_bbox
events:
  - event_id: 2019-flood-001
    date_window: [2019-MM-DD, 2019-MM-DD]
    sentinel1_scenes: [...]
tile_checksums:
  - tile_id: t_0001
    sha256: ...
  # ...
```

`make verify-data` сверяет фактические файлы с манифестом, exit code 0 при совпадении.

### 6.4 Проекция и тайлинг

- **Целевой CRS:** UTM-зона пилотной территории (например, EPSG:32637 для центральной России). Конкретная зона выбирается после фиксации пилотной территории (OQ-1, см. §16).
- **Разрешение:** 30 м/пиксель.
- **Размер тайла:** 256×256 пикселей по умолчанию. Параметризуется в `configs/data.yaml`.
- **Перекрытие тайлов:** 0 для трейна; для инференса по произвольному bbox используется sliding window с перекрытием 32 пикселя и weighted blending на краях.

### 6.5 Артефакты инференса (`runs/<run_id>/`)

| Файл | Описание |
|---|---|
| `prediction.tif` | Растр вероятностей в целевом CRS |
| `prediction.png` | Тот же растр, перепроецирован в EPSG:4326, цветовая шкала Viridis (или подобная) с прозрачностью, для Leaflet `imageOverlay` |
| `aggregates.json` | Доли площади с p>0.5, p>0.8, средняя вероятность по bbox |
| `metadata.json` | model_version, dataset_version, scenario_id, bbox, run_timestamp, latency_ms |
| `attribution/<feature>.tif` (опционально) | Карты важности признаков для `/explain` |

### 6.6 Экспорт (`exports/<run_id>.zip`)

ZIP содержит:
- `prediction.tif`
- `aggregates.geojson` (полигон bbox + properties с агрегатами; CRS = EPSG:4326)
- `report.pdf` (ReportLab; разделы: заголовок, карта-скриншот, легенда, метаданные)

---

## 7. ML-пайплайн

### 7.1 Постановка задачи

- **Тип:** семантическая сегментация / попиксельная вероятность затопления.
- **Вход:** многоканальный растр признаков (DEM, производные рельефа, землепокрытие, расстояния до водотоков, осадки сценария, индексы влажности), тайл 256×256.
- **Выход:** растр той же геометрии, в каждом пикселе — вероятность ∈ [0, 1].

### 7.2 Модели

#### 7.2.1 U-Net (основная)
- `segmentation_models_pytorch.Unet` с энкодером `resnet34` (опционально `efficientnet-b0`).
- Входные каналы: число каналов признаков (зависит от конфига; целевое — 8–12).
- Loss: `BCE + Dice` (компонент Focal — опционально через конфиг).
- Оптимизатор: AdamW, lr=1e-3 с CosineAnnealing.
- Аугментации (albumentations): повороты ×k·90°, horizontal/vertical flips, шум на метео-каналах.
- Чекпоинтинг по best val IoU.

#### 7.2.2 Табличный бейзлайн
- `sklearn.ensemble.RandomForestClassifier` (либо `LogisticRegression` для скорости).
- Обучение попиксельно: каждый пиксель — отдельный сэмпл с теми же признаками.
- На больших датасетах — подвыборка пикселей (стратифицированная по классу).

### 7.3 Разделение датасета

- **По географическому принципу:** train/val/test — по непересекающимся bbox.
- **Сид:** фиксирован в `manifest.yaml` (`splits.seed`).
- Доли: 70/15/15 (переопределимо в конфиге).

### 7.4 Метрики

- **Основные:** IoU, F1, ROC-AUC, PR-AUC.
- **Дополнительные:** Brier score (калибровка), precision, recall.
- **Пространственные:** разбор ошибок по типам поверхности (поймы, склоны, городская застройка) — в `reports/comparison_v1.md` отдельным разделом.

### 7.5 Сравнение U-Net vs baseline

- `make eval` загружает финальные чекпоинты обеих моделей, прогоняет на test-split, считает метрики с bootstrap-CI 95% (1000 ресэмплов).
- Отчёт `reports/comparison_v1.md`:
  - таблица: метрика | U-Net | baseline | CI U-Net | CI baseline | непересекаются (yes/no).
  - вывод: pass/fail по критерию M-2 (см. §15).

### 7.6 MLflow

- Трекинг-стор: `mlruns/` локально (file-based backend).
- Логируется на эпоху: train/val loss, IoU, F1, ROC-AUC, PR-AUC.
- Логируется по завершении: финальные метрики на test, ссылка на чекпоинт, копия конфига.
- UI: `mlflow ui` (опционально, не часть Compose).

### 7.7 Объяснимость

- **Метод:** Integrated Gradients для U-Net через `captum`. Для бейзлайна — feature_importances_ из sklearn.
- **Окно:** для `/explain` — тайл 256×256 вокруг точки клика (обрезается, если выходит за зону покрытия).
- **Выход:** N растров (по одному на топ-N признаков, N=5 по умолчанию) + JSON-ранжирование.
- **Ограничения:** документируются в README и в UI-подсказке.

### 7.8 Конфиги (YAML)

Пример `configs/unet_v1.yaml`:

```yaml
model:
  arch: unet
  encoder: resnet34
  in_channels: 10
  out_channels: 1
data:
  tile_dir: data/processed/v1/tiles
  index: data/processed/v1/index.parquet
  batch_size: 16
  augmentations: ['rot90', 'hflip', 'vflip', 'meteo_noise']
training:
  max_epochs: 50
  lr: 1.0e-3
  loss: bce_dice
  seed: 42
  precision: 16-mixed
checkpoint:
  monitor: val_iou
  mode: max
  save_top_k: 1
```

---

## 8. REST API контракт

OpenAPI генерируется автоматически. Swagger UI — на `/docs`.

### 8.1 JSON API

#### `POST /api/predict`

**Запрос:**
```json
{
  "bbox": [37.5, 55.6, 37.7, 55.8],
  "scenario_id": "p100y",
  "model_version": "unet-v1"
}
```

**Ответ 200:**
```json
{
  "run_id": "9f8e7c…",
  "prediction_png_url": "/runs/9f8e7c…/prediction.png",
  "prediction_tif_url": "/runs/9f8e7c…/prediction.tif",
  "bounds_wgs84": [55.6, 37.5, 55.8, 37.7],
  "aggregates": {
    "area_km2": 218.4,
    "fraction_p_gt_0_5": 0.073,
    "fraction_p_gt_0_8": 0.021,
    "mean_p": 0.142
  },
  "metadata": {
    "model_version": "unet-v1",
    "dataset_version": "v1",
    "scenario_id": "p100y",
    "bbox": [37.5, 55.6, 37.7, 55.8],
    "run_timestamp": "2026-05-29T12:34:56Z",
    "latency_ms": 1834
  }
}
```

**Ошибки:**
- `422` — bbox вне зоны покрытия: `{"error": "bbox_out_of_coverage", "coverage_wgs84": [...]}`
- `404` — неизвестный `scenario_id` или `model_version`: `{"error": "unknown_scenario", "available": [...]}`
- `400` — некорректный bbox: `{"error": "invalid_bbox", "detail": "..."}`

#### `POST /api/explain`

**Запрос:**
```json
{
  "run_id": "9f8e7c…",
  "lat": 55.71,
  "lon": 37.62
}
```

**Ответ 200:**
```json
{
  "explanation_id": 42,
  "ranking": [
    { "feature": "elevation", "importance": 0.31 },
    { "feature": "twi", "importance": 0.22 },
    { "feature": "distance_to_water", "importance": 0.18 },
    { "feature": "precipitation_p100y", "importance": 0.15 },
    { "feature": "slope", "importance": 0.09 }
  ],
  "attribution_layers": [
    { "feature": "elevation", "png_url": "/runs/9f8e7c…/attribution/elevation.png", "bounds_wgs84": [...] }
  ]
}
```

#### `GET /api/scenarios`
Список преднастроенных сценариев из БД.

#### `GET /api/model-versions`
Список зарегистрированных моделей.

#### `POST /api/runs/{run_id}/export`
Готовит ZIP (TIF + GeoJSON + PDF), возвращает `{"export_url": "/exports/<id>.zip"}`.

#### `GET /api/runs/{run_id}`
Метаданные конкретного запуска.

### 8.2 HTML routes (HTMX)

| Route | Метод | Возвращает |
|---|---|---|
| `/` | GET | Полная HTML-страница (Jinja base + index) |
| `/ui/predict` | POST | HTML-фрагмент `fragments/map_layer.html` + `fragments/aggregates.html`. Содержит `<div hx-swap-oob="true">` для агрегатов и data-атрибуты (`data-png-url`, `data-bounds`) для триггера JS-обновления Leaflet. |
| `/ui/explain` | POST | HTML-фрагмент `fragments/explanation_panel.html`. |
| `/ui/export` | POST | HTML-фрагмент с ссылкой на скачивание + HX-Trigger header `download-ready`. |

HTML-роуты внутри используют тот же `inference.service`, что и JSON API, — единая логика.

### 8.3 Статика

| Префикс | Содержание |
|---|---|
| `/static/` | `app.js`, `app.css` |
| `/runs/` | Артефакты запусков (PNG, TIF) — отдаются как файлы |
| `/exports/` | ZIP-архивы экспортов |

---

## 9. Веб-интерфейс

### 9.1 Структура страницы

Одна страница `index.html`, layout — Tailwind:

```
┌──────────────────────────────────────────────────────────┐
│  Header: название проекта, версия модели                  │
├──────────────────┬───────────────────────────────────────┤
│                  │                                       │
│  Sidebar (256px) │  Leaflet map (растягивается)          │
│  - сценарий      │  + растровый overlay                   │
│    (radio)       │  + базовая OSM подложка                │
│  - модель        │  + клик → /ui/explain                  │
│  - bbox preset   │                                       │
│  - [Пересчитать] │                                       │
│  - агрегаты      │                                       │
│  - [Экспорт]     │                                       │
│                  │                                       │
├──────────────────┴───────────────────────────────────────┤
│  Footer: легенда шкалы вероятности (5 градаций)          │
└──────────────────────────────────────────────────────────┘
```

При активации режима «Объяснение» снизу выезжает панель с топ-5 признаками и переключателями слоёв важности.

### 9.2 Поведение HTMX

- Форма выбора сценария — `hx-post="/ui/predict"`, `hx-trigger="change"` на radio. Целевой элемент — невидимый `<div id="prediction-data">` с data-атрибутами. Агрегаты обновляются out-of-band через `hx-swap-oob`.
- Кнопка «Экспорт» — `hx-post="/ui/export"`, ответ заменяет блок с ссылкой на скачивание, header `HX-Trigger: download-ready` запускает `window.location = ...`.
- Клик по карте в режиме объяснения — JS перехватывает событие Leaflet, формирует form-data и шлёт через `htmx.ajax('POST', '/ui/explain', ...)`.

### 9.3 JS (`static/app.js`)

Минимум, ~50 строк:
1. Инициализация Leaflet-карты с базовой подложкой OSM, centered на bbox пилотной территории.
2. Хранение ссылки на текущий `L.imageOverlay` (`window.currentOverlay`).
3. Слушатель `htmx:afterSwap` на `#prediction-data` — читает data-атрибуты, удаляет старый overlay, добавляет новый.
4. Слушатель `htmx:afterSwap` на `#explanation-data` — добавляет/убирает слои важности.
5. Обработчик `map.on('click')` — если активен режим «Объяснение», шлёт запрос через `htmx.ajax`.
6. Слушатель `HX-Trigger: download-ready` — инициирует скачивание.

### 9.4 Доступность

- Не часть критериев приёмки MVP, но: основные действия доступны с клавиатуры (фокус, Enter на radio/button); цветовая шкала Viridis — colorblind-safe.

### 9.5 Браузеры

Последние стабильные Chrome и Firefox (десктоп). Мобильный режим — не поддерживается.

---

## 10. CLI

### 10.1 Команды

```
python -m floodrisk data fetch --bbox W,S,E,N --events EVT1,EVT2 --out data/raw/
python -m floodrisk data preprocess --in data/raw --out data/processed/v1
python -m floodrisk data label --in data/raw/sentinel1 --out data/processed/v1/labels
python -m floodrisk data manifest --dir data/processed/v1
python -m floodrisk data verify --manifest data/manifest.yaml

python -m floodrisk train --config configs/unet_v1.yaml
python -m floodrisk train --config configs/baseline_v1.yaml
python -m floodrisk eval --models unet-v1,baseline-v1 --out reports/comparison_v1.md

python -m floodrisk infer --bbox W,S,E,N --scenario p100y --model unet-v1 --out runs/<id>
python -m floodrisk explain --run-id <id> --lat L --lon LN --out runs/<id>/attribution
```

### 10.2 Makefile (entry points)

```makefile
# окружение
make venv           # uv venv .venv (создать venv)
make install        # uv pip sync requirements.lock (только core)
make install-ml     # uv pip sync requirements-ml.lock (для тренировки)
make lock           # uv pip compile pyproject.toml -> requirements.lock + requirements-ml.lock

# данные и ML
make data           # fetch + preprocess + label + manifest для пилотной территории
make train          # обучение U-Net + baseline по дефолтным конфигам (требует [ml])
make eval           # сравнение, отчёт (требует [ml])
make verify-data    # проверка манифеста

# БД
make db-reset       # удалить app.db, пересоздать схему, выполнить seed
make seed           # загрузка scenarios + model_versions из конфигов в БД

# сервис
make run            # uvicorn floodrisk.app:app --reload --port 8000

# docker
make docker-build   # docker build -t floodrisk:latest .
make docker-run     # docker run -p 8000:8000 -v $(PWD)/runs:/app/runs ... floodrisk:latest

# качество
make test           # pytest
make lint           # ruff check . && ruff format --check .
make fmt            # ruff format .
```

---

## 11. Функциональные требования (mapping)

> Каждое FR — атомарное, тестируемое, с признаком готовности (Done). Нумерация наследуется из [archive/prd.md §6](archive/prd.md), уточнения учитывают новый стек.

### 11.1 Data layer

**FR-1. Скрипт загрузки сырых данных.** `python -m floodrisk data fetch --bbox=...` скачивает все источники из §6.1. Идемпотентен, токены берутся из `.env`.
- *Done:* на чистом контейнере отрабатывает без ручных шагов; integration-тест на мини-bbox (10×10 км) проходит.

**FR-2. Препроцессинг и нарезка тайлов.** Скрипт строит производные признаки, приводит к единому CRS/разрешению, нарезает на тайлы.
- *Done:* `data/processed/v1/tiles/` непустой; `index.parquet` создан; unit-тест на идентичность shape/CRS у случайной выборки тайлов; train/val/test не пересекаются по bbox.

**FR-3. Разметка Sentinel-1.** Скрипт детектирует затопленные пиксели, вычитает маску постоянной воды (OSM + JRC GSW).
- *Done:* для каждого события — бинарные маски в `labels/<event_id>/`; метод и пороги — в манифесте; на эталонном событии — площадь пересечения с ручной разметкой ≥ порога OQ-2.

**FR-4. Манифест и контрольные суммы.** `data/manifest.yaml` создаётся автоматически; `make verify-data` даёт exit 0 при совпадении.
- *Done:* unit-тест меняет один тайл и убеждается, что `verify` падает с exit≠0.

### 11.2 ML layer

**FR-5. Обучение U-Net.** `make train MODEL=unet` логирует метрики в MLflow, сохраняет чекпоинт.
- *Done:* smoke-тест: 1 эпоха на mini_dataset → метрики записаны, `best.ckpt` сохранён.

**FR-6. Обучение бейзлайна.** `make train MODEL=baseline` обучает sklearn-модель на тех же splits.
- *Done:* аналогично FR-5.

**FR-7. Сравнение моделей.** `make eval` строит `reports/comparison_v1.md` с bootstrap-CI 95%.
- *Done:* отчёт содержит таблицу + pass/fail по M-2.

**FR-8. CLI-инференс.** `python -m floodrisk infer --bbox=... --scenario=p100y --model=unet-v1 --out=...` — без поднятия API.
- *Done:* отрабатывает на чистом контейнере; на выходе `prediction.tif` + `aggregates.json`.

### 11.3 Inference service

**FR-9. `POST /api/predict`.** См. §8.1.
- *Done:* Swagger доступен; запрос на фиксированном bbox даёт 200 + валидный GeoTIFF; smoke-test в CI; PNG для Leaflet перепроецирован в EPSG:4326.

**FR-10. `POST /api/explain`.** См. §8.1.
- *Done:* запрос отдаёт ранжирование признаков + массив PNG-attributions; latency — см. NFR-3.

**FR-11. `GET /api/scenarios`.** См. §8.1.
- *Done:* возвращает ≥ 3 сценария из БД; описания приходят из `configs/scenarios.yaml` через сидинг.

**FR-12. Валидация bbox и обработка ошибок.** См. §8.1.
- *Done:* unit-тесты на каждый кейс (out of coverage, unknown scenario, unknown model, invalid bbox).

**FR-13. Резидентная модель в памяти.** Контейнер загружает чекпоинты U-Net и бейзлайна при старте.
- *Done:* замер показывает, что latency от 2-го запроса стабильна (нет «холодного старта»).

### 11.4 Frontend

**FR-14. Leaflet-карта с растровым overlay.** Одна HTML-страница, Leaflet через CDN, базовая подложка OSM, полупрозрачный PNG-overlay.
- *Done:* растр виден поверх базовой карты; зум/пан работают; шкала соответствует легенде.

**FR-15. Селектор сценария.** Radio с 3 сценариями; смена триггерит `/ui/predict`.
- *Done:* переключение между `p1y/p10y/p100y` визуально меняет overlay; во время загрузки — спиннер.

**FR-16. Легенда и агрегаты.** Панель с легендой (≥ 5 градаций) и агрегатами по текущему bbox.
- *Done:* при смене сценария агрегаты обновляются согласованно с растром (через HTMX OOB swap).

**FR-17. Экспорт.** Кнопка отдаёт ZIP (TIF + GeoJSON + PDF) с метаданными.
- *Done:* скачивание работает в Chrome и Firefox; PDF открывается; все метаданные присутствуют.

**FR-18. Режим объяснения.** Клик по карте → `/ui/explain` → панель с топ-5 признаками.
- *Done:* 5 случайных кликов на пилотной территории дают ответ ≤ NFR-3; UI отрисовывает без ошибок в консоли.

### 11.5 Воспроизводимость и упаковка

**FR-19. Docker-образ приложения.** Один Dockerfile (multi-stage), `make docker-build` собирает образ ≤ 1 ГБ, `make docker-run` поднимает контейнер с volume-mounts на `runs/`, `exports/`, `models/`, `app.db`.
- *Done:* на чистой машине `make docker-build && make docker-run` поднимает сервис; фронт по `http://localhost:8000`; smoke-test проходит. Без docker-compose.

**FR-20. README.** Полная инструкция: prerequisites, сборка данных, обучение, запуск демо, воспроизведение метрик, ограничения объяснимости.
- *Done:* внешний человек по README поднимает систему за ≤ 1 день и получает карту.

---

## 12. Нефункциональные требования

| # | Категория | Требование | Способ проверки |
|---|---|---|---|
| NFR-1 | Latency `/api/predict` | p95 ≤ 5 сек на CPU (8 ядер), ≤ 1 сек на GPU, на тайле пилотной территории | 20 последовательных запросов с фиксированным bbox |
| NFR-2 | Latency UI | От смены сценария до перерисовки overlay — p95 ≤ 6 сек на CPU | Ручной замер на пилотной территории |
| NFR-3 | Latency `/api/explain` | p95 ≤ 15 сек на CPU для тайла 256×256 вокруг точки | 10 запросов |
| NFR-4 | Объём данных | Датасет v1 — порядка 1k тайлов 256×256 | `wc -l data/processed/v1/index.parquet` (после загрузки в pandas) |
| NFR-5 | Нагрузка | 1 одновременный пользователь; uvicorn 1 worker | Не тестируем выше |
| NFR-6 | Размер артефактов | Чекпоинт U-Net + бейзлайн в сумме ≤ 500 МБ; репозиторий без Git LFS | `ls -lh models/` |
| NFR-7 | Воспроизводимость метрик | Фиксированные сиды (PyTorch, NumPy, Python); requirements.lock; два прогона с одним сидом — метрики совпадают в пределах задокументированного порога (OQ-8) | Прогон CI |
| NFR-8 | Браузеры | Последние стабильные Chrome и Firefox на десктопе | Ручной прогон UF-4 |
| NFR-9 | Безопасность | API доступен только локально (bind 127.0.0.1 в dev; в compose — порт пробрасывается на хост); секреты — через `.env`, не в коде; `.env` — в `.gitignore` | Code review |
| NFR-10 | Логирование | Backend пишет каждый запрос в stdout: bbox, scenario, model_version, latency_ms, status | `docker compose logs backend` |
| NFR-11 | Покрытие тестами | Минимум 60% строк core-модулей (`inference`, `api`, `data.manifest`); ML — smoke-тесты, не покрытие | `pytest --cov` |
| NFR-12 | Линт | `ruff check .` exit 0; `ruff format --check .` exit 0 | CI |

---

## 13. Тестирование

### 13.1 Уровни

| Уровень | Инструмент | Что тестируется |
|---|---|---|
| Unit | pytest | Чистые функции: агрегаты, валидация bbox, конверсия GeoTIFF→PNG, манифест |
| API | pytest + httpx (async client) | Все эндпоинты, включая ошибки 4xx |
| Integration | pytest + временный SQLite | Полный flow: predict → run в БД → export → файл на диске |
| ML smoke | pytest | 1 эпоха U-Net и 1 fit бейзлайна на `tests/fixtures/mini_dataset/` |
| Data smoke | pytest, опционально | На мини-bbox: fetch (если есть токены) + preprocess + manifest. Помечается `@pytest.mark.requires_network` и пропускается в дефолтном прогоне |

### 13.2 Фикстуры

`tests/fixtures/mini_dataset/` — детерминированный мини-датасет из 8 тайлов (4 train, 2 val, 2 test) с синтетическими признаками и метками. Коммитится в репозиторий (≤ 5 МБ).

### 13.3 CI

- Один workflow: `ruff` → `pytest` (без `requires_network`) → smoke-build Docker.
- Без матриц по версиям — Python 3.11 фикс.
- CI-токены для CDS/Copernicus — не закладываются (см. OQ-12, разрешено в SRS как «не блокирует MVP»).

---

## 14. Воспроизводимость и сборка

### 14.1 Зависимости

- Верхнеуровневые — в `pyproject.toml` (PEP 621), секции `[project.dependencies]` (core) и `[project.optional-dependencies].ml`.
- Закрепление: `uv pip compile pyproject.toml -o requirements.lock` (core), `uv pip compile pyproject.toml --extra ml -o requirements-ml.lock` (для training-окружения).
- Установка: `uv pip sync requirements.lock` (быстро, в 10-100× быстрее pip).
- В Docker-образе ставится только `requirements.lock` (без ML extras) — образ ~800 МБ против ~3+ ГБ.
- GDAL/PROJ-зависимости (rasterio, geopandas) ставятся через manylinux wheels (`rasterio>=1.3`, `geopandas>=0.14` — wheel включает GDAL). Образ — на базе `python:3.11-slim`, apt-зависимости не нужны для wheels.

### 14.2 Сиды

- Глобально установить в начале обучения: `random`, `numpy`, `torch`, `torch.cuda` (если есть). Pytorch Lightning — `seed_everything(seed, workers=True)`.
- Сид нарезки данных — отдельный, фиксируется в манифесте.

### 14.3 Docker

`Dockerfile` — multi-stage, один образ (app + inference, без ML extras):
1. **builder:** ставит `uv`, копирует `pyproject.toml` + `requirements.lock`, выполняет `uv pip sync` в venv.
2. **runtime:** на `python:3.11-slim`, копирует venv + код, EXPOSE 8000, CMD `uvicorn floodrisk.app:app --host 0.0.0.0 --port 8000`.

В образе монтируются volumes: `./runs`, `./exports`, `./models`, `./app.db`. Для запуска используется `make docker-run` (см. §10.2).

> **Без docker-compose.** Сервис один (backend + SQLite), оркестрация не нужна. `docker run` с volume-mount-ами проще, понятнее в README, и быстрее запускается.

### 14.4 Переменные окружения (`.env`)

```
DATABASE_URL=sqlite:///./app.db          # дефолт, можно не задавать
CDS_API_KEY=...                          # нужен только для `make data` (загрузка ERA5)
COPERNICUS_TOKEN=...                     # нужен только для `make data` (Sentinel-1)
MLFLOW_TRACKING_URI=file://./mlruns      # нужен только для `make train`/`make eval`
LOG_LEVEL=INFO
```

`.env.example` коммитится с пустыми значениями. `.env` — в `.gitignore`. App/inference контейнеру токены не нужны (он не качает данные).

### 14.5 Рекомендации по среде разработки

Не часть приёмки, но критично для соблюдения M-1 (сборка ≤ 1 день).

**GDAL/rasterio на Windows.** Wheels стали стабильнее, но конфигурация PROJ периодически ломается. **Рекомендация:** всё, что трогает геоданные (`make data`, `make train`, `make eval`), запускать через `make docker-run` даже на Windows. Хост-Python — только для редактора, тестов чистой логики и `make run` с hot-reload.

**GPU для тренировки U-Net.** На CPU 50 эпох на 1k тайлов 256×256 — порядка десятков часов. Целевые варианты:
- Локальная NVIDIA GPU 8+ ГБ VRAM (RTX 3060 и выше).
- Google Colab Pro / Colab Pro+ (T4 / A100).
- vast.ai / RunPod (на час).

В `requirements-ml.lock` фиксируется CPU-версия torch для портабельности; GPU-версия устанавливается отдельно в тренировочном окружении из CUDA-индекса PyTorch (`uv pip install torch --index-url https://download.pytorch.org/whl/cu121`). Это документируется в README.

**MLflow UI.** Запускается локально (`mlflow ui --backend-store-uri ./mlruns`), не в Docker. Не блокирует ничего.

---

## 15. Критерии приёмки (Definition of Done)

Демо считается принятым, если **все** критерии выполнены:

| # | Критерий | Способ проверки | Порог |
|---|---|---|---|
| M-1 | Воспроизводимая сборка | `docker compose up` + `make data` + `make train` на чистой машине из тага | ≤ 1 рабочего дня до получения карты |
| M-2 | Качество модели | Метрики из `reports/comparison_v1.md` | U-Net > бейзлайн по IoU **и** F1 **и** PR-AUC; bootstrap-CI 95% не пересекаются |
| M-3 | Скорость инференса | 20 запросов к `/api/predict` на фиксированном bbox, замер p95 | ≤ 5 сек CPU, ≤ 1 сек GPU |
| M-4 | UI-сценарии | Ручной прогон U1 + U2 + U3 на пилотной территории | 0 ошибок, 0 фризов > 2 сек на действие |
| M-5 | Объяснимость | Для выбранного тайла открывается карта важности признаков | Карта отрисована; метод + ограничения в README + 1 пример |
| M-6 | Документация | Внешний человек по README поднимает систему | ≤ 1 день, 0 обращений «не запустилось» |
| M-7 | Тесты + линт | `make test` + `make lint` | Оба exit 0; pytest coverage core ≥ 60% |

---

## 16. Открытые вопросы (переносятся из PRD, не блокируют старт)

> Эти вопросы должны быть закрыты по ходу R&D. Пока — параметризуем, чтобы не блокировать разработку.

| # | Вопрос | Что блокирует | Workaround на старте |
|---|---|---|---|
| OQ-1 | Целевой EPSG и размер тайла | FR-2, контракт `/api/predict` | Параметризовано в `configs/data.yaml`, дефолт: UTM-зона из bbox, 256 px |
| OQ-2 | Методика разметки Sentinel-1 | FR-3, M-2 | На первой итерации — пороги обратного рассеяния из литературы + ручная валидация на 1 событии |
| OQ-3 | Что делать при обновлении внешних источников | FR-4, M-1 | На старте — фиксация версий по дате выгрузки; если разъезжается — зеркало на S3 (выносится в LATER) |
| OQ-4 | Финальный метод объяснимости + размер окна | FR-10, FR-18, M-5 | На старте — Integrated Gradients (captum), окно 256×256 |
| OQ-5 | Формат PDF-отчёта | FR-17, UF-4 | ReportLab, минимальный шаблон: заголовок, карта, легенда, метаданные |
| OQ-6 | Метод оценки значимости | FR-7, M-2 | Bootstrap-CI 95% (1000 ресэмплов) |
| OQ-7 | Параметры сценариев `p1y/p10y/p100y` | FR-11, FR-15, UF-4 | На старте — плейсхолдеры в `configs/scenarios.yaml`, заполняются после R&D-фазы данных |
| OQ-8 | Допустимая дисперсия метрик между прогонами | NFR-7 | На старте — ±2% IoU, уточняется после 5 прогонов |
| OQ-9 | Финальный выбор пилотной территории | Закрытие FR data-слоя в production-режиме | Параметризованный bbox |
| OQ-10 | Где живёт фронт в Compose | FR-19 | **Закрыто:** статика отдаётся самим FastAPI через `StaticFiles`. Отдельного nginx нет. |
| OQ-11 | MLflow vs W&B | FR-5, FR-6, FR-7 | **Закрыто:** MLflow file-backend в `mlruns/` |
| OQ-12 | Токены для CI, кешированный мини-датасет | Smoke-тесты в CI | **Закрыто:** мини-датасет в `tests/fixtures/`; data-smoke с реальными API — за `@pytest.mark.requires_network`, не в CI |
| OQ-13 | Граница зоны покрытия для FR-12 | `/api/predict` 422 | На старте — bbox манифеста + буфер 0; уточняется после фиксации тайлинга |
| OQ-14 | Добавлять ли Alembic к фазе подготовки к защите | Сохранение истории запусков в БД между релизами | На старте — без Alembic, `make db-reset` пересоздаёт схему. Решается на Этапе 6 (подготовка к защите): если хочется сохранить историю прогонов — initial-миграция + autogenerate. |

---

## 17. План реализации (этапы)

> Этапы упорядочены по зависимостям. Внутри этапа задачи можно параллелить.

### Этап 0. Каркас (1–2 дня)
- `pyproject.toml` (с extras `[ml]`), `Makefile`, `Dockerfile`, `.dockerignore`, `.env.example`, ruff-конфиг, pre-commit.
- `uv venv` + `uv pip compile` → `requirements.lock` + `requirements-ml.lock`.
- Скелет `floodrisk/` (пустые модули).
- `db/models.py` (SQLModel, 5 таблиц из §5) + `db/session.py` (`create_db_and_tables()`).
- `make seed` — наполнение `scenarios` из YAML.
- FastAPI app factory + `/health` endpoint + `StaticFiles` + Jinja2 + minimal `index.html` с Tailwind/HTMX/Alpine.js/Leaflet CDN-подключениями.
- pytest setup + `tests/fixtures/mini_dataset/` (синтетический).
- CI workflow (ruff + pytest без `requires_network`).
- **Готовность:** `make run` поднимает приложение, открывается пустая страница с Leaflet-картой; `make test` зелёный; `make lint` зелёный.

### Этап 1. Data pipeline (R&D-fase 1, тайм-бокс)
- FR-1, FR-2, FR-3, FR-4.
- `configs/data.yaml`, `configs/scenarios.yaml` (плейсхолдеры).
- **Риск:** разметка Sentinel-1 (OQ-2). При блокере — сужаем пилотную территорию, не урезаем ML.
- **Готовность:** `make data` собирает датасет v1 + manifest; `make verify-data` exit 0.

### Этап 2. ML training (R&D-fase 2)
- FR-5, FR-6, FR-7.
- `configs/unet_v1.yaml`, `configs/baseline_v1.yaml`.
- MLflow интеграция.
- `make train`, `make eval`.
- **Готовность:** `reports/comparison_v1.md` существует; M-2 либо pass, либо явно зафиксированный fail с планом исправления.

### Этап 3. Inference service
- FR-8 (CLI), FR-9, FR-11, FR-12, FR-13.
- Резидентная модель, конверсия TIF→PNG для Leaflet.
- Регистрация моделей в БД (`make seed`).
- **Готовность:** Swagger показывает все эндпоинты; smoke-test в CI зелёный.

### Этап 4. Frontend
- FR-14, FR-15, FR-16, FR-17.
- `templates/`, `static/app.js`, Tailwind через CDN.
- **Готовность:** UF-4 проходит руками в Chrome и Firefox.

### Этап 5. Объяснимость
- FR-10, FR-18.
- Captum integrated gradients.
- **Готовность:** M-5 выполнен; README дополнен.

### Этап 6. Документация и приёмка
- FR-20 (полный README).
- Прогон M-1 на чистой машине внешним человеком.
- Финализация `reports/comparison_v1.md`.
- **Готовность:** все M-1…M-7 pass.

---

## 18. Журнал решений

- **2026-05-29 — v1.0.** SRS сформировано на основе discovery v1.0, concept v0.1, PRD v0.1. Согласованы изменения относительно PRD:
  1. Frontend: React + MapLibre → **одна HTML-страница + HTMX + Tailwind + Leaflet через CDN**. Никаких SPA-фреймворков и сборщиков.
  2. БД: PostGIS → **SQLite (dev) / Postgres (compose, без PostGIS)**, только метаданные. Геоданные — файлы.
  3. ORM/миграции: добавлены SQLAlchemy + Alembic.
  4. Карта: Leaflet вместо MapLibre (проще для растровых overlay, меньше зависимостей).
  5. Закрыты OQ-10 (фронт отдаётся самим FastAPI), OQ-11 (MLflow file-backend), OQ-12 (мини-датасет в фикстурах).
  6. Добавлены NFR-11 (coverage core ≥ 60%) и NFR-12 (ruff).
  7. ML-стек (PyTorch + Lightning + smp + MLflow + captum) сохранён полностью.

- **2026-05-29 — v1.1.** Ревизия стека по итогам критической оценки:
  1. **SQLAlchemy + Alembic → SQLModel без миграций.** Для 5-таблиц прототипа с регенерируемыми из конфигов данными — Alembic-ceremony избыточна. `make db-reset` пересоздаёт схему. Возврат к Alembic — опция OQ-14, решается на Этапе 6.
  2. **PostgreSQL (compose-профиль) — выкинут.** Один пользователь, демо на ноуте, никакой PostGIS-специфики. SQLite WAL — единственная СУБД. Миграция на Postgres в LATER = смена `DATABASE_URL`, без кода.
  3. **pip-tools → uv.** Тот же автор, что ruff (Astral). Установка в 10-100× быстрее — прямой выигрыш под M-1.
  4. **ML-зависимости через extras `[ml]`.** Pytorch-Lightning / MLflow / segmentation-models-pytorch / albumentations / xarray ставятся только в тренировочном окружении. App/inference Docker-образ сократился с ~3 ГБ до ~800 МБ. Captum остаётся в core (нужен для `/api/explain` в рантайме).
  5. **Alpine.js добавлен.** Микро-реактивность (режимы UI, спиннеры) — на Alpine; HTMX — на серверный обмен; vanilla JS — только мост к Leaflet. Каждая часть удерживается в 50-100 строк.
  6. **docker-compose выкинут.** Сервис один (backend + SQLite). `docker run` с volume-mounts проще и быстрее.
  7. **Добавлен §14.5** с рекомендациями: GDAL/rasterio на Windows запускать в Docker; план под GPU для тренировки (локальная NVIDIA / Colab Pro / vast.ai); CPU torch в lockfile, GPU torch — отдельной командой из CUDA-индекса.
  8. Добавлены: `pre-commit`, `pytest-cov`, отдельный `requirements-ml.lock`.
