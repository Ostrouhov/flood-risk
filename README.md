# floodrisk

Веб-сервис, который по **bbox** территории и **метео-сценарию** (`p1y` / `p10y` / `p100y`)
строит растровую карту **пространственной подверженности затоплению** — вероятность ∈ [0, 1]
на каждый пиксель, предсказанную моделью U-Net по рельефу и сопутствующим геопризнакам.

> **Постановка (важно).** Это **не прогноз во времени**. Модель не предсказывает, *когда*
> случится наводнение. Она оценивает, *насколько подвержена* территория затоплению при
> заданной интенсивности осадков. «Будущее» здесь = выбранный сценарий повторяемости
> (1 / 10 / 100 лет), а не дата. Прототип создан под защиту диссертации; один пользователь,
> демо на ноутбуке, не продакшн.

Полное ТЗ — [docs/srs.md](docs/srs.md) (приоритет при любых расхождениях с этим README).

---

## Содержание

- [Архитектура](#архитектура)
- [Предусловия](#предусловия)
- [Установка](#установка)
- [Быстрый старт: демо](#быстрый-старт-демо)
- [Сборка данных](#сборка-данных-этап-1)
- [Обучение моделей](#обучение-моделей-этап-2)
- [Воспроизведение метрик](#воспроизведение-метрик-m-2)
- [CLI](#cli)
- [Docker](#docker)
- [Объяснимость и её ограничения](#объяснимость-и-её-ограничения-m-5)
- [Тесты, линт, покрытие](#тесты-линт-покрытие)
- [Ограничения и known-gaps](#ограничения-и-known-gaps)
- [Статус](#статус)

---

## Архитектура

```
Browser (одна HTML-страница: HTMX + Alpine.js + Tailwind + Leaflet, всё через CDN)
   │  HTTP (HTMX form-posts, JSON API)
FastAPI + Uvicorn (1 worker)
   ├─ JSON API: /api/predict, /api/explain, /api/scenarios, /api/runs/{id}/export …
   ├─ HTML-роуты (HTMX-фрагменты, Jinja2): /ui/predict, /ui/explain, /ui/export
   ├─ Inference service: резидентные U-Net (TorchScript) + baseline (sklearn) в памяти
   └─ SQLModel ORM ──► SQLite (app.db, только метаданные)
                       Файлы ──► data/  models/  runs/  exports/
```

Геоданные (растры, векторы) — файлы; БД хранит только метаданные запусков. ML-стек
(PyTorch Lightning / smp / MLflow) — отдельное окружение `[ml]`, в инференс-образ не входит.
Подробнее — [docs/srs.md §2](docs/srs.md).

---

## Предусловия

- **Python 3.11** (строго; проект не собирается на 3.12+).
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** ≥ 0.4 — менеджер зависимостей и venv.
- **GNU Make** — единая точка входа (на Windows — через Git Bash; либо запускать команды напрямую, см. ниже).
- **Docker** (опционально) — для воспроизводимого запуска и для GDAL-шагов на Windows.

> **Windows.** Все Python-команды — через `.venv\Scripts\python.exe` (не системный `python`).
> Примеры ниже даны и в Make-форме, и в прямой PowerShell-форме.

---

## Установка

Проект делит зависимости на три набора (extras):

| Набор | Что входит | Когда нужен |
|---|---|---|
| **core** | FastAPI, rasterio, torch (CPU), sklearn, captum… | всегда (сервис + инференс + объяснимость) |
| **`[data]`** | xarray, rioxarray, cdsapi, pystac-client… | сборка датасета (Этап 1) |
| **`[ml]`** | pytorch-lightning, smp, albumentations, mlflow (+ `[data]`) | обучение (Этап 2), отдельное окружение |

```bash
uv venv .venv --python 3.11
# Linux/macOS: source .venv/bin/activate ; Windows (Git Bash): . .venv/Scripts/activate
make lock        # сгенерировать requirements*.lock
make install     # core
# опционально:
make install-data   # core + [data]
make install-ml     # core + [ml]  (лучше в отдельный .venv-ml)
```

Прямой вызов без make (любая ОС):

```bash
uv pip sync requirements.lock          # core
```

> **GPU для обучения.** В lock зафиксирован CPU-torch (портабельность). Для GPU поставьте
> CUDA-сборку отдельно в тренировочном окружении:
> `uv pip install torch --index-url https://download.pytorch.org/whl/cu121` (см. [docs/srs.md §14.5](docs/srs.md)).

---

## Быстрый старт: демо

Датасет v1 и обученные модели уже лежат в репозитории (см. [Статус](#статус)), поэтому для
демо достаточно поднять сервис. Пошаговый сценарий показа под защиту (что нажать → что видно →
что сказать) — [docs/demo-walkthrough.md](docs/demo-walkthrough.md).

```bash
make db-reset    # создать app.db + засеять сценарии и модели
make run         # http://127.0.0.1:8000
```

PowerShell без make:

```powershell
.venv\Scripts\python.exe -m floodrisk.db.session create
.venv\Scripts\python.exe -m floodrisk.db.seed
.venv\Scripts\python.exe -m uvicorn floodrisk.app:app --port 8000
```

В браузере (Chrome/Firefox, десктоп):

1. **Поиск места** (геокодер OpenStreetMap) или панорама карты; **подложка** OSM ↔ спутник Esri (control справа вверху).
2. Выберите **сценарий** (`p1y` / `p10y` / `p100y`), нарисуйте **bbox** мышью («Выделить на карте»).
3. **Пересчитать** → поверх карты ляжет полупрозрачный растр (Viridis); слева — агрегаты и метаданные расчёта.
4. **Горячие точки риска** → топ-зоны (связные кластеры p≥0.5, ранжированы по площади×вероятности):
   пронумерованные маркеры на карте + список; клик по строке/маркеру — зум к зоне.
5. **Валидация «Реальность S1 ↔ предсказание» (шторка)** → справа реальная маска затопления 2019
   (Sentinel-1), слева предсказание; в таблице **IoU/F1** на зоне. **U-Net ↔ baseline (шторка)** — суть M-2.
6. **Экспорт** → ZIP с `prediction.tif` + `aggregates.geojson` + `report.pdf`.
7. Режим **«Объяснение»** → клик по карте → панель важности признаков + слой атрибуции.
8. **Permalink**: текущий вид (bbox + сценарий + модель) пишется в URL — ссылку можно сохранить/расшарить.

> **Валидированные зоны (быстрая офлайн-мозаика, 2 контура на карте):**
> **Тулун** (р. Ия, EPSG:32647, обучен+проверен) и **Канск** (р. Кан, EPSG:32646; признаки корректны,
> но S1-лейблы экспериментальны — помечены `experimental`, в M-2 не входят). Вне обеих зон — **глобальный
> онлайн-режим** (см. ниже). Инференс выбирает регион по bbox автоматически.

## Глобальный режим (онлайн-инференс по произвольной зоне)

Можно выбрать зону **в любой точке мира**: панорамируйте карту, приблизьте район, выберите
«Текущий вид карты» → Пересчитать. Признаки рельефа (DEM, уклон/экспозиция/кривизна/TWI,
землепокров, расстояние до воды) собираются **на лету** из глобальных открытых источников
(Copernicus DEM, ESA WorldCover, JRC GSW через Planetary Computer), без Sentinel-1.

- **Источник** выбирается полем `source` (`POST /api/predict`): `auto` (по умолчанию — мозаика
  в покрытии Тулуна, иначе онлайн), `mosaic`, `online`. См. [inference/online_features.py](floodrisk/inference/online_features.py).
- **⚠ Экспериментально.** Модель обучена только на Тулуне-2019 → вне него качество **не
  валидировано** (domain shift). В UI и в `metadata.experimental=true` стоит пометка.
- **Latency:** холодный кэш ~10–40 с (доминирует загрузка тайлов), тёплый ~3–6 с. Это **выше
  NFR-1 (5 с)** — онлайн-режим намеренно отдельный путь с другим ожиданием; тайлы кэшируются в
  `data/cache/online/`.
- **Ограничения:** размер зоны ≤ ~30 км/сторона (`BBoxTooLarge` иначе); TWI считается на буфере
  вокруг bbox (приближение, без полного водосбора крупных рек). Глубину воды режим не даёт
  (как и основной — выход = вероятность зоны затопления).
- **Объяснимость работает и онлайн:** клик в режиме «Объяснение» вне Тулуна собирает окно
  признаков вокруг точки на лету (отсюда на холодном кэше отклик до ~минуты; в покрытии — ~5 с).

---

## Сборка данных (Этап 1)

> Датасет v1 **уже собран** (208 тайлов 256×256, 7 сырых каналов, Тулун-2019). Этот раздел —
> для воспроизведения с нуля или переноса на другой регион.

Источники: Copernicus DEM GLO-30, ESA WorldCover, OSM, ERA5/ERA5-Land, Sentinel-1 GRD, JRC GSW.
Регион и событие задаются в [configs/data.yaml](configs/data.yaml) (bbox + временное окно).

```bash
cp .env.example .env     # заполнить CDS_API_KEY, COPERNICUS_TOKEN (нужны только здесь)
make install-data
make data                # fetch → preprocess → label (Sentinel-1) → manifest
make verify-data         # сверка контрольных сумм, exit 0
```

Пайплайн: загрузка сырья → производные рельефа (slope/aspect/curvature/TWI) + расстояние до
воды → разметка затопления по Sentinel-1 (Otsu + change detection, минус постоянная вода
OSM/JRC) → нарезка тайлов + `index.parquet` + `data/manifest.yaml` (с sha256).

> **Мозаика источников.** DEM/WorldCover/JRC покрываются несколькими 1°-тайлами; `preprocess`
> склеивает **все** перекрывающие сетку (`mosaic_source_to_file`). **Второй регион** = отдельный
> конфиг (напр. [configs/data_kansk.yaml](configs/data_kansk.yaml), своя UTM-зона) и отдельный прогон:
> `python -m floodrisk data fetch --config configs/data_kansk.yaml --skip-era5` → `preprocess` → `label`.
> Демо-инференс затем сам выбирает региональную мозаику по bbox.

> **GDAL/PROJ на Windows** иногда нестабильны. SRS §14.5 рекомендует гонять `make data` в
> Docker-контейнере; хост-Python — для редактора, тестов и `make run`.

---

## Обучение моделей (Этап 2)

```bash
make install-ml          # отдельное окружение с тяжёлым ML-стеком
make train               # U-Net (smp resnet34) + baseline (RandomForest)
```

- Вход модели — **18 каналов**: 5 непрерывных z-score + sin/cos(aspect) + one-hot(worldcover, 11).
  Преобразование общее для train и inference — [floodrisk/feature_transform.py](floodrisk/feature_transform.py).
- Loss = BCE + Dice + pos_weight (дисбаланс flood ~1.2 %). Сиды фиксированы.
- Чекпоинт по best val IoU; метрики на эпоху → MLflow (`mlruns/`, `mlflow ui` опционально).
- На CPU (Ryzen 5 2600) ~15 мин. GPU-вариант — см. [Установка](#установка).
- Конфиги: [configs/unet_v1.yaml](configs/unet_v1.yaml), [configs/baseline_v1.yaml](configs/baseline_v1.yaml).

U-Net для инференса экспортируется в **TorchScript** (`models/unet/v1/model.ts.pt`) — рантайму
не нужен `segmentation-models-pytorch`.

---

## Воспроизведение метрик (M-2)

```bash
make eval     # → reports/comparison_v1.md (bootstrap-CI 95% по тайлам, 1000 ресэмплов)
```

Отчёт: [reports/comparison_v1.md](reports/comparison_v1.md).

**Результат M-2 — задокументированный FAIL (валидный научный итог).** U-Net направленно
лучше baseline по **всем** метрикам (IoU 0.0343 vs 0.0292; F1 0.0664 vs 0.0567;
PR-AUC 0.0348 vs 0.0304; ROC-AUC 0.896 vs 0.874; Brier 0.074 vs 0.112), но 95 %-CI по
IoU/F1/PR-AUC **пересекаются** (всего 32 test-тайла). Лимит — **датасет** (одно событие, сильный
дисбаланс, шум разметки), не кодировка признаков. Путь к pass — расширение датасета (>1
события/региона), это отдельный R&D.

> **Переоценено на исправленных признаках (2026-06).** Найден и устранён критический баг
> препроцесса: `run_preprocess` мозаичил только ОДИН исходный тайл (`_first`) вместо всех
> перекрывающих сетку → рельеф был пуст в ~94 % площади. После фикса
> (`preprocess.mosaic_source_to_file`) Тулун перестроен (DEM заполнен 98.6 %), модели
> переобучены, M-2 переоценён — вывод (лимит = данные) подтверждён уже на валидных признаках.
> См. [reports/acceptance.md](reports/acceptance.md).

**Калибровка вероятностей (только в отчёте).** Reliability показывает, что U-Net **завышает p**
(следствие `pos_weight=20` на редком классе ~1.2 %). В отчёт добавлена секция «Калибровка (Platt)»:
параметры подобраны на **val**, применены к **test** — Brier U-Net падает **0.074 → 0.010**.
Преобразование монотонно, поэтому ранжирование (ROC-AUC/PR-AUC) и хотспоты не меняются. В карту/serving
калибровку намеренно **не вносим**: сжатие p к базовой частоте обесцветило бы карту и обнулило порог
хотспотов p≥0.5 — карта использует именно **сырое ранжирование риска** (полезный сигнал), а калибровка
служит отчётной строгости. См. [reports/comparison_v1.md](reports/comparison_v1.md).

---

## CLI

`python -m floodrisk <subcommand>` (или `floodrisk …` после установки):

```bash
# инференс без поднятия API
floodrisk infer --bbox 100.64,54.54,100.89,54.71 --scenario p100y --model unet-v1 --out runs/demo

# данные / обучение / сравнение
floodrisk data verify --manifest data/manifest.yaml
floodrisk train --config configs/unet_v1.yaml
floodrisk eval  --models unet-v1,baseline-v1 --out reports/comparison_v1.md
```

`infer` пишет `prediction.tif`, `prediction.png`, `aggregates.json`, `metadata.json` в `--out`.
Объяснимость по точке доступна через API/UI (`/api/explain`).

---

## Docker

Образ — multi-stage на `python:3.11-slim`, **только core** (без `[ml]`). torch ставится с
CPU-индекса PyTorch, иначе на linux PyPI подтянул бы CUDA-сборку и образ превысил бы лимит
FR-19 (≤ 1 ГБ). См. [Dockerfile](Dockerfile).

**Стратегия артефактов — монтирование, не запекание.** Образ остаётся слим и переносимым;
данные/модели/БД подключаются томами при запуске. На хосте перед `docker-run` должны быть:

| Том | Что нужно | Как получить |
|---|---|---|
| `models/` | `unet/v1/model.ts.pt`, `baseline/v1/model.pkl` | в репозитории / `make train` |
| `data/`   | `processed/v1/features/stack.tif`, `processed/v1/norm_stats.json` | в репозитории / `make data` |
| `app.db`  | засеянные scenarios + model_versions | **сначала `make db-reset`** (иначе `-v app.db` создаст каталог) |
| `runs/`, `exports/` | результаты инференса (пишутся контейнером) | создаются автоматически |

```bash
make db-reset        # на хосте: создать и засеять app.db
make docker-build    # docker build -t floodrisk:latest .
make docker-run      # -p 8000:8000 + тома runs/ exports/ models/ data/ app.db
# → http://localhost:8000 ; smoke: curl http://localhost:8000/api/health
```

> Без docker-compose: сервис один (backend + SQLite), `docker run` с томами проще.
> CI собирает образ в job `docker-build-smoke` ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

> **Если `docker build` падает с TLS-таймаутом/EOF к pypi.org или CDN** (типично на
> Docker Desktop + WSL2 из-за MTU): задайте MTU в Docker Desktop → Settings → Docker Engine,
> добавив в JSON `"mtu": 1400`, **Apply & Restart**; либо `wsl --shutdown` и перезапуск Docker.
> Сама сборка проверена в CI на чистой сети.

---

## Объяснимость и её ограничения (M-5)

**Метод** ([floodrisk/inference/explain.py](floodrisk/inference/explain.py)):

- **U-Net** — Integrated Gradients (`captum`) по TorchScript-модели; окно 256×256 вокруг точки
  клика, n_steps=16, baseline=0. Атрибуции [18,H,W] агрегируются обратно в **7 семантических
  признаков** (DEM, slope, aspect, curvature, TWI, worldcover, dist_to_water).
- **baseline** — `RandomForest.feature_importances_`, тоже свёрнутые в 7 признаков (глобальная важность, без растров).
- **Выход** — топ-5 ранжирование (доля важности) + PNG-слои атрибуции (magma, EPSG:4326) в
  `runs/<id>/attribution/`. В UI: клик → панель + наложение слоя топ-признака.

**Пример** (реальный, p100y, центр Тулуна): Землепокров 0.47, Экспозиция склона 0.22,
TWI 0.14, dist_to_water 0.10, DEM 0.05. Реальная latency explain ~5.1 с (≪ NFR-3 15 с).

**Ограничения (читать перед интерпретацией):**

1. **Атрибуция ≠ причинность.** IG показывает, на что *чувствителен выход модели*, а не
   физический механизм затопления.
2. **Зависимость от baseline.** IG считается относительно нулевого baseline; при другом
   baseline ранжирование может сместиться. Неактивные one-hot worldcover дают атрибуцию 0
   (вход = baseline), поэтому важность не раздувается числом каналов.
3. **Один регион/событие в обучении** → атрибуции отражают специфику Тулуна-2019, не обобщены.
4. **Сценарий — пост-хок эвристика.** Влияние сценария = сдвиг логита β = log10(RP/10) (якорь
   p10y), применяется *после* инференса. Это **не обученная** зависимость от осадков и не
   входит в IG-атрибуцию.
5. Важности **не калиброваны** как вероятности и не имеют доверительных интервалов.

---

## Тесты, линт, покрытие

Как в CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)):

```bash
make lint     # ruff check . + ruff format --check .
make test     # pytest
# точная CI-выборка:
.venv\Scripts\python.exe -m pytest -m "not requires_network and not ml_smoke" -q
```

- Маркеры: `requires_network` (внешние API) и `ml_smoke` (требует `[ml]`) исключаются из CI-прогона.
- **Покрытие core ≥ 60 % (NFR-11): фактически ~81 %** (`pytest --cov`); ML-модули — smoke, не покрытие.
- Latency-бенчмарк: `make bench` → [reports/latency.md](reports/latency.md) (M-3/NFR-1, p95 ≤ 5 с).

---

## Ограничения и known-gaps

- **M-2 = FAIL** (см. [выше](#воспроизведение-метрик-m-2)): датасет = одно событие; расширение — отдельный R&D.
- **Лейблы Канска экспериментальны** — S1-детектор дал ~6.8 % воды (≈6× Тулуна, диффузно, без эталона
  Copernicus EMS) → Канск в M-2 не входит, в UI помечен `experimental`. Чистые лейблы 2-го события
  (тюнинг порогов/EMS) + обучение на 2+ событиях (`ml/combine.py` готов) — кандидат на следующий R&D.
- **OQ-8 (дисперсия метрик между прогонами, порог ±2 % IoU) измерена ✅.** U-Net обучен 3× с
  разными сидами (`python scripts/oq8_seed_variance.py`): размах IoU@0.5 = **0.0009** (mean 0.0328),
  глубоко внутри ±2 %. Вывод: модель воспроизводима по сидам; низкое абсолютное IoU — эффект редкого
  класса, не нестабильность. Детали — [reports/seed_variance.md](reports/seed_variance.md).
- БД-история запусков некритична: `make db-reset` пересоздаёт схему; `runs/<id>/` на диске
  остаются, но строки в БД пропадают (Alembic не используется — OQ-14).
- Не поддерживаются: мобильный режим, мультитенантность, real-time, дообучение через UI
  (полный список anti-goals — [docs/srs.md §1.3](docs/srs.md)).

---

## Статус

Этапы 0–6 закрыты (каркас → данные → ML → инференс → фронтенд → объяснимость → документация и
приёмка). Сводка критериев приёмки M-1…M-7 — [reports/acceptance.md](reports/acceptance.md).
План этапов — [docs/srs.md §17](docs/srs.md).
