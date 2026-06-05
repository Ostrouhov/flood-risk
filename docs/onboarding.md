# floodrisk — онбординг (для внесения правок)

> Подробный разбор: зачем проект, как устроен по этапам, где что править, и FAQ по концепциям.
> Финальное ТЗ — [srs.md](srs.md) (приоритет при расхождениях). Краткий запуск — [../README.md](../README.md).

---

## Часть 1. Зачем это и какая ценность

### Что делает (одним абзацем)
Веб-сервис: по **bbox** (прямоугольнику на карте) и **метео-сценарию** (повторяемость осадков
1 / 10 / 100 лет) рисует **растровую карту подверженности затоплению** — в каждом пикселе число
0…1, насколько место склонно затапливаться. Карту строит нейросеть **U-Net**, обученная на рельефе
и сопутствующих геопризнаках открытых данных.

### Чего НЕ делает (критично для защиты)
- ❌ **Не прогноз во времени** — не «затопит во вторник», а «при таком ливне эта низина затопится с вероятностью 0.8».
- ❌ Не гидродинамика (волна паводка по часам), не глубина воды, не ущерб в деньгах, не замена госэкспертизы.
- «Будущее» здесь = выбранный **сценарий осадков**, не дата.

### Ценность
1. Метод: попиксельная подверженность по рельефу нейросетью на **бесплатных открытых данных** (DEM, Sentinel-1, WorldCover, OSM, ERA5).
2. Честное сравнение U-Net против простого бейзлайна с **доверительными интервалами**.
3. Воспроизводимость: манифест с sha256, фиксированные сиды, lock-файлы, Docker.
4. Объяснимость: для любой точки — какие признаки повлияли (Integrated Gradients).
5. Рабочее демо под защиту: открыл браузер → выбрал сценарий → карта → объяснение → отчёт.

> **Главный честный результат (M-2 = FAIL):** на датасете из одного события (Тулун-2019) U-Net
> направленно лучше бейзлайна по всем метрикам, но доверительные интервалы пересекаются →
> статистически значимого превосходства нет. Узкое место — **данные** (одно событие, дисбаланс
> ~1.2% затопленных пикселей), не метод. Это сильный результат: показывает границы и путь развития.

---

## Часть 2. Поток данных end-to-end

```
ОФФЛАЙН (один раз, тяжёлое, окружение [ml]):
  Открытые источники ─fetch─► сырые растры (DEM, S1, WorldCover, OSM, ERA5, JRC)
     ─preprocess─► производные рельефа + проекция EPSG:32647 + тайлы 256×256
     ─labels_s1─►  бинарные маски «затоплено/нет» по Sentinel-1 (минус постоянная вода)
     ─manifest─►   data/manifest.yaml (sha256, splits train/val/test)
     ─train─►      U-Net (smp resnet34) + baseline (RandomForest) → models/
     ─export─►     U-Net → TorchScript (model.ts.pt) для лёгкого инференса
     ─eval─►       reports/comparison_v1.md (метрики + bootstrap-CI, вердикт M-2)

ОНЛАЙН (демо, лёгкое, окружение core):
  Браузер ─bbox+сценарий─► FastAPI /ui/predict ─► inference.service.predict():
     1) вырезать окно из мозаики data/processed/v1/features/stack.tif
     2) нормировать (feature_transform: 7 сырых → 18 каналов)
     3) прогнать резидентную U-Net (sliding window) → base_prob
     4) применить сценарий = сдвиг логита β=log10(RP/10) → prob
     5) записать prediction.tif (32647) + prediction.png (4326, Viridis) + агрегаты
  ◄─ HTML-фрагмент с data-атрибутами ─► app.js рисует overlay на Leaflet
  Клик «Объяснение» ─► /ui/explain ─► captum IG ─► топ-5 признаков + PNG-слои (magma)
  Экспорт ─► /ui/export ─► ZIP (prediction.tif + aggregates.geojson + report.pdf)
```

---

## Часть 3. Этапы 0–6

- **Этап 0 — каркас.** [app.py](../floodrisk/app.py) (фабрика FastAPI), [settings.py](../floodrisk/settings.py) (все пути/конфиг), [db/](../floodrisk/db/) (SQLite/SQLModel, 5 таблиц, seed). Команды: `make db-reset`, `make run`.
- **Этап 1 — данные (FR-1…4).** [data/](../floodrisk/data/): fetch (Planetary Computer STAC + OSM + ERA5) → preprocess (рельеф+TWI, проекция, тайлы) → labels_s1 (Otsu + change detection, минус JRC/OSM воду) → manifest (sha256). Конфиг [configs/data.yaml](../configs/data.yaml). Датасет v1 = 208 тайлов, 7 каналов, flood ~1.2%. Команды: `make data`, `make verify-data`. **NB (багфикс 2026-06):** `preprocess.mosaic_source_to_file` мозаичит ВСЕ перекрывающие сетку тайлы DEM/WorldCover/JRC (раньше брался один → рельеф был пуст вне его покрытия). Мультирегион: 2-й конфиг [configs/data_kansk.yaml](../configs/data_kansk.yaml) (Канск, EPSG:32646) обрабатывается отдельным прогоном `--config`; см. `reports/acceptance.md`.
- **Этап 2 — ML (FR-5…7).** [ml/](../floodrisk/ml/): model.py (U-Net smp resnet34, BCE+Dice+pos_weight), baseline.py (RandomForest попиксельно), train.py (+MLflow), export.py (TorchScript), evaluate.py+metrics.py (bootstrap-CI). Вход = **18 каналов** (см. FAQ). Команды: `make train`, `make eval`.
- **Этап 3 — инференс (FR-8/9/12/13).** [inference/service.py](../floodrisk/inference/service.py) — сердце (predict). [features.py](../floodrisk/inference/features.py) (окно+sliding+покрытие), [registry.py](../floodrisk/inference/registry.py) (резидентные модели), [scenario.py](../floodrisk/inference/scenario.py) (logit-сдвиг). Артефакты в `runs/<id>/`.
- **Этап 4 — фронтенд (FR-14…17).** [templates/index.html](../floodrisk/templates/index.html) + [fragments/](../floodrisk/templates/fragments/), [routes_html.py](../floodrisk/api/routes_html.py) (HTMX), [static/app.js](../static/app.js) (мост к Leaflet), [exporter.py](../floodrisk/inference/exporter.py) (ZIP/PDF).
- **Этап 5 — объяснимость (FR-10/18).** [inference/explain.py](../floodrisk/inference/explain.py): captum IG по U-Net (18→7 признаков) + RF feature_importances; топ-5 + PNG-слои magma.
- **Этап 6 — документация и приёмка (FR-19/20, M-1…7).** README, Dockerfile (CPU-torch), [reports/latency.md](../reports/latency.md), [reports/acceptance.md](../reports/acceptance.md).

### Расширения после Этапа 6 (2026-06)

Не из исходного плана этапов, добавлено итеративно (см. `reports/acceptance.md` и авто-память):

- **Глобальный онлайн-режим** — инференс по любому bbox; признаки собираются на лету ([online_features.py](../floodrisk/inference/online_features.py)), `source=auto/mosaic/online`.
- **UX-кластеры 1–3** — рисуемый bbox, инспектор точки, метаданные, прозрачность; swipe U-Net↔baseline; **валидация «реальность S1 ↔ предсказание»** ([validation.py](../floodrisk/inference/validation.py), `GET /api/runs/{id}/groundtruth`) с живыми IoU/F1; геокодер (Nominatim), permalink, подложка-спутник (Esri).
- **🛠 КРИТ-багфикс признаков** — `run_preprocess` брал ОДИН тайл вместо мозаики → рельеф пуст в ~94%; фикс `preprocess.mosaic_source_to_file`; **Тулун перестроен и переобучен**, M-2 переоценён на валидных данных (всё ещё FAIL, но честно).
- **Мультирегион (демо)** — 2-й регион **Канск**; инференс выбирает региональную мозаику по bbox (`service._mosaic_stacks`); 2 контура покрытия; Канск помечен `experimental` (S1-лейблы шумные, не в M-2). `ml/combine.py` + мультирегион-`tile_paths` — scaffolding под обучение на 2+ событиях (НЕ задействовано).

---

## Часть 4. Как пользоваться

| Способ | Команда / действие |
|---|---|
| UI | `make run` → http://127.0.0.1:8000 |
| CLI инференс | `floodrisk infer --bbox 100.64,54.54,100.89,54.71 --scenario p100y --model unet-v1 --out runs/x` |
| REST API | `POST /api/predict` `{bbox, scenario_id, model_version}`; Swagger `/docs` |
| Бенчмарк | `make bench` |
| Обучение | `make install-ml` → `make train` → `make eval` |
| Тесты/линт | `make test`, `make lint` |

---

## Часть 5. Где что править

| Изменить… | Файл |
|---|---|
| Логику инференса | [inference/service.py](../floodrisk/inference/service.py) |
| Мультирегион-мозаику (выбор стека по bbox) | [inference/service.py](../floodrisk/inference/service.py): `_region_stacks`/`_mosaic_stacks`/`mosaic_coverages` |
| Онлайн-сборку признаков (вне покрытия) | [inference/online_features.py](../floodrisk/inference/online_features.py) |
| Формулу сценария | [inference/scenario.py](../floodrisk/inference/scenario.py) |
| Нормировку/каналы | [feature_transform.py](../floodrisk/feature_transform.py) + [configs/unet_v1.yaml](../configs/unet_v1.yaml) |
| Объяснимость | [inference/explain.py](../floodrisk/inference/explain.py) |
| Валидацию «реальность S1 ↔ предсказание» (IoU/F1) | [inference/validation.py](../floodrisk/inference/validation.py) |
| JSON API | [api/routes_api.py](../floodrisk/api/routes_api.py), [api/schemas.py](../floodrisk/api/schemas.py) |
| HTML/HTMX-роуты | [api/routes_html.py](../floodrisk/api/routes_html.py) |
| Внешний вид | [templates/](../floodrisk/templates/) |
| Поведение карты (шторка, геокодер, permalink, подложки) | [static/app.js](../static/app.js) |
| ZIP/PDF | [inference/exporter.py](../floodrisk/inference/exporter.py) |
| Параметры сценариев | [configs/scenarios.yaml](../configs/scenarios.yaml) → `make seed` |
| Гиперпараметры | [configs/unet_v1.yaml](../configs/unet_v1.yaml), [ml/train.py](../floodrisk/ml/train.py) |
| Метрики | [ml/evaluate.py](../floodrisk/ml/evaluate.py), [ml/metrics.py](../floodrisk/ml/metrics.py) |
| Сборку данных (мозаика тайлов!) | [data/preprocess.py](../floodrisk/data/preprocess.py) `mosaic_source_to_file`, [data/pipeline.py](../floodrisk/data/pipeline.py), [configs/data.yaml](../configs/data.yaml) |
| Доп. регион (2-е событие) | новый `configs/data_<region>.yaml` → `data <action> --config …` (EPSG своей UTM-зоны) |
| Объединение регионов для обучения (scaffolding) | [ml/combine.py](../floodrisk/ml/combine.py) + [ml/data.py](../floodrisk/ml/data.py) `tile_paths` (пер-строчные пути) |
| Схему БД | [db/models.py](../floodrisk/db/models.py) (после — `make db-reset`) |

**Перед «готово» (как в CI):** `ruff format --check . ; ruff check . ; pytest -m "not requires_network and not ml_smoke"`.

---

## Часть 6. FAQ по концепциям

### 1. Почему именно U-Net?
Задача — **семантическая сегментация растра**: на входе многоканальное «изображение» местности,
на выходе карта **той же геометрии**, где каждый пиксель = вероятность. U-Net создан ровно для этого:
- **encoder-decoder со skip-connections** — сначала сжимает картину (улавливает контекст: «это дно
  долины, вокруг склоны»), потом восстанавливает в исходное разрешение, не теряя мелких деталей рельефа;
- затопление **локально-пространственно** (зависит от соседних пикселей — куда стекает вода), а не от
  одного пикселя в отрыве — свёрточная сеть это учитывает, табличная модель нет;
- хорошо работает на **небольших датасетах** с аугментацией; есть готовая реализация
  (`segmentation-models-pytorch`), лёгкий энкодер resnet34 тянется на CPU.
Альтернативы (попиксельные классификаторы) теряют пространственный контекст; более тяжёлые сети
избыточны для прототипа и данных. U-Net — баланс «контекст + мало данных + готовый инструмент».
Выбор зафиксирован в [srs.md §7.2](srs.md).

### 2. Что значит «повторяемость 1 / 10 / 100 лет»?
**Это НЕ длительность дождя.** Дождь не идёт «10 лет подряд». Повторяемость (return period) — это
**насколько редкий и сильный ОДИН ливень**:
- **p1y** — заурядный ливень, какой случается примерно раз в год (слабый сценарий);
- **p10y** — сильный ливень, какой бывает в среднем раз в 10 лет;
- **p100y** — экстремальный ливень, какой бывает раз в 100 лет (очень мощный, «столетний паводок»).

Формально: «событие повторяемостью N лет» = осадки такой силы, что вероятность их превышения
в любой отдельный год = 1/N. Чем больше N — тем **интенсивнее** разовое событие, а не дольше.
В проекте каждый сценарий — пресет интенсивности в [configs/scenarios.yaml](../configs/scenarios.yaml)
(например, `precipitation_mm_24h`: p1y=25 мм, p10y=55 мм, p100y=110 мм за сутки). Сильнее ливень →
больше площади подвержено затоплению (на карте — больше «горячих» зон).

### 3. «Низина уйдёт под воду» — на сколько? На сантиметр? На метр?
Модель **не предсказывает глубину**. Она предсказывает **факт затопления** (да/нет) и выдаёт
**вероятность этого факта** ∈ [0,1]. То есть «0.8» = «с вероятностью 80% этот пиксель окажется
**внутри зоны затопления**», а не «вода поднимется на 0.8 м».
Причина: разметка для обучения бралась с Sentinel-1 как **бинарная маска** (затоплено / не затоплено),
без уровней. Глубина и динамика воды — это гидродинамика, явный **анти-гол** проекта
([srs.md §1.3](srs.md)). Если на защите спросят про глубину — честный ответ: «вне скоупа; выход —
вероятность принадлежности к зоне затопления, не уровень воды».

### 4. Что такое baseline? Это не нейросеть?
**baseline (бейзлайн) — это «эталон для сравнения», намеренно простая модель.** Здесь это
**Random Forest** (случайный лес) из scikit-learn — классический ML, **НЕ нейросеть**.
- Как работает: каждый пиксель рассматривается **независимо** как строка-объект со своими 18 признаками;
  лес из решающих деревьев голосует «затоплено / нет». Пространственного контекста (соседей) он не видит.
- Зачем нужен: чтобы ответить на главный научный вопрос — **а оправдана ли сложная пространственная
  нейросеть?** Если U-Net обыгрывает простой попиксельный лес значимо — сложность оправдана. Если нет
  (наш случай, M-2 fail) — значит, при текущих данных «умная» модель не даёт статистически значимого
  выигрыша, и узкое место — данные.
- Как сравниваем: обе модели гоняются на **одних и тех же** тестовых тайлах, считаются одинаковые
  метрики (IoU, F1, ROC-AUC, PR-AUC, Brier) с **bootstrap доверительными интервалами 95%**. Сравнение
  и вердикт — в [reports/comparison_v1.md](../reports/comparison_v1.md), код — [ml/evaluate.py](../floodrisk/ml/evaluate.py).
- Метрики коротко: **IoU/F1** — насколько предсказанная зона совпадает с реальной; **ROC-AUC/PR-AUC** —
  насколько хорошо модель ранжирует пиксели по риску; **Brier** — насколько откалибрована вероятность
  (меньше = лучше).
