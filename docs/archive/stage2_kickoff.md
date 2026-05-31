# Этап 2 — старт новой сессии (ML training)

Документ-подготовка к Этапу 2. Скопируй блок «Промт для новой сессии» в первое
сообщение новой сессии. Остальное — контекст и открытые решения.

## Промт для новой сессии (скопировать целиком)

Привет. Продолжаем floodrisk — прототип ИНС-оценки подверженности затоплению
(защита диссертации). Начинаем Этап 2 — ML training (FR-5, FR-6, FR-7 по SRS §17).

Контекст в памяти: project_floodrisk.md (статус, стек, железо — учим на CPU),
oq9-oq2-pilot-decisions.md (Этап 1 ЗАКРЫТ и перепроверен). Финальное ТЗ —
docs/srs.md (§7 ML-пайплайн, §7.8 конфиги, §11.2 FR-5..FR-7, §15 критерий M-2,
§16 OQ-6/OQ-8). docs/stage2_kickoff.md — открытые решения Этапа 2.

Что готово (Этап 0+1): датасет v1 проверен — data/processed/v1/, 208 тайлов
256x256, 7 каналов (dem, slope, aspect, curvature, twi, worldcover,
dist_to_water), лейблы затопления Тулун-2019, index.parquet split
train144/val32/test32, data/manifest.yaml. Окружение .venv (Python 3.11) с
extras [data]; ML-стек [ml] (lightning/smp/mlflow/albumentations) ЕЩЁ НЕ
установлен; torch CPU-версия. Конфиги-плейсхолдеры configs/unet_v1.yaml,
configs/baseline_v1.yaml. git УДАЛЁН (пересоздать позже).

Задача: обучить U-Net (FR-5) и табличный baseline (FR-6) на одних splits,
логировать в MLflow, сделать reports/comparison_v1.md с bootstrap-CI 95% (FR-7),
оценить M-2.

ВАЖНО про железо: учим на CPU (GTX 1660 SUPER 4GB не используем). Конфиг
тренировки щадящий: малый batch, num_workers около 6, лимит потоков, ПК не висит.

Сначала план, потом код. В плане первым делом разреши развилку «7 vs 10 каналов»
и установку ML-окружения. Работай решительно, мало вопросов; data/ML-шаги НЕ
запускай параллельно (были гонки за файлы); верь только exit-кодам и содержимому
файлов, не промежуточному выводу терминала.

## Открытые решения Этапа 2 (разрулить в начале)

### 1. Рассинхрон каналов: датасет даёт 7, unet_v1.yaml ждёт 10
Датасет v1 = 7 каналов (dem, slope, aspect, curvature, twi, worldcover,
dist_to_water). В configs/unet_v1.yaml стоит in_channels: 10 — заложены 3
метео-канала сценария (осадки), которых в тайлах НЕТ (ERA5 не качали, нет
CDS_API_KEY).

- (A) Учить на 7 каналах — in_channels: 7, убрать meteo_noise из аугментаций.
  Быстро, но модель не видит сценарий осадков, на инференсе сценарий не влияет
  (для демо-карты подверженности по рельефу — приемлемо).
- (B) Добавить 3 метео-канала из сценария — на сборке тайла подмешивать
  константные слои осадков сценария (из configs/scenarios.yaml, нормированные).
  Тогда сценарий реально влияет на выход (ближе к замыслу SRS). Требует доработки
  data/ml-слоя. Реальные поля ERA5 — отдельная опция, нужен CDS_API_KEY.

Рекомендация: (B) с константными каналами сценария (без ERA5) — даёт зависимость
от сценария малой кровью; (A) — быстрый MVP. Решить с пользователем.

### 2. Установка ML-окружения
Отдельный .venv-ml: uv venv .venv-ml; uv pip install -r requirements-ml.lock
(рабочий .venv не трогаем). Либо доустановить [ml] в текущий .venv (проще, но
мешает стеки). torch остаётся CPU. На Windows: $env:UV_HTTP_TIMEOUT="300".

### 3. Параметры тренировки под CPU
Сейчас unet_v1.yaml: batch_size 16, num_workers 4, precision 16-mixed,
max_epochs 50. Под CPU: precision -> 32 (16-mixed для CPU бессмыслен),
batch_size 4-8, num_workers 6, max_epochs скромнее на первый прогон (15-20),
лимит потоков torch. baseline (RandomForest) n_jobs можно -1.

### 4. Дисбаланс классов
flood ~1.2% пикселей — сильный дисбаланс. Loss BCE+Dice частично лечит;
рассмотреть Focal (опц. через конфиг) и/или взвешивание. baseline уже с
class_weight=balanced_subsample.

### 5. Сетка задач Этапа 2 (SRS §17, §11.2)
- FR-5: floodrisk.ml — Dataset/DataModule (index.parquet + тайлы),
  LightningModule (smp.Unet), train loop, MLflow-логирование, чекпойнт по best
  val_iou. make train MODEL=unet.
- FR-6: табличный baseline (sklearn RandomForest) попиксельно с подвыборкой.
  make train MODEL=baseline.
- FR-7: make eval -> reports/comparison_v1.md, метрики (IoU,F1,ROC-AUC,PR-AUC,
  Brier), bootstrap-CI 95% (1000 ресэмплов, OQ-6), pass/fail по M-2.
- Готовность Этапа 2: reports/comparison_v1.md существует; M-2 pass либо явный
  fail с планом.

### 6. Git
Локальный репозиторий удалён по просьбе пользователя. Пересоздать позже:
git init, .gitignore готов (игнорит .venv, data/raw, data/processed, app.db,
.env, .claude/, runs, exports, models, mlruns), коммитить data/manifest.yaml,
(Подсказать пользователю команды и когда и как комитить и пушить в удаленный репозиторий, чтоб он это сделал самостоятельно)

## Гочи окружения (из Этапа 1)
- data/ML-шаги НЕ параллелить — были гонки за файлы. Последовательно, ждать
  завершения.
- Верить только exit-кодам и содержимому файлов; вывод терминала искажался.
- PROJ_NETWORK=OFF уже выставляется в floodrisk/data/__init__.py.
- Этап 2 (обучение) — чистый torch/sklearn на хост-CPU, без GDAL-проблем.
