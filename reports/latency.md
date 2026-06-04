# Latency-бенчмарк /api/predict (M-3 / NFR-1)

Замер: 2026-05-31 18:39:59Z. Хост — CPU (Ryzen 5 2600, 6 ядер), Windows, `.venv` (torch CPU).
Метод: 20 последовательных `POST /api/predict` через FastAPI TestClient
(in-process, резидентная модель — FR-13), warmup-запрос в замер не входит.

- bbox: `[100.64, 54.54, 100.89, 54.71]` (зона покрытия, Тулун р. Ия)
- scenario_id: `p10y`, model_version: `unet-v1`

| Метрика | Значение, с |
|---|---|
| min  | 1.565 |
| p50  | 1.616 |
| mean | 1.632 |
| **p95** | **1.740** |
| max  | 1.744 |

**Цель NFR-1:** p95 ≤ 5 с на CPU → **PASS**.

> NFR-3 (`/api/explain`, ≤15 с): реальный explain на U-Net ~5.1 с (Этап 5), с запасом.
