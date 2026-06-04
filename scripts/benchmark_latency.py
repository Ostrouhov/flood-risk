"""Бенчмарк latency POST /api/predict (M-3 / NFR-1).

20 последовательных запросов на фиксированном bbox через FastAPI TestClient
(in-process): модель резидентная (floodrisk.inference.registry кеширует загрузку),
поэтому замер отражает FR-13 — стабильную latency без «холодного старта».

Запуск:  .venv\\Scripts\\python.exe scripts\\benchmark_latency.py
Результат печатается и пишется в reports/latency.md. Цель NFR-1: p95 ≤ 5 с на CPU.
"""

from __future__ import annotations

import shutil
import statistics
import time
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from floodrisk.app import app
from floodrisk.settings import settings

# Фиксированный bbox в зоне покрытия (Тулун, р. Ия) и параметры запроса.
BBOX = [100.64, 54.54, 100.89, 54.71]
SCENARIO_ID = "p10y"
MODEL_VERSION = "unet-v1"
N_REQUESTS = 20
P95_TARGET_S = 5.0


def _percentile(samples: list[float], pct: float) -> float:
    """Перцентиль по методу nearest-rank (без внешних зависимостей)."""
    ordered = sorted(samples)
    k = max(1, -(-len(ordered) * pct // 100))  # ceil(n*pct/100), >=1
    return ordered[int(k) - 1]


def main() -> int:
    payload = {"bbox": BBOX, "scenario_id": SCENARIO_ID, "model_version": MODEL_VERSION}
    created_runs: list[str] = []

    with TestClient(app) as client:
        # warmup — загрузка резидентной модели, в замер НЕ входит.
        warm = client.post("/api/predict", json=payload)
        if warm.status_code != 200:
            print(f"warmup failed: {warm.status_code} {warm.text}")
            return 1
        created_runs.append(warm.json()["run_id"])

        latencies_s: list[float] = []
        for i in range(N_REQUESTS):
            t0 = time.perf_counter()
            resp = client.post("/api/predict", json=payload)
            dt = time.perf_counter() - t0
            if resp.status_code != 200:
                print(f"request {i} failed: {resp.status_code} {resp.text}")
                return 1
            latencies_s.append(dt)
            created_runs.append(resp.json()["run_id"])

    # уборка артефактов бенчмарка
    for rid in created_runs:
        shutil.rmtree(settings.runs_dir / rid, ignore_errors=True)

    p50 = statistics.median(latencies_s)
    p95 = _percentile(latencies_s, 95)
    mean = statistics.mean(latencies_s)
    lo, hi = min(latencies_s), max(latencies_s)
    verdict = "PASS" if p95 <= P95_TARGET_S else "FAIL"

    summary = (
        f"min={lo:.3f}s  p50={p50:.3f}s  mean={mean:.3f}s  "
        f"p95={p95:.3f}s  max={hi:.3f}s  ->  M-3/NFR-1 ({P95_TARGET_S:.0f}s): {verdict}"
    )
    print(summary)

    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    report = f"""# Latency-бенчмарк /api/predict (M-3 / NFR-1)

Замер: {ts}. Хост — CPU (Ryzen 5 2600, 6 ядер), Windows, `.venv` (torch CPU).
Метод: {N_REQUESTS} последовательных `POST /api/predict` через FastAPI TestClient
(in-process, резидентная модель — FR-13), warmup-запрос в замер не входит.

- bbox: `{BBOX}` (зона покрытия, Тулун р. Ия)
- scenario_id: `{SCENARIO_ID}`, model_version: `{MODEL_VERSION}`

| Метрика | Значение, с |
|---|---|
| min  | {lo:.3f} |
| p50  | {p50:.3f} |
| mean | {mean:.3f} |
| **p95** | **{p95:.3f}** |
| max  | {hi:.3f} |

**Цель NFR-1:** p95 ≤ {P95_TARGET_S:.0f} с на CPU → **{verdict}**.

> NFR-3 (`/api/explain`, ≤15 с): реальный explain на U-Net ~5.1 с (Этап 5), с запасом.
"""
    out = settings.project_root / "reports" / "latency.md"
    out.write_text(report, encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
