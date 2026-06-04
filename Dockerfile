# syntax=docker/dockerfile:1.6
# Multi-stage build: app/inference image без ML extras (~800 MB).

# ────────── builder ──────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 \
    PIP_DEFAULT_TIMEOUT=120

RUN pip install --no-cache-dir --timeout 120 --retries 6 uv==0.4.*

WORKDIR /app

COPY pyproject.toml requirements.lock ./

# torch берём с CPU-индекса PyTorch (иначе на linux PyPI тянет CUDA-сборку ~2.5 ГБ и
# образ превышает лимит FR-19 ≤1 ГБ). uv с дефолтной стратегией first-index берёт каждый
# пакет из первого индекса, где он есть: torch* → CPU-индекс (2.12.0+cpu), остальное → PyPI.
# requirements.lock без хешей, поэтому переключение индекса безопасно.
RUN uv venv /opt/venv --python 3.11 \
    && uv pip install --python /opt/venv/bin/python -r requirements.lock \
       --index-url https://download.pytorch.org/whl/cpu \
       --extra-index-url https://pypi.org/simple

# ────────── runtime ──────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY floodrisk ./floodrisk
COPY configs ./configs
COPY static ./static

RUN mkdir -p /app/runs /app/exports /app/models /app/data

EXPOSE 8000

CMD ["uvicorn", "floodrisk.app:app", "--host", "0.0.0.0", "--port", "8000"]
