# syntax=docker/dockerfile:1.6
# Multi-stage build: app/inference image без ML extras (~800 MB).

# ────────── builder ──────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never

RUN pip install --no-cache-dir uv==0.4.*

WORKDIR /app

COPY pyproject.toml requirements.lock ./

RUN uv venv /opt/venv --python 3.11 \
    && uv pip sync --python /opt/venv/bin/python requirements.lock

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
