# Floodrisk Makefile — единая точка входа для разработки.
# Работает в Git Bash / WSL / Linux / macOS. PowerShell — через `make` из Git Bash.

PYTHON ?= python
UV     ?= uv
PORT   ?= 8000

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Окружение:"
	@echo "  make venv          создать .venv через uv"
	@echo "  make install       поставить core-зависимости (requirements.lock)"
	@echo "  make install-data  поставить core + [data] (Этап 1, data-пайплайн)"
	@echo "  make install-ml    поставить core + [ml] (тренировка; отдельное окружение)"
	@echo "  make lock          перегенерировать requirements.lock и requirements-ml.lock"
	@echo ""
	@echo "БД:"
	@echo "  make db-reset      удалить app.db, пересоздать схему, сидинг"
	@echo "  make seed          залить scenarios + model_versions из конфигов"
	@echo ""
	@echo "Данные и ML (extras [ml]):"
	@echo "  make data          fetch + preprocess + label + manifest"
	@echo "  make train         тренировка U-Net и бейзлайна"
	@echo "  make eval          сравнительный отчёт U-Net vs baseline"
	@echo "  make verify-data   проверить контрольные суммы манифеста"
	@echo ""
	@echo "Сервис:"
	@echo "  make run           uvicorn с --reload"
	@echo "  make bench         latency-бенчмарк /api/predict (M-3/NFR-1)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  собрать образ floodrisk:latest"
	@echo "  make docker-run    запустить контейнер с volume-mount-ами"
	@echo ""
	@echo "Качество:"
	@echo "  make test          pytest"
	@echo "  make lint          ruff check + ruff format --check"
	@echo "  make fmt           ruff format"

# ──────────────── окружение ────────────────

.PHONY: venv
venv:
	$(UV) venv .venv --python 3.11

.PHONY: install
install:
	$(UV) pip sync requirements.lock

.PHONY: install-data
install-data:
	$(UV) pip sync requirements-data.lock

.PHONY: install-ml
install-ml:
	$(UV) pip sync requirements-ml.lock

.PHONY: lock
lock:
	$(UV) pip compile pyproject.toml --python-version 3.11 -o requirements.lock
	$(UV) pip compile pyproject.toml --python-version 3.11 --extra data -o requirements-data.lock
	$(UV) pip compile pyproject.toml --python-version 3.11 --extra ml -o requirements-ml.lock

# ──────────────── БД ────────────────

.PHONY: db-reset
db-reset:
	$(PYTHON) -c "from pathlib import Path; p=Path('app.db'); p.unlink(missing_ok=True); print('removed', p)"
	$(PYTHON) -m floodrisk.db.session create
	$(MAKE) seed

.PHONY: seed
seed:
	$(PYTHON) -m floodrisk.db.seed

# ──────────────── данные и ML ────────────────

.PHONY: data
data:
	$(PYTHON) -m floodrisk data fetch
	$(PYTHON) -m floodrisk data preprocess
	$(PYTHON) -m floodrisk data label
	$(PYTHON) -m floodrisk data manifest

.PHONY: train
train:
	$(PYTHON) -m floodrisk train --config configs/unet_v1.yaml
	$(PYTHON) -m floodrisk train --config configs/baseline_v1.yaml

.PHONY: eval
eval:
	$(PYTHON) -m floodrisk eval --models unet-v1,baseline-v1 --out reports/comparison_v1.md

.PHONY: verify-data
verify-data:
	$(PYTHON) -m floodrisk data verify --manifest data/manifest.yaml

# ──────────────── сервис ────────────────

.PHONY: run
run:
	$(PYTHON) -m uvicorn floodrisk.app:app --reload --host 127.0.0.1 --port $(PORT)

.PHONY: bench
bench:
	$(PYTHON) scripts/benchmark_latency.py

# ──────────────── Docker ────────────────

.PHONY: docker-build
docker-build:
	docker build -t floodrisk:latest .

.PHONY: docker-run
docker-run:
	docker run --rm -p $(PORT):8000 \
		-v "$(CURDIR)/runs:/app/runs" \
		-v "$(CURDIR)/exports:/app/exports" \
		-v "$(CURDIR)/models:/app/models" \
		-v "$(CURDIR)/data:/app/data" \
		-v "$(CURDIR)/app.db:/app/app.db" \
		floodrisk:latest

# ──────────────── качество ────────────────

.PHONY: test
test:
	$(PYTHON) -m pytest

.PHONY: lint
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

.PHONY: fmt
fmt:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .
