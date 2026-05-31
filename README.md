# floodrisk

Прототип ИНС-оценки подверженности затоплению. См. [docs/srs.md](docs/srs.md) — финальное ТЗ.

## Быстрый старт (dev)

Предусловия: Python 3.11, `uv` ([установка](https://docs.astral.sh/uv/getting-started/installation/)), `make` (Git Bash на Windows).

```bash
uv venv .venv --python 3.11
. .venv/Scripts/activate     # Windows (Git Bash); Linux/macOS: source .venv/bin/activate
make lock                    # сгенерировать requirements.lock и requirements-ml.lock
make install                 # core-зависимости
make db-reset                # создать app.db + засеять scenarios
make run                     # http://127.0.0.1:8000
```

`make help` — полный список команд.

## Структура

См. [docs/srs.md §4](docs/srs.md#4-структура-проекта).

## Статус

Этап 0 (каркас). Дальнейшие этапы — см. [docs/srs.md §17](docs/srs.md#17-план-реализации-этапы).
