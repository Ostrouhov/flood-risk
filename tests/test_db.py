"""Проверяем, что схема создаётся и сидинг работает."""

import json

from sqlmodel import Session

from floodrisk.db.models import Scenario
from floodrisk.db.seed import seed_scenarios
from floodrisk.db.session import create_db_and_tables, engine


def test_schema_creation_idempotent() -> None:
    create_db_and_tables()
    create_db_and_tables()  # повторный вызов не должен падать


def test_seed_scenarios_from_yaml() -> None:
    create_db_and_tables()
    with Session(engine) as session:
        n = seed_scenarios(session)
        assert n >= 3, "ожидаются как минимум p1y / p10y / p100y из configs/scenarios.yaml"

        from sqlmodel import select

        rows = list(session.exec(select(Scenario)).all())
        ids = {r.scenario_id for r in rows}
        assert {"p1y", "p10y", "p100y"}.issubset(ids)

        # params_json должен парситься обратно в dict
        for r in rows:
            params = json.loads(r.params_json)
            assert isinstance(params, dict)
