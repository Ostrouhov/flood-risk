"""`make seed`: наполняет scenarios из YAML, регистрирует model_versions из models/.

Идемпотентен: повторный запуск обновляет существующие записи, не дублирует.
"""

import json
from pathlib import Path

import yaml
from sqlmodel import Session, select

from floodrisk.db.models import ModelVersion, Scenario
from floodrisk.db.session import create_db_and_tables, engine
from floodrisk.settings import settings


def seed_scenarios(session: Session) -> int:
    path = settings.configs_dir / "scenarios.yaml"
    if not path.exists():
        print(f"[seed] {path} not found, skipping scenarios")
        return 0

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenarios = data.get("scenarios", [])
    n = 0
    for entry in scenarios:
        scenario_id = entry["scenario_id"]
        existing = session.exec(select(Scenario).where(Scenario.scenario_id == scenario_id)).first()
        params_json = json.dumps(entry.get("params", {}), ensure_ascii=False)
        if existing:
            existing.name = entry["name"]
            existing.description = entry.get("description")
            existing.params_json = params_json
        else:
            session.add(
                Scenario(
                    scenario_id=scenario_id,
                    name=entry["name"],
                    description=entry.get("description"),
                    params_json=params_json,
                )
            )
        n += 1
    session.commit()
    return n


def seed_model_versions(session: Session) -> int:
    """Регистрирует обученные модели из models/<name>/<version>/.

    На Этапе 0 моделей ещё нет — функция просто ничего не делает.
    """
    models_dir: Path = settings.models_dir
    if not models_dir.exists():
        return 0

    n = 0
    for name_dir in models_dir.iterdir():
        if not name_dir.is_dir():
            continue
        name = name_dir.name
        for version_dir in name_dir.iterdir():
            if not version_dir.is_dir():
                continue
            version = version_dir.name
            ckpt_candidates = list(version_dir.glob("*.ckpt")) + list(version_dir.glob("*.pkl"))
            if not ckpt_candidates:
                continue
            checkpoint_path = str(ckpt_candidates[0])
            config_path = str(version_dir / "config.yaml")
            metrics_path = version_dir / "metrics.json"
            metrics_json = (
                metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else None
            )

            existing = session.exec(
                select(ModelVersion).where(
                    ModelVersion.name == name, ModelVersion.version == version
                )
            ).first()
            if existing:
                existing.checkpoint_path = checkpoint_path
                existing.config_path = config_path
                existing.metrics_json = metrics_json
            else:
                session.add(
                    ModelVersion(
                        name=name,
                        version=version,
                        checkpoint_path=checkpoint_path,
                        config_path=config_path,
                        metrics_json=metrics_json,
                        dataset_version="v1",
                    )
                )
            n += 1
    session.commit()
    return n


def main() -> None:
    create_db_and_tables()
    with Session(engine) as session:
        s = seed_scenarios(session)
        m = seed_model_versions(session)
    print(f"[seed] scenarios upserted: {s}, model_versions upserted: {m}")


if __name__ == "__main__":
    main()
