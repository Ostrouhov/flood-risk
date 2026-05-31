"""Тонкие репозитории поверх SQLModel.Session. Наполняются по мере необходимости."""

from sqlmodel import Session, select

from floodrisk.db.models import ModelVersion, Run, Scenario


def list_scenarios(session: Session) -> list[Scenario]:
    return list(session.exec(select(Scenario).order_by(Scenario.scenario_id)).all())


def get_scenario(session: Session, scenario_id: str) -> Scenario | None:
    return session.exec(select(Scenario).where(Scenario.scenario_id == scenario_id)).first()


def list_model_versions(session: Session) -> list[ModelVersion]:
    return list(session.exec(select(ModelVersion)).all())


def get_model_version(session: Session, name: str, version: str) -> ModelVersion | None:
    stmt = select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
    return session.exec(stmt).first()


def get_run(session: Session, run_id: str) -> Run | None:
    return session.get(Run, run_id)
