"""Engine, get_session(), create_db_and_tables(). WAL + foreign_keys для SQLite."""

from collections.abc import Iterator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

# Импорт моделей нужен, чтобы SQLModel.metadata знала о таблицах перед create_all().
from floodrisk.db import models  # noqa: F401
from floodrisk.settings import settings

_is_sqlite = settings.database_url.startswith("sqlite")
_is_memory = _is_sqlite and ":memory:" in settings.database_url

_engine_kwargs: dict = {"echo": False}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
if _is_memory:
    # in-memory SQLite: один shared-коннект, иначе каждая сессия видит свою пустую БД.
    _engine_kwargs["poolclass"] = StaticPool

engine = create_engine(settings.database_url, **_engine_kwargs)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _connection_record) -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def main() -> None:
    """Entry point для `python -m floodrisk.db.session create`."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create":
        create_db_and_tables()
        print(f"schema created in {settings.database_url}")
    else:
        print("usage: python -m floodrisk.db.session create")
        sys.exit(2)


if __name__ == "__main__":
    main()
