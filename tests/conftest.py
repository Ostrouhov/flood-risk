"""Общие фикстуры. Используем in-memory SQLite, чтобы тесты не трогали app.db."""

import os

# До импорта приложения подменяем DATABASE_URL — настройки читают env при импорте.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from floodrisk.app import app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
