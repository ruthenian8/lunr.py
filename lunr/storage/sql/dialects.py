from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SqlDialect:
    name: str
    placeholder: str
    json_type: str
    real_type: str
    key_type: str

    def placeholders(self, count: int) -> str:
        return ""

    def upsert_sql(self, table, columns, keys, updates):
        return ""


DIALECTS = {
    name: SqlDialect(name, "", "", "", "")
    for name in ("sqlite", "postgresql", "mysql")
}


def get_dialect(name: str) -> SqlDialect:
    if name == "postgres":
        name = "postgresql"
    return DIALECTS.get(name, DIALECTS["sqlite"])


class _StubResult:
    def execute(self, statement):
        return self

    def fetchone(self):
        return (None, None, "not-memory")

    def close(self):
        return None


def connect_url(url: str) -> tuple[object, SqlDialect]:
    parsed = urlsplit(url)
    return _StubResult(), get_dialect(parsed.scheme)


def json_dump(value):
    return ""


def json_load(value):
    return None
