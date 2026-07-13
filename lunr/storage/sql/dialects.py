from __future__ import annotations

import json
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
        return ", ".join([self.placeholder] * count)

    def upsert_sql(self, table, columns, keys, updates):
        values = self.placeholders(len(columns))
        base = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values})"
        if self.name == "sqlite":
            return base + f" ON CONFLICT ({', '.join(keys)}) DO UPDATE SET " + ", ".join(
                f"{column}=excluded.{column}" for column in updates
            )
        if self.name == "postgresql":
            return base + f" ON CONFLICT ({', '.join(keys)}) DO UPDATE SET " + ", ".join(
                f"{column}=EXCLUDED.{column}" for column in updates
            )
        return base + " ON DUPLICATE KEY UPDATE " + ", ".join(
            f"{column}=VALUES({column})" for column in updates
        )


DIALECTS = {
    "sqlite": SqlDialect("sqlite", "?", "TEXT", "REAL", "TEXT"),
    "postgresql": SqlDialect("postgresql", "%s", "JSON", "REAL", "TEXT"),
    "mysql": SqlDialect("mysql", "%s", "JSON", "DOUBLE", "VARCHAR(191)"),
}


def get_dialect(name: str) -> SqlDialect:
    if name == "postgres":
        name = "postgresql"
    try:
        return DIALECTS[name]
    except KeyError:
        raise ValueError(f"Unsupported SQL dialect: {name}") from None


def connect_url(url: str) -> tuple[object, SqlDialect]:
    parsed = urlsplit(url)
    dialect = get_dialect(parsed.scheme)

    if dialect.name == "sqlite":
        import sqlite3

        path = parsed.path[1:] if parsed.path.startswith("/") else parsed.path
        return sqlite3.connect(path or ":memory:"), dialect

    if dialect.name == "postgresql":
        try:
            import psycopg

            return psycopg.connect(url), dialect
        except ImportError:
            import psycopg2

            return psycopg2.connect(url), dialect

    try:
        import pymysql

        connection = pymysql.connect(
            host=parsed.hostname,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip("/"),
            port=parsed.port or 3306,
        )
    except ImportError:
        import MySQLdb

        connection = MySQLdb.connect(
            host=parsed.hostname,
            user=parsed.username,
            passwd=parsed.password,
            db=parsed.path.lstrip("/"),
            port=parsed.port or 3306,
        )
    return connection, dialect


def json_dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_load(value):
    if value is None or isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)
