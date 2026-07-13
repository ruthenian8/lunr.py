import sys
from types import SimpleNamespace

import pytest

from lunr.storage.sql.dialects import connect_url, get_dialect, json_dump, json_load


def test_get_dialect_supports_postgres_alias_and_rejects_unknown_names():
    assert get_dialect("postgres") is get_dialect("postgresql")
    assert get_dialect("sqlite").name == "sqlite"
    assert get_dialect("mysql").name == "mysql"

    with pytest.raises(ValueError, match="Unsupported SQL dialect: oracle"):
        get_dialect("oracle")


@pytest.mark.parametrize(
    "lookup, expected",
    [
        ("sqlite", ("sqlite", "?", "TEXT", "REAL", "TEXT")),
        ("postgresql", ("postgresql", "%s", "JSON", "REAL", "TEXT")),
        ("mysql", ("mysql", "%s", "JSON", "DOUBLE", "VARCHAR(191)")),
    ],
)
def test_dialect_metadata(lookup, expected):
    dialect = get_dialect(lookup)
    assert (
        dialect.name,
        dialect.placeholder,
        dialect.json_type,
        dialect.real_type,
        dialect.key_type,
    ) == expected


@pytest.mark.parametrize(
    "name, begin_write_sql, lock_suffix",
    [
        ("sqlite", "BEGIN IMMEDIATE", ""),
        ("postgresql", None, " FOR UPDATE"),
        ("mysql", None, " FOR UPDATE"),
    ],
)
def test_dialect_logical_index_locking(name, begin_write_sql, lock_suffix):
    dialect = get_dialect(name)
    assert dialect.begin_write_sql == begin_write_sql
    assert dialect.row_lock_suffix == lock_suffix


class SyntheticDatabaseError(Exception):
    def __init__(self, *args, sqlstate=None, pgcode=None):
        super().__init__(*args)
        self.sqlstate = sqlstate
        self.pgcode = pgcode


@pytest.mark.parametrize(
    "name, error, expected",
    [
        ("sqlite", Exception("no such table: docs"), True),
        ("sqlite", Exception("not authorized"), False),
        (
            "postgresql",
            SyntheticDatabaseError("undefined", sqlstate="42P01"),
            True,
        ),
        (
            "postgresql",
            SyntheticDatabaseError("denied", pgcode="42501"),
            False,
        ),
        ("mysql", SyntheticDatabaseError(1146, "table missing"), True),
        ("mysql", SyntheticDatabaseError(1142, "permission denied"), False),
    ],
)
def test_dialect_classifies_only_undefined_table_errors(name, error, expected):
    assert get_dialect(name).is_undefined_table_error(error) is expected


@pytest.mark.parametrize(
    "name, count, expected",
    [
        ("sqlite", 3, "?, ?, ?"),
        ("postgresql", 2, "%s, %s"),
        ("mysql", 1, "%s"),
    ],
)
def test_dialect_placeholders(name, count, expected):
    assert get_dialect(name).placeholders(count) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "sqlite",
            "INSERT INTO docs (id, body) VALUES (?, ?) "
            "ON CONFLICT (id) DO UPDATE SET body=excluded.body",
        ),
        (
            "postgresql",
            "INSERT INTO docs (id, body) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET body=EXCLUDED.body",
        ),
        (
            "mysql",
            "INSERT INTO docs (id, body) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE body=VALUES(body)",
        ),
    ],
)
def test_dialect_upsert_sql(name, expected):
    assert (
        get_dialect(name).upsert_sql("docs", ["id", "body"], ["id"], ["body"])
        == expected
    )


def test_sqlite_url_forms(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    relative, dialect = connect_url("sqlite:///relative.db")
    assert dialect.name == "sqlite"
    relative.execute("CREATE TABLE marker (id INTEGER)")
    relative.close()
    assert (tmp_path / "relative.db").exists()

    memory, _ = connect_url("sqlite:///:memory:")
    assert memory.execute("PRAGMA database_list").fetchone()[2] == ""
    memory.close()

    absolute_path = tmp_path / "absolute.db"
    absolute, _ = connect_url(f"sqlite:////{str(absolute_path).lstrip('/')}")
    absolute.close()
    assert absolute_path.exists()


def test_postgresql_url_prefers_psycopg(monkeypatch):
    connection = object()
    calls = []
    psycopg = SimpleNamespace(connect=lambda url: calls.append(url) or connection)
    psycopg2 = SimpleNamespace(
        connect=lambda url: pytest.fail("psycopg2 fallback should not be used")
    )
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)

    actual, dialect = connect_url("postgresql://user:pass@db.example/lunr")

    assert actual is connection
    assert dialect.name == "postgresql"
    assert calls == ["postgresql://user:pass@db.example/lunr"]


def test_postgresql_url_falls_back_to_psycopg2(monkeypatch):
    connection = object()
    calls = []
    psycopg2 = SimpleNamespace(connect=lambda url: calls.append(url) or connection)
    monkeypatch.setitem(sys.modules, "psycopg", None)
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)

    actual, dialect = connect_url("postgres://user:pass@db.example/lunr")

    assert actual is connection
    assert dialect.name == "postgresql"
    assert calls == ["postgres://user:pass@db.example/lunr"]


def test_mysql_url_prefers_pymysql(monkeypatch):
    connection = object()
    calls = []
    pymysql = SimpleNamespace(
        connect=lambda **kwargs: calls.append(kwargs) or connection
    )
    mysql_db = SimpleNamespace(
        connect=lambda **kwargs: pytest.fail("MySQLdb fallback should not be used")
    )
    monkeypatch.setitem(sys.modules, "pymysql", pymysql)
    monkeypatch.setitem(sys.modules, "MySQLdb", mysql_db)

    actual, dialect = connect_url("mysql://user:pass@db.example:3307/lunr")

    assert actual is connection
    assert dialect.name == "mysql"
    assert calls == [
        {
            "host": "db.example",
            "user": "user",
            "password": "pass",
            "database": "lunr",
            "port": 3307,
        }
    ]


def test_mysql_url_falls_back_to_mysqldb(monkeypatch):
    connection = object()
    calls = []
    mysql_db = SimpleNamespace(
        connect=lambda **kwargs: calls.append(kwargs) or connection
    )
    monkeypatch.setitem(sys.modules, "pymysql", None)
    monkeypatch.setitem(sys.modules, "MySQLdb", mysql_db)

    actual, dialect = connect_url("mysql://user:pass@db.example/lunr")

    assert actual is connection
    assert dialect.name == "mysql"
    assert calls == [
        {
            "host": "db.example",
            "user": "user",
            "passwd": "pass",
            "db": "lunr",
            "port": 3306,
        }
    ]


def test_json_dump_is_deterministic_and_compact():
    assert json_dump({"b": 2, "a": 1}) == '{"a":1,"b":2}'


@pytest.mark.parametrize(
    "value, expected",
    [
        ('{"a": 1}', {"a": 1}),
        (b"[1, 2]", [1, 2]),
        ({"a": 1}, {"a": 1}),
        ([1, 2], [1, 2]),
        (None, None),
    ],
)
def test_json_load_accepts_driver_native_and_encoded_values(value, expected):
    assert json_load(value) == expected


@pytest.mark.parametrize("value", [1, 1.5, True, False])
def test_json_load_preserves_driver_native_numeric_values(value):
    assert json_load(value) is value
