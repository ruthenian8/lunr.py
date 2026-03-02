import importlib.util
import os
import sqlite3
import uuid

import pytest

from lunr import get_default_builder, lunr
from lunr.exceptions import BaseLunrException
from lunr.query import QueryPresence
from lunr.storage.sql import SqlStorage


@pytest.fixture
def sql_storage():
    conn = sqlite3.connect(":memory:")
    return SqlStorage.from_conn(conn, index_name="test_idx", dialect="sqlite")


def _parse_mysql_user_and_db(raw_value):
    separators = (":", "/", ",")
    for separator in separators:
        if separator in raw_value:
            user, database = raw_value.split(separator, 1)
            user = user.strip()
            database = database.strip()
            if not user or not database:
                break
            return user, database

    value = raw_value.strip()
    if not value:
        raise ValueError("MAINDB must not be empty")
    return value, value


def _connect_mysql(host, user, password, database, port):
    if importlib.util.find_spec("pymysql") is not None:
        import pymysql

        return pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
        )

    if importlib.util.find_spec("MySQLdb") is not None:
        import MySQLdb

        return MySQLdb.connect(
            host=host,
            user=user,
            passwd=password,
            db=database,
            port=port,
        )

    raise ModuleNotFoundError("No MySQL Python driver installed (pymysql or MySQLdb)")


@pytest.fixture
def mysql_storage():
    user_and_db = os.getenv("MAINDB")
    password = os.getenv("PASSWDDB")

    if not user_and_db or password is None:
        pytest.skip("MySQL tests require MAINDB and PASSWDDB to be set")

    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    index_name = f"mysql_idx_{uuid.uuid4().hex}"

    try:
        user, database = _parse_mysql_user_and_db(user_and_db)
    except ValueError as exc:
        pytest.skip(f"Invalid MAINDB value: {exc}")

    try:
        conn = _connect_mysql(host, user, password, database, port)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))
    except Exception as exc:
        pytest.skip(f"Unable to connect to MySQL with local credentials: {exc}")

    storage = SqlStorage.from_conn(conn, index_name=index_name, dialect="mysql")

    cursor = None
    try:
        cursor = storage.conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
    except Exception as exc:
        pytest.skip(f"MySQL connection check failed: {exc}")
    finally:
        if cursor is not None:
            cursor.close()

    return storage


def _build_sql_index(documents, storage):
    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage)
    for document in documents:
        builder.add(document)
    return builder.build()


def test_sql_backend_matches_memory_for_positive_queries(documents, sql_storage):
    mem_idx = lunr(ref="id", fields=("title", "body"), documents=documents)
    sql_idx = _build_sql_index(documents, sql_storage)

    query = "green study"
    mem_refs = [result["ref"] for result in mem_idx.search(query)]
    sql_refs = [result["ref"] for result in sql_idx.search(query)]

    assert sql_refs == mem_refs


def test_sql_backend_wildcard_expansion(documents, sql_storage):
    idx = _build_sql_index(documents, sql_storage)

    starts_with = {result["ref"] for result in idx.search("pl*")}
    ends_with = {result["ref"] for result in idx.search("*reen")}

    assert starts_with == {"b", "c"}
    assert ends_with == {"a", "b", "c"}


def test_sql_backend_disables_prohibited_and_negated_queries(documents, sql_storage):
    idx = _build_sql_index(documents, sql_storage)

    query = idx.create_query()
    query.term("green", presence=QueryPresence.PROHIBITED)
    query.term("study", presence=QueryPresence.OPTIONAL)
    with pytest.raises(BaseLunrException, match="Prohibited clauses"):
        idx.query(query)

    with pytest.raises(BaseLunrException, match="Negated queries"):
        idx.search("-green")


def test_sql_backend_disables_edit_distance(documents, sql_storage):
    idx = _build_sql_index(documents, sql_storage)

    query = idx.create_query()
    query.term("gren", edit_distance=1)

    with pytest.raises(BaseLunrException, match="Edit distance"):
        idx.query(query)


def test_sql_backend_not_serializable(documents, sql_storage):
    idx = _build_sql_index(documents, sql_storage)

    with pytest.raises(BaseLunrException, match="cannot be serialized"):
        idx.serialize()


@pytest.mark.mysql
def test_mysql_backend_matches_memory_for_positive_queries(documents, mysql_storage):
    mem_idx = lunr(ref="id", fields=("title", "body"), documents=documents)
    sql_idx = _build_sql_index(documents, mysql_storage)

    query = "green study"
    mem_refs = [result["ref"] for result in mem_idx.search(query)]
    sql_refs = [result["ref"] for result in sql_idx.search(query)]

    assert sql_refs == mem_refs


@pytest.mark.mysql
def test_mysql_backend_wildcard_expansion(documents, mysql_storage):
    idx = _build_sql_index(documents, mysql_storage)

    starts_with = {result["ref"] for result in idx.search("pl*")}
    ends_with = {result["ref"] for result in idx.search("*reen")}

    assert starts_with == {"b", "c"}
    assert ends_with == {"a", "b", "c"}


@pytest.mark.mysql
def test_mysql_backend_disables_prohibited_and_negated_queries(documents, mysql_storage):
    idx = _build_sql_index(documents, mysql_storage)

    query = idx.create_query()
    query.term("green", presence=QueryPresence.PROHIBITED)
    query.term("study", presence=QueryPresence.OPTIONAL)
    with pytest.raises(BaseLunrException, match="Prohibited clauses"):
        idx.query(query)

    with pytest.raises(BaseLunrException, match="Negated queries"):
        idx.search("-green")


@pytest.mark.mysql
def test_mysql_backend_disables_edit_distance(documents, mysql_storage):
    idx = _build_sql_index(documents, mysql_storage)

    query = idx.create_query()
    query.term("gren", edit_distance=1)

    with pytest.raises(BaseLunrException, match="Edit distance"):
        idx.query(query)


@pytest.mark.mysql
def test_mysql_backend_not_serializable(documents, mysql_storage):
    idx = _build_sql_index(documents, mysql_storage)

    with pytest.raises(BaseLunrException, match="cannot be serialized"):
        idx.serialize()


@pytest.mark.mysql
def test_mysql_storage_uses_mysql_dialect(mysql_storage):
    assert mysql_storage.dialect.name == "mysql"
    assert mysql_storage.dialect.placeholder == "%s"
    assert mysql_storage.dialect.upsert == "duplicate"


def test_parse_mysql_user_and_db_supports_compound_values():
    assert _parse_mysql_user_and_db("user:db") == ("user", "db")
    assert _parse_mysql_user_and_db("user/db") == ("user", "db")
    assert _parse_mysql_user_and_db("user,db") == ("user", "db")


def test_parse_mysql_user_and_db_defaults_database_to_user():
    assert _parse_mysql_user_and_db("onlyvalue") == ("onlyvalue", "onlyvalue")


def test_parse_mysql_user_and_db_rejects_empty_value():
    with pytest.raises(ValueError, match="must not be empty"):
        _parse_mysql_user_and_db("   ")


def test_connect_mysql_requires_driver(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(ModuleNotFoundError, match="No MySQL Python driver"):
        _connect_mysql("127.0.0.1", "user", "pass", "db", 3306)
