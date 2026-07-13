import sqlite3

import pytest

from lunr import lunr
from lunr.storage.sql import SqlRebuildRequiredError, SqlStorage


@pytest.fixture
def v1_connection():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE lunr_terms ("
        "index_name TEXT NOT NULL, term TEXT NOT NULL, term_index INTEGER NOT NULL)"
    )
    try:
        yield connection
    finally:
        connection.close()


def test_storage_context_closes_only_owned_connection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with SqlStorage.from_url("sqlite:///owned.db", "docs") as owned:
        conn = owned.conn

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")

    external = sqlite3.connect(":memory:")
    try:
        with SqlStorage.from_conn(external, "docs"):
            pass
        assert external.execute("SELECT 1").fetchone() == (1,)
    finally:
        external.close()


def test_v1_only_database_requires_rebuild(v1_connection):
    storage = SqlStorage.from_conn(v1_connection, "docs")

    with pytest.raises(SqlRebuildRequiredError, match="rebuild"):
        storage.reader()


def test_temporary_legacy_writer_reader_lifecycle_is_scoped_to_one_facade():
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "docs")
    try:
        index = lunr(
            ref="id",
            fields=("title",),
            documents=[{"id": "1", "title": "green plant"}],
            storage=storage,
        )

        assert [result["ref"] for result in index.search("green")] == ["1"]

        reopened = SqlStorage.from_conn(connection, "docs")
        with pytest.raises(SqlRebuildRequiredError, match="rebuild"):
            reopened.reader()
    finally:
        connection.close()
