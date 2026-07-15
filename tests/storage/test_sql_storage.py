import sqlite3

import pytest

from lunr import lunr
from lunr.storage.sql import (
    SqlIndexReader,
    SqlIndexWriter,
    SqlRebuildRequiredError,
    SqlStorage,
)
from lunr.storage.sql.reader import SqlIndexReader as V2SqlIndexReader
from lunr.storage.sql.writer import SqlIndexWriter as V2SqlIndexWriter


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


def test_storage_close_closes_an_owned_connection_directly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = SqlStorage.from_url("sqlite:///owned.db", "docs")
    connection = storage.conn

    storage.close()

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_v1_only_database_requires_rebuild(v1_connection):
    storage = SqlStorage.from_conn(v1_connection, "docs")

    with pytest.raises(SqlRebuildRequiredError, match="rebuild"):
        storage.reader()


def test_public_reader_writer_imports_alias_v2_implementations():
    assert SqlIndexReader is V2SqlIndexReader
    assert SqlIndexWriter is V2SqlIndexWriter


def test_writer_requires_an_explicit_v2_generation():
    storage = SqlStorage.from_conn(sqlite3.connect(":memory:"), "docs")
    try:
        with pytest.raises(TypeError, match="generation"):
            storage.writer()
    finally:
        storage.conn.close()


def test_prune_inactive_generations_is_explicit_and_index_scoped():
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "docs")
    try:
        old = lunr(
            "id", ("body",), [{"id": "old", "body": "retained"}], storage=storage
        )
        old_generation = old.storage_reader.generation
        lunr("id", ("body",), [{"id": "new", "body": "active"}], storage=storage)
        other = SqlStorage.from_conn(connection, "other")
        lunr("id", ("body",), [{"id": "other", "body": "safe"}], storage=other)

        assert storage.prune_inactive_generations() == [old_generation]
        assert connection.execute(
            "SELECT COUNT(*) FROM lunr_v2_generations "
            "WHERE index_name=? AND generation=?",
            ("docs", old_generation),
        ).fetchone() == (0,)
        assert other.open_index().search("safe")[0]["ref"] == "other"
    finally:
        connection.close()
