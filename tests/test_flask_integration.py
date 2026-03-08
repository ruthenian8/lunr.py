import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunr.integrations.flask import (
    _dialect_name_from_engine,
    build_or_rebuild_index,
    create_app,
    sql_lunr_index,
)
from lunr.storage.sql import SqlStorage


class _TrackingConnection:
    def __init__(self, conn):
        self._conn = conn
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        self.closed = True
        self._conn.close()


class _Engine:
    def __init__(self, db_path: Path, dialect_name="sqlite"):
        self._db_path = db_path
        self.dialect = SimpleNamespace(name=dialect_name)
        self._last = None

    def raw_connection(self):
        conn = sqlite3.connect(self._db_path)
        self._last = _TrackingConnection(conn)
        return self._last


class _DB:
    def __init__(self, db_path: Path, dialect_name="sqlite"):
        self.engine = _Engine(db_path, dialect_name=dialect_name)


def _refs(idx, query):
    return [result["ref"] for result in idx.search(query)]


def test_dialect_name_translation_for_postgresql():
    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    assert _dialect_name_from_engine(engine) == "postgres"


def test_dialect_name_falls_back_to_str_value():
    class UnknownDialect:
        def __str__(self):
            return "sqlite"

    engine = SimpleNamespace(dialect=UnknownDialect())
    assert _dialect_name_from_engine(engine) == "sqlite"


def test_build_or_rebuild_and_query_roundtrip(documents, tmp_path):
    db = _DB(tmp_path / "site.db")

    build_or_rebuild_index(
        db,
        "site_search",
        documents,
        ref_field="id",
        text_fields=("title", "body"),
        workers=2,
        parallel_backend="thread",
    )

    with sql_lunr_index(db, "site_search") as idx:
        refs = _refs(idx, "green study")

    assert refs == ["b", "a", "c"]
    assert db.engine._last.closed


def test_rebuild_replaces_existing_documents(documents, tmp_path):
    db = _DB(tmp_path / "site.db")

    build_or_rebuild_index(db, "site_search", documents)
    with sql_lunr_index(db, "site_search") as idx:
        assert _refs(idx, "mustard") == ["a"]

    replacement_docs = [
        {"id": "x", "title": "Fresh index", "body": "completely different content"}
    ]
    build_or_rebuild_index(db, "site_search", replacement_docs)

    with sql_lunr_index(db, "site_search") as idx:
        assert _refs(idx, "mustard") == []
        assert _refs(idx, "fresh") == ["x"]


def test_sql_lunr_index_returns_empty_results_for_blank_query(documents, tmp_path):
    db = _DB(tmp_path / "site.db")
    build_or_rebuild_index(db, "site_search", documents)

    with sql_lunr_index(db, "site_search") as idx:
        assert idx.search("") == []


def test_sql_lunr_index_handles_fresh_database_without_existing_tables(tmp_path):
    db = _DB(tmp_path / "fresh.db")

    with sql_lunr_index(db, "site_search") as idx:
        assert idx.search("anything") == []


def test_sql_lunr_index_discovers_fields_from_doc_fields_when_no_postings(tmp_path):
    db = _DB(tmp_path / "site.db")

    conn = db.engine.raw_connection()
    try:
        storage = SqlStorage.from_conn(conn, index_name="site_search", dialect="sqlite")
        storage.ensure_schema()
        conn.execute(
            "INSERT INTO lunr_doc_fields (index_name, field_ref, field, doc_ref, length) VALUES (?, ?, ?, ?, ?)",
            ("site_search", "title/doc-1", "title", "doc-1", 0),
        )
        conn.commit()
    finally:
        conn.close()

    with sql_lunr_index(db, "site_search") as idx:
        assert idx.fields == ["title"]


def test_build_or_rebuild_with_languages_none_roundtrip(documents, tmp_path):
    """Passing ``languages=None`` explicitly should behave like the default."""
    db = _DB(tmp_path / "site.db")

    build_or_rebuild_index(db, "site_search", documents, languages=None)

    with sql_lunr_index(db, "site_search", languages=None) as idx:
        refs = _refs(idx, "green")

    assert "b" in refs


def test_sql_lunr_index_accepts_languages_kwarg(documents, tmp_path):
    """``sql_lunr_index`` should accept a ``languages`` keyword argument."""
    db = _DB(tmp_path / "site.db")
    build_or_rebuild_index(db, "site_search", documents)

    with sql_lunr_index(db, "site_search", languages=None) as idx:
        refs = _refs(idx, "green")

    assert "b" in refs


def test_create_app_defers_imports_until_called(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name in {"flask", "flask_sqlalchemy"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ModuleNotFoundError):
        create_app()


def test_create_app_accepts_languages_parameter(monkeypatch):
    """``create_app`` should accept a ``languages`` keyword."""
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name in {"flask", "flask_sqlalchemy"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ModuleNotFoundError):
        create_app(languages=None)
