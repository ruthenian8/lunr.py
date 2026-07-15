import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import lunr as lunr_package
import lunr.integrations.flask as flask_integration
from lunr.integrations.flask import (
    _dialect_name_from_engine,
    build_or_rebuild_index,
    create_app,
    sql_lunr_index,
)
from lunr.storage.sql import SqlRebuildRequiredError


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


def test_rebuild_failure_preserves_searchable_index(tmp_path):
    db = _DB(tmp_path / "site.db")
    build_or_rebuild_index(
        db, "site", [{"id": "1", "title": "first", "body": "safe"}]
    )

    with pytest.raises(KeyError):
        build_or_rebuild_index(
            db, "site", [{"id": "2", "title": "missing body"}]
        )

    with sql_lunr_index(db, "site") as idx:
        assert _refs(idx, "first") == ["1"]


def test_reader_pinned_before_rebuild_remains_searchable(tmp_path):
    db = _DB(tmp_path / "site.db")
    build_or_rebuild_index(
        db, "site", [{"id": "1", "title": "old", "body": "generation"}]
    )

    with sql_lunr_index(db, "site") as pinned:
        build_or_rebuild_index(
            db, "site", [{"id": "2", "title": "new", "body": "generation"}]
        )
        assert _refs(pinned, "old") == ["1"]

    with sql_lunr_index(db, "site") as current:
        assert _refs(current, "new") == ["2"]


def test_sqlite_streaming_source_can_cross_index_flush_boundary(tmp_path):
    path = tmp_path / "site.db"
    source = sqlite3.connect(path)
    source.execute("CREATE TABLE source_docs (id TEXT, title TEXT, body TEXT)")
    source.executemany(
        "INSERT INTO source_docs VALUES (?, ?, ?)",
        [(str(index), f"title {index}", "streamed body") for index in range(12)],
    )
    source.commit()
    db = _DB(path)

    def documents():
        cursor = source.cursor()
        try:
            cursor.execute("SELECT id, title, body FROM source_docs ORDER BY id")
            while True:
                rows = cursor.fetchmany(2)
                if not rows:
                    break
                for doc_ref, title, body in rows:
                    yield {"id": doc_ref, "title": title, "body": body}
        finally:
            cursor.close()

    stream = documents()
    try:
        build_or_rebuild_index(db, "site", stream, row_batch_size=3)
        assert source.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        stream.close()
        source.close()

    with sql_lunr_index(db, "site") as idx:
        assert len(idx.search("streamed")) == 12


def test_rebuilding_one_index_does_not_delete_another(tmp_path):
    db = _DB(tmp_path / "site.db")
    build_or_rebuild_index(
        db, "a", [{"id": "a1", "title": "old", "body": "replace me"}]
    )
    build_or_rebuild_index(
        db, "b", [{"id": "b1", "title": "stable", "body": "unchanged"}]
    )

    build_or_rebuild_index(
        db, "a", [{"id": "a2", "title": "new", "body": "replacement"}]
    )

    with sql_lunr_index(db, "b") as idx:
        assert _refs(idx, "unchanged") == ["b1"]


def test_sql_lunr_index_returns_empty_results_for_blank_query(documents, tmp_path):
    db = _DB(tmp_path / "site.db")
    build_or_rebuild_index(db, "site_search", documents)

    with sql_lunr_index(db, "site_search") as idx:
        assert idx.search("") == []


def test_sql_lunr_index_handles_fresh_database_without_existing_tables(tmp_path):
    db = _DB(tmp_path / "fresh.db")

    with sql_lunr_index(db, "site_search") as idx:
        assert idx.search("anything") == []


def test_sql_lunr_index_rejects_v1_only_database(tmp_path):
    db = _DB(tmp_path / "legacy.db")
    conn = sqlite3.connect(tmp_path / "legacy.db")
    try:
        conn.execute("CREATE TABLE lunr_terms (marker INTEGER)")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(SqlRebuildRequiredError, match="rebuild"):
        with sql_lunr_index(db, "site"):
            pass


def test_sql_lunr_index_loads_stored_fields_for_empty_index(tmp_path):
    db = _DB(tmp_path / "site.db")
    build_or_rebuild_index(db, "site_search", [], text_fields=["title"])

    with sql_lunr_index(db, "site_search") as idx:
        assert idx.fields == ["title"]


def test_sql_lunr_index_uses_stored_languages(monkeypatch, documents, tmp_path):
    db = _DB(tmp_path / "site.db")
    build_or_rebuild_index(db, "site_search", documents)
    conn = sqlite3.connect(tmp_path / "site.db")
    try:
        conn.execute(
            "UPDATE lunr_v2_generations SET languages = ? WHERE index_name = ?",
            (json.dumps(["stored-language"]), "site_search"),
        )
        conn.commit()
    finally:
        conn.close()

    requested = []
    real_get_default_builder = lunr_package.get_default_builder

    def recording_builder(languages):
        requested.append(languages)
        return real_get_default_builder(None)

    monkeypatch.setattr(lunr_package, "get_default_builder", recording_builder)
    with sql_lunr_index(db, "site_search"):
        pass

    assert requested == [["stored-language"]]


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


def test_search_endpoint_returns_json_result_and_streamed_count(tmp_path):
    pytest.importorskip("flask")
    pytest.importorskip("flask_sqlalchemy")
    app = create_app(database_uri=f"sqlite:///{tmp_path / 'app.db'}")
    db = app.extensions["sqlalchemy"]
    with app.app_context():
        db.create_all()
        document_model = next(
            mapper.class_
            for mapper in db.Model._sa_registry.mappers
            if mapper.class_.__tablename__ == "documents"
        )
        db.session.add_all(
            [
                document_model(title="Green", body="first study"),
                document_model(title="Blue", body="second study"),
            ]
        )
        db.session.commit()

    client = app.test_client()
    rebuild = client.post("/reindex")
    response = client.get("/search?q=green")

    assert rebuild.status_code == 200
    assert rebuild.get_json() == {"status": "ok", "indexed": 2}
    assert response.status_code == 200
    result = response.get_json()[0]
    assert result.keys() == {"ref", "score", "match_data"}
    assert isinstance(result["match_data"], dict)


def test_serialize_search_result_is_json_safe(index, documents):
    result = index.search("green")[0]

    serialized = flask_integration.serialize_search_result(result)

    assert json.loads(json.dumps(serialized))["match_data"] == serialized["match_data"]
