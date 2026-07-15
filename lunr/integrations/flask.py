"""Helpers for integrating SQL-backed Lunr indexes into Flask applications."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator

from lunr import get_default_builder
from lunr.index import Index
from lunr.storage.sql.indexer import SqlIndexer
from lunr.storage.sql.reader import SqlIndexReader
from lunr.storage.sql.schema import ensure_schema, get_active_generation
from lunr.storage.sql import SqlFieldVectorsProxy, SqlInvertedIndexProxy, SqlStorage
from lunr.token_set import TokenSet


def _dialect_name_from_engine(engine: Any) -> str:
    """Return the SQL dialect string expected by :class:`SqlStorage`."""
    name = getattr(engine, "dialect", None)
    if name is not None:
        name = getattr(name, "name", name)

    if not isinstance(name, str):
        name = str(name)

    if name == "postgresql":
        return "postgres"

    return name


@contextmanager
def sql_lunr_index(
    db: Any, index_name: str, *, languages: "str | list[str] | None" = None
) -> Iterator[Index]:
    """Yield an SQL-backed Lunr index using a fresh connection from ``db.engine``.

    Args:
        db: A Flask-SQLAlchemy ``db`` instance (or any object whose ``engine``
            attribute exposes ``raw_connection()``).
        index_name: Logical name of the index inside the database.
        languages: Optional language(s) passed to
            :func:`~lunr.get_default_builder` so that the search pipeline
            uses the correct language-specific stemmer / stop-word filter.
    """
    conn = db.engine.raw_connection()
    try:
        dialect = _dialect_name_from_engine(db.engine)
        storage = SqlStorage.from_conn(conn, index_name=index_name, dialect=dialect)
        active = get_active_generation(conn, storage.dialect, index_name)
        ensure_schema(conn, storage.dialect)
        if active is None:
            yield Index(
                inverted_index={},
                field_vectors={},
                token_set=TokenSet(),
                fields=[],
                pipeline=get_default_builder(languages).search_pipeline,
            )
            return

        reader = SqlIndexReader(storage, active.generation)

        idx = Index(
            inverted_index=SqlInvertedIndexProxy(reader),
            field_vectors=SqlFieldVectorsProxy(reader),
            token_set=None,
            fields=active.fields,
            pipeline=get_default_builder(active.languages or None).search_pipeline,
            storage_reader=reader,
        )
        yield idx
    finally:
        conn.close()


def build_or_rebuild_index(
    db: Any,
    index_name: str,
    documents: Iterable[Dict[str, Any]],
    *,
    doc_batch_size: int = 500,
    row_batch_size: int = 5000,
    commit_docs: int = 2000,
    ref_field: str = "id",
    text_fields: Iterable[str] | None = None,
    metadata_whitelist: Iterable[str] | None = None,
    workers: int | None = None,
    parallel_backend: str | None = None,
    languages: "str | list[str] | None" = None,
) -> None:
    """Build or rebuild a SQL-backed Lunr index from ``documents``.

    Args:
        db: A Flask-SQLAlchemy ``db`` instance (or any object whose ``engine``
            attribute exposes ``raw_connection()``).
        index_name: Logical name of the index inside the database.
        documents: Iterable of document dicts to index.
        doc_batch_size: Documents per batch when flushing to SQL.
        row_batch_size: Rows per batch when flushing to SQL.
        commit_docs: Commit interval (number of documents).
        ref_field: Document key used as the reference field.
        text_fields: Fields to index; defaults to ``["title", "body"]``.
        metadata_whitelist: Additional metadata keys to store.
        workers: Number of parallel workers (requires SQL storage).
        parallel_backend: ``"thread"`` or ``"process"``.
        languages: Optional language(s) passed to
            :func:`~lunr.get_default_builder` so that the builder uses
            language-specific stemming / stop-word pipelines.
    """
    fields = list(text_fields) if text_fields is not None else ["title", "body"]

    conn = db.engine.raw_connection()
    try:
        dialect = _dialect_name_from_engine(db.engine)
        storage = SqlStorage.from_conn(conn, index_name=index_name, dialect=dialect)
        if storage.dialect.name == "sqlite":
            cursor = conn.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.fetchone()
            finally:
                cursor.close()
        builder = get_default_builder(languages)
        SqlIndexer(storage).build(
            documents,
            ref_field,
            [(field, 1, None) for field in fields],
            {"languages": languages},
            list(metadata_whitelist or ()),
            workers=workers,
            backend=parallel_backend or "process",
            batch_sizes={"rows": row_batch_size},
            search_pipeline=builder.search_pipeline,
        )
    finally:
        conn.close()


def serialize_search_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one Lunr result into a value accepted by Flask's JSON encoder."""
    match_data = result.get("match_data")
    return {
        "ref": result["ref"],
        "score": result["score"],
        "match_data": match_data.metadata if match_data is not None else {},
    }


def create_app(
    languages: "str | list[str] | None" = None,
    *,
    database_uri: str = "sqlite:///app.db",
    doc_batch_size: int = 500,
) -> "flask.Flask":  # type: ignore[name-defined]
    """Create a minimal Flask app exposing ``/search`` and ``/reindex`` endpoints.

    Args:
        languages: Optional language(s) forwarded to
            :func:`build_or_rebuild_index` and :func:`sql_lunr_index` so
            that the index pipelines use the correct language-specific
            stemmer / stop-word filter.
    """
    from flask import Flask, jsonify, request  # type: ignore
    from flask_sqlalchemy import SQLAlchemy  # type: ignore

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db = SQLAlchemy(app)

    index_name = "site_search_v1"

    class Document(db.Model):  # type: ignore[name-defined]
        """Simple database model used for reindexing."""

        __tablename__ = "documents"
        id = db.Column(db.Integer, primary_key=True)
        title = db.Column(db.Text, nullable=False, default="")
        body = db.Column(db.Text, nullable=False, default="")

    @app.get("/search")
    def search():  # type: ignore[no-redef]
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify([])

        with sql_lunr_index(db, index_name, languages=languages) as idx:
            results = idx.search(query)

        return jsonify([serialize_search_result(result) for result in results[:20]])

    @app.post("/reindex")
    def reindex():  # type: ignore[no-redef]
        """Repopulate the search index from the ``Document`` model."""
        indexed = 0

        def documents():
            nonlocal indexed
            for doc in Document.query.yield_per(doc_batch_size):
                indexed += 1
                yield {"id": str(doc.id), "title": doc.title, "body": doc.body}

        build_or_rebuild_index(
            db,
            index_name,
            documents(),
            doc_batch_size=doc_batch_size,
            ref_field="id",
            text_fields=["title", "body"],
            languages=languages,
        )
        return jsonify({"status": "ok", "indexed": indexed})

    return app
