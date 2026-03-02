"""Helpers for integrating SQL-backed Lunr indexes into Flask applications."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable

from lunr import get_default_builder
from lunr.index import Index
from lunr.storage.sql import SqlFieldVectorsProxy, SqlInvertedIndexProxy, SqlStorage


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
def sql_lunr_index(db: Any, index_name: str) -> Index:
    """Yield an SQL-backed Lunr index using a fresh connection from ``db.engine``."""
    conn = db.engine.raw_connection()
    try:
        dialect = _dialect_name_from_engine(db.engine)
        storage = SqlStorage.from_conn(conn, index_name=index_name, dialect=dialect)
        storage.ensure_schema()
        reader = storage.reader()

        placeholder = storage.dialect.placeholder
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"SELECT DISTINCT field FROM lunr_postings WHERE index_name = {placeholder}",
                (index_name,),
            )
            rows = cursor.fetchall()
            if not rows:
                cursor.execute(
                    f"SELECT DISTINCT field FROM lunr_doc_fields WHERE index_name = {placeholder}",
                    (index_name,),
                )
                rows = cursor.fetchall()

            fields = [row[0] for row in rows]
        finally:
            cursor.close()

        idx = Index(
            inverted_index=SqlInvertedIndexProxy(reader),
            field_vectors=SqlFieldVectorsProxy(reader),
            token_set=None,
            fields=fields,
            pipeline=get_default_builder().search_pipeline,
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
) -> None:
    """Build or rebuild a SQL-backed Lunr index from ``documents``."""
    fields = list(text_fields) if text_fields is not None else ["title", "body"]

    conn = db.engine.raw_connection()
    try:
        dialect = _dialect_name_from_engine(db.engine)
        storage = SqlStorage.from_conn(conn, index_name=index_name, dialect=dialect)
        storage.ensure_schema()
        placeholder = storage.dialect.placeholder

        cursor = conn.cursor()
        try:
            for table in (
                "lunr_terms",
                "lunr_postings",
                "lunr_field_vectors",
                "lunr_doc_fields",
                "lunr_term_frequencies",
            ):
                cursor.execute(
                    f"DELETE FROM {table} WHERE index_name = {placeholder}",
                    (index_name,),
                )
            conn.commit()
        finally:
            cursor.close()

        builder = get_default_builder()
        builder.ref(ref_field)
        for field in fields:
            builder.field(field)
        if metadata_whitelist:
            builder.metadata_whitelist.extend(metadata_whitelist)
        builder.storage(storage)

        builder.sql_flush(
            enabled=True,
            doc_batch_size=doc_batch_size,
            row_batch_size=row_batch_size,
        )
        builder.sql_commit_every(docs=commit_docs)

        if workers and workers > 1:
            builder.parallel(workers=workers, backend=parallel_backend or "thread")

        for doc in documents:
            builder.add(doc)

        builder.build()
    finally:
        conn.close()


def create_app() -> "flask.Flask":  # type: ignore[name-defined]
    """Create a minimal Flask app exposing a ``/search`` endpoint."""
    from flask import Flask, jsonify, request  # type: ignore
    from flask_sqlalchemy import SQLAlchemy  # type: ignore

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db = SQLAlchemy(app)

    index_name = "site_search_v1"

    @app.get("/search")
    def search():  # type: ignore[no-redef]
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify([])

        with sql_lunr_index(db, index_name) as idx:
            results = idx.search(query)

        return jsonify(results[:20])

    return app
