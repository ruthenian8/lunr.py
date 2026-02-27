"""SQL storage backend for lunr.py.

This module provides a thin layer of classes that allow a lunr index to be
persisted into a relational database and queried back without loading the
entire inverted index into memory. It was originally designed for larger
back‑ends like MySQL or PostgreSQL, but for the purposes of demonstration
and testing it can operate against SQLite as well.

The design follows the architecture described in the accompanying plan:

* Terms and their integer indices are stored in a table ``lunr_terms``.
* Postings (term -> field -> doc_ref -> metadata) are stored in
  ``lunr_postings``.
* Field vectors (used for scoring) are stored in ``lunr_field_vectors``.

The public API comprises three main classes:

``SqlStorage`` encapsulates the database connection and exposes ``writer``
and ``reader`` factories. ``SqlIndexWriter`` persists index information
into the database. ``SqlIndexReader`` reconstructs postings and vectors
on demand for use by the querying engine. Two small proxy classes,
``SqlInvertedIndexProxy`` and ``SqlFieldVectorsProxy``, present a dictionary‑like
interface to satisfy the expectations of ``lunr.Index`` without materialising
all data up front.

This implementation is intentionally minimal. It only supports exact and
wildcard term expansion via SQL's ``LIKE`` operator and does not support
negated or edit‑distance queries. Users of the storage backend should be
aware of these limitations. See the documentation in the pull request plan
for further discussion of caveats and performance considerations.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, KeysView, Iterator
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lunr.vector import Vector


class SqlStorage:
    """Encapsulates a connection to a relational database used as the
    persistence layer for a lunr index.

    Parameters
    ----------
    conn : sqlite3.Connection
        A database connection. For demonstration purposes this module uses
        SQLite, but in principle any DB‑API 2.0 compatible connection that
        understands the necessary SQL dialect would work.
    index_name : str
        A name for the index. Multiple indexes can share the same database
        provided they use different names. The index name is stored with
        every record to isolate rows belonging to each logical index.
    """

    def __init__(self, conn: sqlite3.Connection, index_name: str) -> None:
        self.conn = conn
        self.index_name = index_name
        # Make sure foreign keys and WAL are enabled where available. This
        # doesn't matter for SQLite in memory mode but is harmless otherwise.
        try:
            self.conn.execute("PRAGMA foreign_keys = ON")
        except Exception:
            pass

    @classmethod
    def from_url(cls, url: str, index_name: str) -> "SqlStorage":
        """Create a storage backend from a URL.

        The only supported URL scheme in this simple implementation is
        ``sqlite:///path/to/db.sqlite`` or ``sqlite:///:memory:``. The
        database will be created if it does not exist.
        """
        if not url.startswith("sqlite:///"):
            raise ValueError("Only sqlite URLs are supported in this demo")
        path = url[len("sqlite:///"):]
        conn = sqlite3.connect(path)
        return cls(conn, index_name)

    @classmethod
    def from_conn(cls, conn: sqlite3.Connection, index_name: str) -> "SqlStorage":
        return cls(conn, index_name)

    def ensure_schema(self) -> None:
        """Create the necessary tables if they don't already exist."""
        c = self.conn.cursor()
        # Terms table: stores term string and its integer index.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS lunr_terms (
                index_name TEXT NOT NULL,
                term TEXT NOT NULL,
                term_index INTEGER NOT NULL,
                PRIMARY KEY (index_name, term)
            )
            """
        )
        # Postings: one row per (term, field, doc_ref). Metadata is stored as
        # JSON text. We do not impose foreign key constraints for simplicity.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS lunr_postings (
                index_name TEXT NOT NULL,
                term TEXT NOT NULL,
                field TEXT NOT NULL,
                doc_ref TEXT NOT NULL,
                metadata TEXT NOT NULL,
                PRIMARY KEY (index_name, term, field, doc_ref)
            )
            """
        )
        # Field vectors: one row per fieldRef (fieldName/docRef). Elements is a
        # JSON encoded list of [index, value] pairs and magnitude is stored
        # separately to avoid recomputing it at query time.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS lunr_field_vectors (
                index_name TEXT NOT NULL,
                field_ref TEXT NOT NULL,
                field TEXT NOT NULL,
                doc_ref TEXT NOT NULL,
                elements TEXT NOT NULL,
                magnitude REAL NOT NULL,
                PRIMARY KEY (index_name, field_ref)
            )
            """
        )
        # Flush table creation before creating indexes. Commit ensures the
        # tables exist; subsequent index creation statements will not error
        # if the tables have just been created.
        self.conn.commit()

        # In production environments it is beneficial to create indexes on
        # frequently queried columns. These indexes improve lookup speed for
        # exact term matching and posting retrieval. SQLite supports
        # conditional creation via IF NOT EXISTS; other databases may use
        # different syntax. Use try/except to silently ignore failures on
        # unsupported dialects.
        try:
            # Index terms for exact matches and prefix LIKE searches.
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_lunr_terms_term ON lunr_terms (term)"
            )
        except Exception:
            pass
        try:
            # Index postings by term and field to accelerate inverted lookup.
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_lunr_postings_term_field ON lunr_postings (term, field)"
            )
        except Exception:
            pass
        try:
            # Index field vectors by field_ref for faster retrieval.
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_lunr_field_vectors_ref ON lunr_field_vectors (field_ref)"
            )
        except Exception:
            pass
        self.conn.commit()

    def writer(self) -> "SqlIndexWriter":
        """Return a writer bound to this storage backend."""
        return SqlIndexWriter(self)

    def reader(self) -> "SqlIndexReader":
        """Return a reader bound to this storage backend."""
        return SqlIndexReader(self)


class SqlIndexWriter:
    """Writes index data into the database.

    The writer performs upserts into the various tables. It should be
    obtained from ``SqlStorage.writer`` and used in the build phase.
    """

    def __init__(self, storage: SqlStorage) -> None:
        self.storage = storage
        self.conn = storage.conn
        self.index_name = storage.index_name
        # Ensure schema exists before writing.
        storage.ensure_schema()
        self._term_cache: Dict[str, int] = {}
        # Begin a transaction for batch writes. Wrapping all upserts in a
        # transaction improves performance significantly when building large
        # indexes. SQLite will implicitly manage nested BEGIN statements.
        try:
            self.conn.execute("BEGIN")
        except Exception:
            # Some DB-API drivers may not allow explicit BEGIN; ignore if not needed.
            pass

    def upsert_term(self, term: str, term_index: int) -> None:
        """Insert or update a term and its numeric index."""
        # Cache to avoid redundant writes in the same session.
        if term in self._term_cache:
            return
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO lunr_terms (index_name, term, term_index) "
            "VALUES (?, ?, ?)",
            (self.index_name, term, term_index),
        )
        self._term_cache[term] = term_index

    def upsert_posting(self, term: str, field: str, doc_ref: str, metadata: Dict[str, List[Any]]) -> None:
        """Insert or update a posting entry."""
        # Serialize metadata dict as JSON text. Use json.dumps to ensure
        # reproducibility across Python versions.
        metadata_json = json.dumps(metadata, sort_keys=True)
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO lunr_postings (index_name, term, field, doc_ref, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (self.index_name, term, field, doc_ref, metadata_json),
        )

    def upsert_field_vector(self, field_ref: str, field: str, doc_ref: str, vector: Vector) -> None:
        """Insert or update a field vector."""
        # Serialize the vector into a list of [index, value] pairs and store
        # alongside its magnitude.
        elements = json.dumps(vector.serialize(), sort_keys=True)
        magnitude = vector.magnitude
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO lunr_field_vectors (index_name, field_ref, field, doc_ref, elements, magnitude) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self.index_name, field_ref, field, doc_ref, elements, magnitude),
        )

    def commit(self) -> None:
        self.conn.commit()


class SqlIndexReader:
    """Reads index data from the database.

    The reader is responsible for reconstructing postings and field vectors on
    demand and providing term expansion functionality via SQL ``LIKE``.
    """

    def __init__(self, storage: SqlStorage) -> None:
        self.storage = storage
        self.conn = storage.conn
        self.index_name = storage.index_name

    def expand_terms(self, pattern: str) -> List[str]:
        """Return a list of indexed terms matching the given pattern.

        The pattern may contain the wildcard character '*' which will be
        translated into SQL's '%' wildcard. All other characters are matched
        literally. Backslash is used as the escape character. If no wildcard
        is present the lookup is performed exactly.
        """
        # Determine if the pattern contains wildcards.
        if "*" in pattern:
            # Escape '%' and '_' so they match literally. SQLite uses
            # backslash as the default escape character when specified in
            # LIKE ... ESCAPE clause.
            sql_pattern = pattern.replace("%", r"\%").replace("_", r"\_")
            sql_pattern = sql_pattern.replace("*", "%")
            query = (
                "SELECT term FROM lunr_terms WHERE index_name = ? AND term LIKE ? ESCAPE '\\'"
            )
            params = (self.index_name, sql_pattern)
        else:
            # Exact match.
            query = (
                "SELECT term FROM lunr_terms WHERE index_name = ? AND term = ?"
            )
            params = (self.index_name, pattern)
        c = self.conn.cursor()
        c.execute(query, params)
        rows = c.fetchall()
        return [row[0] for row in rows]

    def get_posting(self, term: str) -> Dict[str, Any]:
        """Reconstruct the posting for a given term.

        The returned structure mirrors the one created by the Builder: a
        dictionary keyed by field names whose values are dictionaries keyed
        by doc_ref mapping to metadata dicts. A special ``"_index"`` key
        contains the integer term index.
        """
        # First fetch the term index. If the term does not exist return an
        # empty posting.
        c = self.conn.cursor()
        c.execute(
            "SELECT term_index FROM lunr_terms WHERE index_name = ? AND term = ?",
            (self.index_name, term),
        )
        row = c.fetchone()
        if row is None:
            # Return an empty posting with no fields. The caller will handle
            # non‑existent terms appropriately.
            return {"_index": -1}
        term_index = row[0]
        posting: Dict[str, Any] = {"_index": term_index}
        # Now fetch all rows for this term and build the nested dictionaries.
        c.execute(
            "SELECT field, doc_ref, metadata FROM lunr_postings WHERE index_name = ? AND term = ?",
            (self.index_name, term),
        )
        rows = c.fetchall()
        for field, doc_ref, metadata_json in rows:
            # Initialise dictionaries on demand.
            if field not in posting:
                posting[field] = {}
            # Metadata is stored as a JSON object mapping metadata keys to
            # arrays of values. Convert from JSON text to Python dict.
            metadata = json.loads(metadata_json)
            posting[field][doc_ref] = metadata
        return posting

    def get_field_vector(self, field_ref: str) -> Vector:
        """Reconstruct a Vector object from storage."""
        c = self.conn.cursor()
        c.execute(
            "SELECT elements, magnitude FROM lunr_field_vectors WHERE index_name = ? AND field_ref = ?",
            (self.index_name, field_ref),
        )
        row = c.fetchone()
        if row is None:
            # Return an empty vector if not found.
            return Vector()
        elements_json, magnitude = row
        # Elements were stored using Vector.serialize(), which returns a flat
        # array of alternating index and value. We can construct a Vector
        # directly from this list and then assign the magnitude. See
        # lunr.vector.Vector.__init__ for details.
        elements_list: List[float] = json.loads(elements_json)
        vector = Vector(elements_list)
        vector._magnitude = magnitude
        return vector

    def iter_all_field_refs(self) -> Iterator[str]:
        """Return an iterator over all field_ref keys.

        This is rarely needed in SQL mode because negated queries are not
        supported. It is implemented here for completeness.
        """
        c = self.conn.cursor()
        c.execute(
            "SELECT field_ref FROM lunr_field_vectors WHERE index_name = ?",
            (self.index_name,),
        )
        for (field_ref,) in c.fetchall():
            yield field_ref


class SqlInvertedIndexProxy(Mapping):
    """A mapping proxy around the underlying postings stored in SQL.

    Only ``__getitem__`` is implemented, returning a reconstructed posting
    dictionary from the storage backend. Iteration and ``len`` are not
    implemented because they would require scanning the entire table and are
    unnecessary for query execution.
    """

    def __init__(self, reader: SqlIndexReader) -> None:
        self.reader = reader

    def __getitem__(self, term: str) -> Dict[str, Any]:
        return self.reader.get_posting(term)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover
        # Not implemented; would require scanning entire postings table.
        raise NotImplementedError("Iteration over SqlInvertedIndexProxy is not supported")

    def __len__(self) -> int:  # pragma: no cover
        # Not implemented; would require counting rows.
        raise NotImplementedError("__len__ is not supported on SqlInvertedIndexProxy")


class SqlFieldVectorsProxy(Mapping):
    """A mapping proxy around field vectors stored in SQL.

    ``__getitem__`` retrieves a vector by field_ref. ``keys`` is implemented
    for completeness but will materialise all field_ref strings in memory.
    ``__iter__`` defers to ``keys``. ``len`` is not supported.
    """

    def __init__(self, reader: SqlIndexReader) -> None:
        self.reader = reader

    def __getitem__(self, field_ref: str) -> Vector:
        return self.reader.get_field_vector(field_ref)

    def keys(self) -> KeysView[str]:  # pragma: no cover
        # Materialise all field_refs; required only if the caller iterates
        # through all vectors (e.g. in negated queries). Use with care.
        return set(self.reader.iter_all_field_refs()).keys()

    def __iter__(self) -> Iterator[str]:  # pragma: no cover
        return self.keys().__iter__()

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError("__len__ is not supported on SqlFieldVectorsProxy")