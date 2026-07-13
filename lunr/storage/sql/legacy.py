from __future__ import annotations

import json
from collections.abc import Mapping, Iterator
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from lunr.vector import Vector


@dataclass(frozen=True)
class SqlDialect:
    name: str
    placeholder: str = "?"
    upsert: str = "replace"

    @property
    def is_sqlite(self) -> bool:
        return self.name == "sqlite"


DIALECTS = {
    "sqlite": SqlDialect("sqlite", "?", "replace"),
    "postgresql": SqlDialect("postgresql", "%s", "conflict"),
    "postgres": SqlDialect("postgres", "%s", "conflict"),
    "mysql": SqlDialect("mysql", "%s", "duplicate"),
}


class _LegacySqlStorage:
    """Former package facade retained only for the temporary V1 implementation."""

    def __init__(self, conn, index_name: str, dialect: str = "sqlite") -> None:
        self.conn = conn
        self.index_name = index_name
        self.dialect = DIALECTS[dialect]

    @classmethod
    def from_url(cls, url: str, index_name: str) -> "_LegacySqlStorage":
        parsed = urlparse(url)
        if parsed.scheme == "sqlite":
            import sqlite3

            path = parsed.path if parsed.path else ":memory:"
            if path.startswith("/") and path != "/:memory:":
                db_path = path
            else:
                db_path = ":memory:"
            return cls(
                sqlite3.connect(db_path), index_name=index_name, dialect="sqlite"
            )

        if parsed.scheme in {"postgresql", "postgres"}:
            try:
                import psycopg

                return cls(
                    psycopg.connect(url), index_name=index_name, dialect="postgresql"
                )
            except ImportError:
                import psycopg2

                return cls(
                    psycopg2.connect(url), index_name=index_name, dialect="postgresql"
                )

        if parsed.scheme == "mysql":
            try:
                import pymysql

                return cls(
                    pymysql.connect(
                        host=parsed.hostname,
                        user=parsed.username,
                        password=parsed.password,
                        database=parsed.path.lstrip("/"),
                        port=parsed.port or 3306,
                    ),
                    index_name=index_name,
                    dialect="mysql",
                )
            except ImportError:
                import MySQLdb

                return cls(
                    MySQLdb.connect(
                        host=parsed.hostname,
                        user=parsed.username,
                        passwd=parsed.password,
                        db=parsed.path.lstrip("/"),
                        port=parsed.port or 3306,
                    ),
                    index_name=index_name,
                    dialect="mysql",
                )

        raise ValueError("Unsupported SQL URL scheme")

    @classmethod
    def from_conn(
        cls, conn, index_name: str, dialect: str = "sqlite"
    ) -> "_LegacySqlStorage":
        return cls(conn=conn, index_name=index_name, dialect=dialect)

    def _ph(self, n: int) -> str:
        return ", ".join([self.dialect.placeholder] * n)

    def ensure_schema(self) -> None:
        c = self.conn.cursor()
        try:
            if self.dialect.name == "mysql":
                key_text = "VARCHAR(191)"
                metadata_type = "JSON"
                magnitude_type = "DOUBLE"
            elif self.dialect.name == "sqlite":
                key_text = "TEXT"
                metadata_type = "TEXT"
                magnitude_type = "REAL"
            else:
                key_text = "TEXT"
                metadata_type = "JSON"
                magnitude_type = "REAL"

            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS lunr_terms (
                    index_name {key_text} NOT NULL,
                    term {key_text} NOT NULL,
                    term_index INTEGER NOT NULL,
                    PRIMARY KEY (index_name, term)
                )
                """
            )
            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS lunr_postings (
                    index_name {key_text} NOT NULL,
                    term {key_text} NOT NULL,
                    field {key_text} NOT NULL,
                    doc_ref {key_text} NOT NULL,
                    metadata {metadata_type} NOT NULL,
                    PRIMARY KEY (index_name, term, field, doc_ref)
                )
                """
            )
            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS lunr_field_vectors (
                    index_name {key_text} NOT NULL,
                    field_ref {key_text} NOT NULL,
                    field {key_text} NOT NULL,
                    doc_ref {key_text} NOT NULL,
                    elements {metadata_type} NOT NULL,
                    magnitude {magnitude_type} NOT NULL,
                    PRIMARY KEY (index_name, field_ref)
                )
                """
            )
            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS lunr_doc_fields (
                    index_name {key_text} NOT NULL,
                    field_ref {key_text} NOT NULL,
                    field {key_text} NOT NULL,
                    doc_ref {key_text} NOT NULL,
                    length INTEGER NOT NULL,
                    PRIMARY KEY (index_name, field_ref)
                )
                """
            )
            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS lunr_term_frequencies (
                    index_name {key_text} NOT NULL,
                    field_ref {key_text} NOT NULL,
                    term {key_text} NOT NULL,
                    tf INTEGER NOT NULL,
                    PRIMARY KEY (index_name, field_ref, term)
                )
                """
            )
            if self.dialect.name == "mysql":
                indexes = [
                    (
                        "idx_lunr_postings_term_field",
                        "lunr_postings",
                        "(index_name, term, field)",
                    ),
                    ("idx_lunr_doc_fields_field", "lunr_doc_fields", "(index_name, field)"),
                    (
                        "idx_lunr_tf_field_ref",
                        "lunr_term_frequencies",
                        "(index_name, field_ref)",
                    ),
                ]
                for index_name, table_name, columns in indexes:
                    c.execute(
                        """
                        SELECT COUNT(1)
                        FROM information_schema.statistics
                        WHERE table_schema = DATABASE()
                          AND table_name = %s
                          AND index_name = %s
                        """,
                        (table_name, index_name),
                    )
                    if c.fetchone()[0] == 0:
                        c.execute(f"CREATE INDEX {index_name} ON {table_name} {columns}")
            else:
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_lunr_postings_term_field ON "
                    "lunr_postings (index_name, term, field)"
                )
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_lunr_doc_fields_field ON "
                    "lunr_doc_fields (index_name, field)"
                )
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_lunr_tf_field_ref ON "
                    "lunr_term_frequencies (index_name, field_ref)"
                )
        finally:
            c.close()
        self.conn.commit()

    def writer(self) -> "SqlIndexWriter":
        return SqlIndexWriter(self)

    def reader(self) -> "SqlIndexReader":
        return SqlIndexReader(self)


class SqlIndexWriter:
    def __init__(self, storage: Any) -> None:
        self.storage = storage
        self.conn = storage.conn
        self.index_name = storage.index_name
        _LegacySqlStorage.ensure_schema(storage)

    def _upsert(
        self,
        table: str,
        columns: List[str],
        values: Tuple[Any, ...],
        key_columns: List[str],
    ) -> None:
        c = self.conn.cursor()
        try:
            placeholders = ", ".join([self.storage.dialect.placeholder] * len(columns))
            cols = ", ".join(columns)
            if self.storage.dialect.upsert == "replace":
                c.execute(
                    f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
                    values,
                )
                return

            if self.storage.dialect.upsert == "conflict":
                non_keys = [col for col in columns if col not in key_columns]
                set_clause = ", ".join([f"{col}=EXCLUDED.{col}" for col in non_keys])
                c.execute(
                    f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
                    f"ON CONFLICT ({', '.join(key_columns)}) DO UPDATE SET {set_clause}",
                    values,
                )
                return

            updates = ", ".join(
                [f"{col}=VALUES({col})" for col in columns if col not in key_columns]
            )
            c.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {updates}",
                values,
            )
        finally:
            c.close()

    def upsert_term(self, term: str, term_index: int) -> None:
        self._upsert(
            "lunr_terms",
            ["index_name", "term", "term_index"],
            (self.index_name, term, term_index),
            ["index_name", "term"],
        )

    def upsert_terms_bulk(self, rows: List[Tuple[str, int]]) -> None:
        if not rows:
            return

        c = self.conn.cursor()
        try:
            placeholders = ", ".join([self.storage.dialect.placeholder] * 3)
            values = [(self.index_name, term, term_index) for term, term_index in rows]

            if self.storage.dialect.upsert == "replace":
                c.executemany(
                    "INSERT OR REPLACE INTO lunr_terms "
                    f"(index_name, term, term_index) VALUES ({placeholders})",
                    values,
                )
                return

            if self.storage.dialect.upsert == "conflict":
                c.executemany(
                    "INSERT INTO lunr_terms (index_name, term, term_index) "
                    f"VALUES ({placeholders}) "
                    "ON CONFLICT (index_name, term) DO UPDATE "
                    "SET term_index=EXCLUDED.term_index",
                    values,
                )
                return

            c.executemany(
                "INSERT INTO lunr_terms (index_name, term, term_index) "
                f"VALUES ({placeholders}) "
                "ON DUPLICATE KEY UPDATE term_index=VALUES(term_index)",
                values,
            )
        finally:
            c.close()

    def upsert_posting(
        self, term: str, field: str, doc_ref: str, metadata: Dict[str, List[Any]]
    ) -> None:
        self._upsert(
            "lunr_postings",
            ["index_name", "term", "field", "doc_ref", "metadata"],
            (
                self.index_name,
                term,
                field,
                doc_ref,
                json.dumps(metadata, sort_keys=True),
            ),
            ["index_name", "term", "field", "doc_ref"],
        )

    def upsert_postings_bulk(
        self, rows: List[Tuple[str, str, str, Dict[str, List[Any]]]]
    ) -> None:
        if not rows:
            return

        c = self.conn.cursor()
        try:
            placeholders = ", ".join([self.storage.dialect.placeholder] * 5)
            values = [
                (
                    self.index_name,
                    term,
                    field,
                    doc_ref,
                    json.dumps(metadata, sort_keys=True),
                )
                for term, field, doc_ref, metadata in rows
            ]

            if self.storage.dialect.upsert == "replace":
                c.executemany(
                    "INSERT OR REPLACE INTO lunr_postings "
                    f"(index_name, term, field, doc_ref, metadata) VALUES ({placeholders})",
                    values,
                )
                return

            if self.storage.dialect.upsert == "conflict":
                c.executemany(
                    "INSERT INTO lunr_postings (index_name, term, field, doc_ref, metadata) "
                    f"VALUES ({placeholders}) "
                    "ON CONFLICT (index_name, term, field, doc_ref) DO UPDATE "
                    "SET metadata=EXCLUDED.metadata",
                    values,
                )
                return

            c.executemany(
                "INSERT INTO lunr_postings (index_name, term, field, doc_ref, metadata) "
                f"VALUES ({placeholders}) "
                "ON DUPLICATE KEY UPDATE metadata=VALUES(metadata)",
                values,
            )
        finally:
            c.close()

    def upsert_field_vector(
        self, field_ref: str, field: str, doc_ref: str, vector: Vector
    ) -> None:
        self._upsert(
            "lunr_field_vectors",
            ["index_name", "field_ref", "field", "doc_ref", "elements", "magnitude"],
            (
                self.index_name,
                field_ref,
                field,
                doc_ref,
                json.dumps(vector.serialize()),
                vector.magnitude,
            ),
            ["index_name", "field_ref"],
        )

    def upsert_field_vectors_bulk(
        self, rows: List[Tuple[str, str, str, Vector]]
    ) -> None:
        if not rows:
            return

        c = self.conn.cursor()
        try:
            placeholders = ", ".join([self.storage.dialect.placeholder] * 6)
            values = [
                (
                    self.index_name,
                    field_ref,
                    field,
                    doc_ref,
                    json.dumps(vector.serialize()),
                    vector.magnitude,
                )
                for field_ref, field, doc_ref, vector in rows
            ]

            if self.storage.dialect.upsert == "replace":
                c.executemany(
                    "INSERT OR REPLACE INTO lunr_field_vectors "
                    f"(index_name, field_ref, field, doc_ref, elements, magnitude) "
                    f"VALUES ({placeholders})",
                    values,
                )
                return

            if self.storage.dialect.upsert == "conflict":
                c.executemany(
                    "INSERT INTO lunr_field_vectors "
                    "(index_name, field_ref, field, doc_ref, elements, magnitude) "
                    f"VALUES ({placeholders}) "
                    "ON CONFLICT (index_name, field_ref) DO UPDATE "
                    "SET field=EXCLUDED.field, doc_ref=EXCLUDED.doc_ref, "
                    "elements=EXCLUDED.elements, magnitude=EXCLUDED.magnitude",
                    values,
                )
                return

            c.executemany(
                "INSERT INTO lunr_field_vectors "
                "(index_name, field_ref, field, doc_ref, elements, magnitude) "
                f"VALUES ({placeholders}) "
                "ON DUPLICATE KEY UPDATE field=VALUES(field), doc_ref=VALUES(doc_ref), "
                "elements=VALUES(elements), magnitude=VALUES(magnitude)",
                values,
            )
        finally:
            c.close()

    def purge_terms_above_df(self, threshold: int) -> None:
        """Delete terms whose document frequency meets or exceeds *threshold*.

        Document frequency is the number of distinct documents that contain the
        term (across any field).  The method removes matching rows from
        ``lunr_terms``, ``lunr_postings``, and ``lunr_term_frequencies`` so
        that subsequent vector computation naturally skips the purged terms.
        """
        ph = self.storage.dialect.placeholder
        c = self.conn.cursor()

        # Identify terms to purge using the postings table.
        c.execute(
            "SELECT term FROM lunr_postings "
            f"WHERE index_name = {ph} "
            "GROUP BY term "
            f"HAVING COUNT(DISTINCT doc_ref) >= {ph}",
            (self.index_name, threshold),
        )
        terms = [row[0] for row in c.fetchall()]
        if not terms:
            return

        # Delete in batches to stay within parameter limits.
        batch_size = 500
        for i in range(0, len(terms), batch_size):
            batch = terms[i : i + batch_size]
            placeholders = ", ".join([ph] * len(batch))
            params: tuple = (self.index_name, *batch)

            c.execute(
                f"DELETE FROM lunr_terms WHERE index_name = {ph} "
                f"AND term IN ({placeholders})",
                params,
            )
            c.execute(
                f"DELETE FROM lunr_postings WHERE index_name = {ph} "
                f"AND term IN ({placeholders})",
                params,
            )
            c.execute(
                f"DELETE FROM lunr_term_frequencies WHERE index_name = {ph} "
                f"AND term IN ({placeholders})",
                params,
            )

    def recompute_doc_field_lengths(self) -> None:
        """Recompute ``lunr_doc_fields.length`` from remaining term frequencies.

        After high-df terms are purged from ``lunr_term_frequencies``, the
        ``length`` stored in ``lunr_doc_fields`` becomes stale.  This method
        first zeroes every length (so fields that lost all terms get length 0),
        then sets each row's ``length`` to ``SUM(tf)`` of its surviving terms
        so that subsequent BM25 scoring normalises by the correct field length.
        """
        ph = self.storage.dialect.placeholder
        c = self.conn.cursor()
        # Zero all lengths first so fields with no surviving terms get 0.
        c.execute(
            f"UPDATE lunr_doc_fields SET length = 0 WHERE index_name = {ph}",
            (self.index_name,),
        )
        c.execute(
            "SELECT field_ref, SUM(tf) FROM lunr_term_frequencies "
            f"WHERE index_name = {ph} GROUP BY field_ref",
            (self.index_name,),
        )
        updates = c.fetchall()
        if not updates:
            return
        batch_size = 500
        for i in range(0, len(updates), batch_size):
            batch = updates[i : i + batch_size]
            c.executemany(
                f"UPDATE lunr_doc_fields SET length = {ph} "
                f"WHERE index_name = {ph} AND field_ref = {ph}",
                [
                    (new_length, self.index_name, field_ref)
                    for field_ref, new_length in batch
                ],
            )

    def commit(self) -> None:
        self.conn.commit()

    def upsert_doc_fields_bulk(self, rows: List[Tuple[str, str, str, int]]) -> None:
        if not rows:
            return

        c = self.conn.cursor()
        try:
            placeholders = ", ".join([self.storage.dialect.placeholder] * 5)
            values = [
                (self.index_name, field_ref, field, doc_ref, length)
                for field_ref, field, doc_ref, length in rows
            ]

            if self.storage.dialect.upsert == "replace":
                c.executemany(
                    "INSERT OR REPLACE INTO lunr_doc_fields "
                    f"(index_name, field_ref, field, doc_ref, length) VALUES ({placeholders})",
                    values,
                )
                return

            if self.storage.dialect.upsert == "conflict":
                c.executemany(
                    "INSERT INTO lunr_doc_fields "
                    "(index_name, field_ref, field, doc_ref, length) "
                    f"VALUES ({placeholders}) "
                    "ON CONFLICT (index_name, field_ref) DO UPDATE "
                    "SET field=EXCLUDED.field, doc_ref=EXCLUDED.doc_ref, "
                    "length=EXCLUDED.length",
                    values,
                )
                return

            c.executemany(
                "INSERT INTO lunr_doc_fields "
                "(index_name, field_ref, field, doc_ref, length) "
                f"VALUES ({placeholders}) "
                "ON DUPLICATE KEY UPDATE field=VALUES(field), doc_ref=VALUES(doc_ref), "
                "length=VALUES(length)",
                values,
            )
        finally:
            c.close()

    def upsert_term_frequencies_bulk(self, rows: List[Tuple[str, str, int]]) -> None:
        if not rows:
            return

        c = self.conn.cursor()
        try:
            placeholders = ", ".join([self.storage.dialect.placeholder] * 4)
            values = [
                (self.index_name, field_ref, term, tf) for field_ref, term, tf in rows
            ]

            if self.storage.dialect.upsert == "replace":
                c.executemany(
                    "INSERT OR REPLACE INTO lunr_term_frequencies "
                    f"(index_name, field_ref, term, tf) VALUES ({placeholders})",
                    values,
                )
                return

            if self.storage.dialect.upsert == "conflict":
                c.executemany(
                    "INSERT INTO lunr_term_frequencies "
                    "(index_name, field_ref, term, tf) "
                    f"VALUES ({placeholders}) "
                    "ON CONFLICT (index_name, field_ref, term) DO UPDATE "
                    "SET tf=EXCLUDED.tf",
                    values,
                )
                return

            c.executemany(
                "INSERT INTO lunr_term_frequencies "
                "(index_name, field_ref, term, tf) "
                f"VALUES ({placeholders}) "
                "ON DUPLICATE KEY UPDATE tf=VALUES(tf)",
                values,
            )
        finally:
            c.close()


class SqlIndexReader:
    def __init__(self, storage: Any) -> None:
        self.storage = storage
        self.conn = storage.conn
        self.index_name = storage.index_name

    def expand_terms(self, term_pattern: str) -> List[str]:
        c = self.conn.cursor()
        try:
            if "*" in term_pattern:
                escaped = (
                    term_pattern.replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                like_pattern = escaped.replace("*", "%")
                if self.storage.dialect.name == "mysql":
                    c.execute(
                        f"SELECT term FROM lunr_terms WHERE index_name = {self.storage.dialect.placeholder} AND term LIKE {self.storage.dialect.placeholder}",
                        (self.index_name, like_pattern),
                    )
                else:
                    c.execute(
                        f"SELECT term FROM lunr_terms WHERE index_name = {self.storage.dialect.placeholder} AND term LIKE {self.storage.dialect.placeholder} ESCAPE '\\'",
                        (self.index_name, like_pattern),
                    )
            else:
                c.execute(
                    f"SELECT term FROM lunr_terms WHERE index_name = {self.storage.dialect.placeholder} AND term = {self.storage.dialect.placeholder}",
                    (self.index_name, term_pattern),
                )
            return [term for (term,) in c.fetchall()]
        finally:
            c.close()

    def get_posting(self, term: str) -> Dict[str, Any]:
        c = self.conn.cursor()
        try:
            c.execute(
                f"SELECT term_index FROM lunr_terms WHERE index_name = {self.storage.dialect.placeholder} AND term = {self.storage.dialect.placeholder}",
                (self.index_name, term),
            )
            row = c.fetchone()
            if row is None:
                return {"_index": -1}

            posting: Dict[str, Any] = {"_index": row[0]}
            c.execute(
                f"SELECT field, doc_ref, metadata FROM lunr_postings WHERE index_name = {self.storage.dialect.placeholder} AND term = {self.storage.dialect.placeholder}",
                (self.index_name, term),
            )
            for field, doc_ref, metadata_json in c.fetchall():
                posting.setdefault(field, {})[doc_ref] = json.loads(metadata_json)
            return posting
        finally:
            c.close()

    def get_field_vector(self, field_ref: str) -> Vector:
        c = self.conn.cursor()
        try:
            c.execute(
                f"SELECT elements, magnitude FROM lunr_field_vectors WHERE index_name = {self.storage.dialect.placeholder} AND field_ref = {self.storage.dialect.placeholder}",
                (self.index_name, field_ref),
            )
            row = c.fetchone()
            if row is None:
                return Vector()
            vector = Vector(json.loads(row[0]))
            vector._magnitude = row[1]
            return vector
        finally:
            c.close()

    def iter_all_field_refs(self) -> Iterator[str]:
        c = self.conn.cursor()
        try:
            c.execute(
                f"SELECT field_ref FROM lunr_field_vectors WHERE index_name = {self.storage.dialect.placeholder}",
                (self.index_name,),
            )
            rows = c.fetchall()
        finally:
            c.close()
        for (field_ref,) in rows:
            yield field_ref

    def iter_doc_fields(self) -> Iterator[Tuple[str, str, str, int]]:
        c = self.conn.cursor()
        try:
            c.execute(
                f"SELECT field_ref, field, doc_ref, length FROM lunr_doc_fields "
                f"WHERE index_name = {self.storage.dialect.placeholder}",
                (self.index_name,),
            )
            rows = c.fetchall()
        finally:
            c.close()
        for row in rows:
            yield row

    def iter_term_frequencies(self) -> Iterator[Tuple[str, str, int]]:
        c = self.conn.cursor()
        try:
            c.execute(
                f"SELECT field_ref, term, tf FROM lunr_term_frequencies "
                f"WHERE index_name = {self.storage.dialect.placeholder} "
                "ORDER BY field_ref",
                (self.index_name,),
            )
            rows = c.fetchall()
        finally:
            c.close()
        for row in rows:
            yield row


class SqlInvertedIndexProxy(Mapping):
    def __init__(self, reader: SqlIndexReader) -> None:
        self.reader = reader

    def __getitem__(self, term: str) -> Dict[str, Any]:
        return self.reader.get_posting(term)

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SqlFieldVectorsProxy(Mapping):
    def __init__(self, reader: SqlIndexReader) -> None:
        self.reader = reader

    def __getitem__(self, field_ref: str) -> Vector:
        return self.reader.get_field_vector(field_ref)

    def __iter__(self) -> Iterator[str]:
        return iter(self.reader.iter_all_field_refs())

    def __len__(self) -> int:
        raise NotImplementedError

    def keys(self):
        return set(self.reader.iter_all_field_refs())
