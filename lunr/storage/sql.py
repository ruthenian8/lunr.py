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


class SqlStorage:
    def __init__(self, conn, index_name: str, dialect: str = "sqlite") -> None:
        self.conn = conn
        self.index_name = index_name
        self.dialect = DIALECTS[dialect]

    @classmethod
    def from_url(cls, url: str, index_name: str) -> "SqlStorage":
        parsed = urlparse(url)
        if parsed.scheme == "sqlite":
            import sqlite3

            path = parsed.path if parsed.path else ":memory:"
            if path.startswith("/") and path != "/:memory:":
                db_path = path
            else:
                db_path = ":memory:"
            return cls(sqlite3.connect(db_path), index_name=index_name, dialect="sqlite")

        if parsed.scheme in {"postgresql", "postgres"}:
            try:
                import psycopg

                return cls(psycopg.connect(url), index_name=index_name, dialect="postgresql")
            except ImportError:
                import psycopg2

                return cls(psycopg2.connect(url), index_name=index_name, dialect="postgresql")

        if parsed.scheme == "mysql":
            try:
                import pymysql

                return cls(pymysql.connect(host=parsed.hostname, user=parsed.username, password=parsed.password, database=parsed.path.lstrip("/"), port=parsed.port or 3306), index_name=index_name, dialect="mysql")
            except ImportError:
                import MySQLdb

                return cls(MySQLdb.connect(host=parsed.hostname, user=parsed.username, passwd=parsed.password, db=parsed.path.lstrip("/"), port=parsed.port or 3306), index_name=index_name, dialect="mysql")

        raise ValueError("Unsupported SQL URL scheme")

    @classmethod
    def from_conn(cls, conn, index_name: str, dialect: str = "sqlite") -> "SqlStorage":
        return cls(conn=conn, index_name=index_name, dialect=dialect)

    def _ph(self, n: int) -> str:
        return ", ".join([self.dialect.placeholder] * n)

    def ensure_schema(self) -> None:
        c = self.conn.cursor()
        metadata_type = "TEXT" if self.dialect.name == "sqlite" else "JSON"
        c.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_terms (
                index_name TEXT NOT NULL,
                term TEXT NOT NULL,
                term_index INTEGER NOT NULL,
                PRIMARY KEY (index_name, term)
            )
            """
        )
        c.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_postings (
                index_name TEXT NOT NULL,
                term TEXT NOT NULL,
                field TEXT NOT NULL,
                doc_ref TEXT NOT NULL,
                metadata {metadata_type} NOT NULL,
                PRIMARY KEY (index_name, term, field, doc_ref)
            )
            """
        )
        c.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_field_vectors (
                index_name TEXT NOT NULL,
                field_ref TEXT NOT NULL,
                field TEXT NOT NULL,
                doc_ref TEXT NOT NULL,
                elements {metadata_type} NOT NULL,
                magnitude REAL NOT NULL,
                PRIMARY KEY (index_name, field_ref)
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_lunr_postings_term_field ON lunr_postings (index_name, term, field)")
        self.conn.commit()

    def writer(self) -> "SqlIndexWriter":
        return SqlIndexWriter(self)

    def reader(self) -> "SqlIndexReader":
        return SqlIndexReader(self)


class SqlIndexWriter:
    def __init__(self, storage: SqlStorage) -> None:
        self.storage = storage
        self.conn = storage.conn
        self.index_name = storage.index_name
        self.storage.ensure_schema()

    def _upsert(self, table: str, columns: List[str], values: Tuple[Any, ...], key_columns: List[str]) -> None:
        c = self.conn.cursor()
        placeholders = ", ".join([self.storage.dialect.placeholder] * len(columns))
        cols = ", ".join(columns)
        if self.storage.dialect.upsert == "replace":
            c.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})", values)
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

        updates = ", ".join([f"{col}=VALUES({col})" for col in columns if col not in key_columns])
        c.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {updates}",
            values,
        )

    def upsert_term(self, term: str, term_index: int) -> None:
        self._upsert(
            "lunr_terms",
            ["index_name", "term", "term_index"],
            (self.index_name, term, term_index),
            ["index_name", "term"],
        )

    def upsert_posting(self, term: str, field: str, doc_ref: str, metadata: Dict[str, List[Any]]) -> None:
        self._upsert(
            "lunr_postings",
            ["index_name", "term", "field", "doc_ref", "metadata"],
            (self.index_name, term, field, doc_ref, json.dumps(metadata, sort_keys=True)),
            ["index_name", "term", "field", "doc_ref"],
        )

    def upsert_field_vector(self, field_ref: str, field: str, doc_ref: str, vector: Vector) -> None:
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

    def commit(self) -> None:
        self.conn.commit()


class SqlIndexReader:
    def __init__(self, storage: SqlStorage) -> None:
        self.storage = storage
        self.conn = storage.conn
        self.index_name = storage.index_name

    def expand_terms(self, term_pattern: str) -> List[str]:
        c = self.conn.cursor()
        if "*" in term_pattern:
            escaped = term_pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like_pattern = escaped.replace("*", "%")
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

    def get_posting(self, term: str) -> Dict[str, Any]:
        c = self.conn.cursor()
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

    def get_field_vector(self, field_ref: str) -> Vector:
        c = self.conn.cursor()
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

    def iter_all_field_refs(self) -> Iterator[str]:
        c = self.conn.cursor()
        c.execute(
            f"SELECT field_ref FROM lunr_field_vectors WHERE index_name = {self.storage.dialect.placeholder}",
            (self.index_name,),
        )
        for (field_ref,) in c.fetchall():
            yield field_ref


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
