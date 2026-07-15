from __future__ import annotations

from .dialects import json_dump


class SqlIndexWriter:
    """Bulk writer for one invisible V2 index generation."""

    def __init__(self, storage, generation: str) -> None:
        self.conn = storage.conn
        self.index_name = storage.index_name
        self.dialect = storage.dialect
        self.generation = generation
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT state FROM lunr_v2_generations "
                f"WHERE index_name={self.dialect.placeholder} "
                f"AND generation={self.dialect.placeholder}",
                (self.index_name, self.generation),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None or row[0] != "building":
            raise ValueError(
                f"{generation!r} is not a building generation for "
                f"index {self.index_name!r}"
            )

    def _upsert_many(self, table, columns, keys, rows) -> None:
        if not rows:
            return
        updates = [column for column in columns if column not in keys]
        sql = self.dialect.upsert_sql(table, columns, keys, updates)
        cursor = self.conn.cursor()
        try:
            cursor.executemany(sql, rows)
        finally:
            cursor.close()

    def write_document_fields(self, rows) -> None:
        self._upsert_many(
            "lunr_v2_doc_fields",
            [
                "index_name",
                "generation",
                "field_ref",
                "field",
                "doc_ref",
                "length",
                "boost",
            ],
            ["index_name", "generation", "field_ref"],
            [
                (
                    self.index_name,
                    self.generation,
                    field_ref,
                    field,
                    doc_ref,
                    length,
                    boost,
                )
                for field_ref, field, doc_ref, length, boost in rows
            ],
        )

    def write_term_frequencies(self, rows) -> None:
        self._upsert_many(
            "lunr_v2_term_frequencies",
            ["index_name", "generation", "field_ref", "term", "tf"],
            ["index_name", "generation", "field_ref", "term"],
            [
                (self.index_name, self.generation, field_ref, term, tf)
                for field_ref, term, tf in rows
            ],
        )

    def write_postings(self, rows) -> None:
        self._upsert_many(
            "lunr_v2_postings",
            ["index_name", "generation", "term", "field", "doc_ref", "metadata"],
            ["index_name", "generation", "term", "field", "doc_ref"],
            [
                (
                    self.index_name,
                    self.generation,
                    term,
                    field,
                    doc_ref,
                    json_dump(metadata),
                )
                for term, field, doc_ref, metadata in rows
            ],
        )

    def finalize_terms(self, rows) -> None:
        self._upsert_many(
            "lunr_v2_terms",
            ["index_name", "generation", "term", "term_index"],
            ["index_name", "generation", "term"],
            [
                (self.index_name, self.generation, term, term_index)
                for term, term_index in rows
            ],
        )

    def write_vectors(self, rows) -> None:
        self._upsert_many(
            "lunr_v2_field_vectors",
            [
                "index_name",
                "generation",
                "field_ref",
                "field",
                "doc_ref",
                "elements",
                "magnitude",
            ],
            ["index_name", "generation", "field_ref"],
            [
                (
                    self.index_name,
                    self.generation,
                    field_ref,
                    field,
                    doc_ref,
                    json_dump(elements),
                    magnitude,
                )
                for field_ref, field, doc_ref, elements, magnitude in rows
            ],
        )
