from __future__ import annotations

from collections.abc import Iterable

from lunr.vector import Vector

from .dialects import json_load


MAX_PARAMETERS = 500


class SqlIndexReader:
    """Batched reader pinned to one active V2 index generation."""

    def __init__(self, storage, generation: str) -> None:
        self.conn = storage.conn
        self.index_name = storage.index_name
        self.dialect = storage.dialect
        self.generation = generation

    def _chunks(self, values: Iterable[str]):
        values = list(dict.fromkeys(values))
        size = MAX_PARAMETERS - 2
        for offset in range(0, len(values), size):
            yield values[offset : offset + size]

    def expand_terms(self, patterns) -> list[str]:
        if isinstance(patterns, str):
            patterns = [patterns]
        exact = [pattern for pattern in patterns if "*" not in pattern]
        wildcard = [pattern for pattern in patterns if "*" in pattern]
        matches = {}
        cursor = self.conn.cursor()
        try:
            for chunk in self._chunks(exact):
                placeholders = self.dialect.placeholders(len(chunk))
                cursor.execute(
                    "SELECT term, term_index FROM lunr_v2_terms "
                    f"WHERE index_name={self.dialect.placeholder} "
                    f"AND generation={self.dialect.placeholder} "
                    f"AND term IN ({placeholders})",
                    (self.index_name, self.generation, *chunk),
                )
                matches.update(cursor.fetchall())
            for pattern in wildcard:
                escaped = (
                    pattern.replace("!", "!!")
                    .replace("%", "!%")
                    .replace("_", "!_")
                    .replace("*", "%")
                )
                cursor.execute(
                    "SELECT term, term_index FROM lunr_v2_terms "
                    f"WHERE index_name={self.dialect.placeholder} "
                    f"AND generation={self.dialect.placeholder} "
                    f"AND term LIKE {self.dialect.placeholder} ESCAPE '!'",
                    (self.index_name, self.generation, escaped),
                )
                matches.update(cursor.fetchall())
        finally:
            cursor.close()
        return [term for term, _ in sorted(matches.items(), key=lambda item: item[1])]

    def get_postings(self, terms) -> dict:
        postings = {}
        cursor = self.conn.cursor()
        try:
            for chunk in self._chunks(terms):
                placeholders = self.dialect.placeholders(len(chunk))
                cursor.execute(
                    "SELECT t.term, t.term_index, p.field, p.doc_ref, p.metadata "
                    "FROM lunr_v2_terms t LEFT JOIN lunr_v2_postings p ON "
                    "p.index_name=t.index_name AND p.generation=t.generation "
                    "AND p.term=t.term "
                    f"WHERE t.index_name={self.dialect.placeholder} "
                    f"AND t.generation={self.dialect.placeholder} "
                    f"AND t.term IN ({placeholders})",
                    (self.index_name, self.generation, *chunk),
                )
                for term, term_index, field, doc_ref, metadata in cursor.fetchall():
                    posting = postings.setdefault(term, {"_index": term_index})
                    if field is not None:
                        posting.setdefault(field, {})[doc_ref] = json_load(metadata)
        finally:
            cursor.close()
        return postings

    def get_field_vectors(self, refs) -> dict:
        vectors = {}
        cursor = self.conn.cursor()
        try:
            for chunk in self._chunks(refs):
                placeholders = self.dialect.placeholders(len(chunk))
                cursor.execute(
                    "SELECT field_ref, elements, magnitude FROM lunr_v2_field_vectors "
                    f"WHERE index_name={self.dialect.placeholder} "
                    f"AND generation={self.dialect.placeholder} "
                    f"AND field_ref IN ({placeholders})",
                    (self.index_name, self.generation, *chunk),
                )
                for field_ref, elements, magnitude in cursor.fetchall():
                    vector = Vector(json_load(elements))
                    vector._magnitude = magnitude
                    vectors[field_ref] = vector
        finally:
            cursor.close()
        return vectors

    def iter_doc_fields(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT field_ref, field, doc_ref, length, boost "
                "FROM lunr_v2_doc_fields "
                f"WHERE index_name={self.dialect.placeholder} "
                f"AND generation={self.dialect.placeholder}",
                (self.index_name, self.generation),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        yield from rows

    def iter_term_frequencies(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT field_ref, term, tf FROM lunr_v2_term_frequencies "
                f"WHERE index_name={self.dialect.placeholder} "
                f"AND generation={self.dialect.placeholder} ORDER BY field_ref",
                (self.index_name, self.generation),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        yield from rows
