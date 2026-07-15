from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from lunr.vector import Vector

from .dialects import json_load


MAX_PARAMETERS = 500


@dataclass
class QueryData:
    reader: object
    expanded_terms: dict
    postings: dict
    field_vectors: dict = field(default_factory=dict)
    _loaded_field_refs: set = field(default_factory=set)

    def load_field_vectors(self, refs):
        missing = set(refs) - self._loaded_field_refs
        if missing:
            self.field_vectors.update(self.reader.get_field_vectors(missing))
            self._loaded_field_refs.update(missing)
        return self.field_vectors


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
        expanded, indexes = self._expand_term_map(patterns)
        terms = {term for values in expanded.values() for term in values}
        return sorted(terms, key=indexes.__getitem__)

    def _expand_term_map(self, patterns):
        if isinstance(patterns, str):
            patterns = [patterns]
        else:
            patterns = list(dict.fromkeys(patterns))
        exact = [pattern for pattern in patterns if "*" not in pattern]
        wildcard = [pattern for pattern in patterns if "*" in pattern]
        expanded = {pattern: [] for pattern in patterns}
        indexes = {}
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
                for term, term_index in cursor.fetchall():
                    expanded[term].append(term)
                    indexes[term] = term_index
            for chunk in self._chunks(wildcard):
                escaped = [
                    pattern.replace("!", "!!")
                    .replace("%", "!%")
                    .replace("_", "!_")
                    .replace("*", "%")
                    for pattern in chunk
                ]
                conditions = " OR ".join(
                    f"term LIKE {self.dialect.placeholder} ESCAPE '!'"
                    for _ in chunk
                )
                cursor.execute(
                    "SELECT term, term_index FROM lunr_v2_terms "
                    f"WHERE index_name={self.dialect.placeholder} "
                    f"AND generation={self.dialect.placeholder} "
                    f"AND ({conditions})",
                    (self.index_name, self.generation, *escaped),
                )
                rows = sorted(cursor.fetchall(), key=lambda row: row[1])
                for term, term_index in rows:
                    indexes[term] = term_index
                for pattern in chunk:
                    matcher = re.compile(
                        "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
                    )
                    expanded[pattern].extend(
                        term for term, _ in rows if matcher.match(term)
                    )
        finally:
            cursor.close()
        return expanded, indexes

    def prepare_query(self, patterns):
        expanded, _ = self._expand_term_map(patterns)
        terms = list(
            dict.fromkeys(term for values in expanded.values() for term in values)
        )
        return QueryData(self, expanded, self.get_postings(terms))

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

    def get_posting(self, term):
        return self.get_postings([term]).get(term, {"_index": -1})

    def get_field_vector(self, field_ref):
        return self.get_field_vectors([field_ref]).get(field_ref, Vector())

    def iter_all_field_refs(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT field_ref FROM lunr_v2_field_vectors "
                f"WHERE index_name={self.dialect.placeholder} "
                f"AND generation={self.dialect.placeholder}",
                (self.index_name, self.generation),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        for (field_ref,) in rows:
            yield field_ref

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
