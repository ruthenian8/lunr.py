import sqlite3
from types import SimpleNamespace

import pytest

from lunr.storage.sql import SqlStorage
from lunr.storage.sql.dialects import get_dialect
from lunr.storage.sql.reader import QueryData, SqlIndexReader as V2SqlIndexReader
from lunr.storage.sql.schema import (
    activate_generation,
    begin_generation,
    ensure_schema,
    fail_generation,
)
from lunr.storage.sql.writer import SqlIndexWriter as V2SqlIndexWriter


@pytest.fixture
def populated_storage():
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "docs")
    ensure_schema(connection, storage.dialect)
    generation = begin_generation(
        connection, storage.dialect, storage.index_name, ["title", "body"], []
    )
    writer = storage.writer(generation)
    writer.write_document_fields(
        [
            ("title/1", "title", "1", 1, 1.0),
            ("body/1", "body", "1", 3, 1.0),
        ]
    )
    writer.write_term_frequencies(
        [
            ("title/1", "green", 1),
            ("body/1", "study", 1),
            ("body/1", "машина", 1),
        ]
    )
    writer.write_postings(
        [
            ("green", "title", "1", {"position": [[0, 5]]}),
            ("study", "body", "1", {"position": [[6, 5]]}),
            ("машина", "body", "1", {"position": [[0, 6]]}),
        ]
    )
    writer.finalize_terms(
        [
            ("green", 0),
            ("study", 1),
            ("машина", 2),
            ("100%real", 3),
            ("100_percent", 4),
            ("100xother", 5),
            ("bang!value", 6),
        ]
    )
    writer.write_vectors(
        [
            ("title/1", "title", "1", [0, 1.25], 1.25),
            ("body/1", "body", "1", [1, 2.5, 2, 3.5], 4.3011626335),
        ]
    )
    connection.commit()
    activate_generation(connection, storage.dialect, storage.index_name, generation)
    try:
        yield storage
    finally:
        connection.close()


def test_writer_reader_json_roundtrip(populated_storage):
    reader = populated_storage.reader()

    posting = reader.get_postings(["машина"])["машина"]

    assert posting["body"]["1"] == {"position": [[0, 6]]}
    assert list(reader.get_field_vectors(["body/1"])["body/1"]) == [1, 2.5, 2, 3.5]


def test_two_term_query_data_uses_three_selects(populated_storage):
    reader = populated_storage.reader()
    statements = []
    populated_storage.conn.set_trace_callback(statements.append)

    terms = reader.expand_terms(["green", "study"])
    postings = reader.get_postings(terms)
    refs = {
        f"{field}/{doc}"
        for posting in postings.values()
        for field in ("title", "body")
        for doc in posting.get(field, {})
    }
    reader.get_field_vectors(refs)

    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) <= 3


def test_expand_terms_escapes_like_metacharacters(populated_storage):
    reader = populated_storage.reader()

    assert reader.expand_terms(["100%*"]) == ["100%real"]
    assert reader.expand_terms(["100_*"]) == ["100_percent"]
    assert reader.expand_terms(["bang!*"]) == ["bang!value"]


def test_wildcard_expansion_accepts_tuple_rows():
    cursor = _NativeJsonCursor((("green", 0), ("growth", 1)))
    storage = SimpleNamespace(
        conn=_SingleCursorConnection(cursor),
        index_name="docs",
        dialect=get_dialect("sqlite"),
    )

    assert V2SqlIndexReader(storage, "generation").expand_terms("gr*") == [
        "green",
        "growth",
    ]


def test_query_data_loads_missing_vectors_incrementally(populated_storage):
    query_data = QueryData(populated_storage.reader(), {}, {})

    first = set(query_data.load_field_vectors(["title/1"]))
    second = query_data.load_field_vectors(["title/1", "body/1"])

    assert first == {"title/1"}
    assert set(second) == {"title/1", "body/1"}


def test_expand_terms_materializes_one_shot_pattern_iterable(populated_storage):
    patterns = (pattern for pattern in ["green", "100%*"])

    assert populated_storage.reader().expand_terms(patterns) == ["green", "100%real"]


def test_exact_term_reads_are_chunked_to_safe_parameter_limit():
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "chunked")
    ensure_schema(connection, storage.dialect)
    generation = begin_generation(connection, storage.dialect, "chunked", ["body"], [])
    writer = storage.writer(generation)
    rows = [(f"term-{index}", index) for index in range(600)]
    writer.finalize_terms(rows)
    connection.commit()
    activate_generation(connection, storage.dialect, "chunked", generation)
    try:
        reader = storage.reader()
        statements = []
        connection.set_trace_callback(statements.append)
        assert len(reader.expand_terms([term for term, _ in rows])) == 600
    finally:
        connection.close()

    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 2


def test_reader_remains_pinned_when_a_new_generation_is_activated(populated_storage):
    pinned = populated_storage.reader()
    connection = populated_storage.conn
    generation = begin_generation(
        connection, populated_storage.dialect, "docs", ["body"], []
    )
    writer = populated_storage.writer(generation)
    writer.finalize_terms([("replacement", 0)])
    connection.commit()

    before_activation = populated_storage.reader()
    activate_generation(connection, populated_storage.dialect, "docs", generation)

    after_activation = populated_storage.reader()

    assert pinned.expand_terms(["green", "replacement"]) == ["green"]
    assert before_activation.expand_terms(["green", "replacement"]) == ["green"]
    assert after_activation.expand_terms(["green", "replacement"]) == ["replacement"]


def test_writer_rejects_non_building_or_foreign_generations(populated_storage):
    connection = populated_storage.conn
    active = connection.execute(
        "SELECT active_generation FROM lunr_v2_indexes WHERE index_name='docs'"
    ).fetchone()[0]
    foreign = begin_generation(
        connection, populated_storage.dialect, "other", ["body"], []
    )
    failed = begin_generation(
        connection, populated_storage.dialect, "docs", ["body"], []
    )
    fail_generation(connection, populated_storage.dialect, "docs", failed, "boom")

    for generation in (active, foreign, failed, "missing"):
        with pytest.raises(ValueError, match="building generation"):
            populated_storage.writer(generation)


def test_building_generation_does_not_override_active_reader(populated_storage):
    generation = begin_generation(
        populated_storage.conn, populated_storage.dialect, "docs", ["body"], []
    )
    populated_storage.writer(generation).finalize_terms([("invisible", 0)])

    assert populated_storage.reader().expand_terms(["green", "invisible"]) == ["green"]


def test_v2_writer_selection_clears_temporary_legacy_mode():
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "docs")
    storage.writer()
    ensure_schema(connection, storage.dialect)
    generation = begin_generation(connection, storage.dialect, "docs", ["body"], [])
    storage.writer(generation).finalize_terms([("native", 0)])
    connection.commit()
    activate_generation(connection, storage.dialect, "docs", generation)
    try:
        assert storage.reader().expand_terms(["native"]) == ["native"]
    finally:
        connection.close()


def test_posting_and_vector_reads_chunk_parameters(populated_storage):
    reader = populated_storage.reader()
    statements = []
    populated_storage.conn.set_trace_callback(statements.append)
    terms = ["green", *(f"missing-term-{index}" for index in range(599))]
    assert reader.get_postings(terms)["green"]["title"]["1"]
    posting_selects = [
        sql for sql in statements if sql.lstrip().upper().startswith("SELECT")
    ]
    assert len(posting_selects) == 2

    statements.clear()
    refs = ["body/1", *(f"missing-ref-{index}" for index in range(599))]
    assert "body/1" in reader.get_field_vectors(refs)
    vector_selects = [
        sql for sql in statements if sql.lstrip().upper().startswith("SELECT")
    ]
    assert len(vector_selects) == 2


class _NativeJsonCursor:
    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.closed = False

    def execute(self, sql, params):
        if self.error:
            raise self.error

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class _SingleCursorConnection:
    def __init__(self, cursor):
        self.value = cursor

    def cursor(self):
        return self.value


def test_reader_accepts_native_json_values():
    posting_cursor = _NativeJsonCursor(
        [("green", 0, "body", "1", {"position": [[0, 5]]})]
    )
    storage = SimpleNamespace(
        conn=_SingleCursorConnection(posting_cursor),
        index_name="docs",
        dialect=get_dialect("sqlite"),
    )
    reader = V2SqlIndexReader(storage, "generation")

    assert reader.get_postings(["green"])["green"]["body"]["1"] == {
        "position": [[0, 5]]
    }

    vector_cursor = _NativeJsonCursor([("body/1", [0, 2.5], 2.5)])
    vector_storage = SimpleNamespace(
        conn=_SingleCursorConnection(vector_cursor),
        index_name="docs",
        dialect=get_dialect("sqlite"),
    )
    reader = V2SqlIndexReader(vector_storage, "generation")
    assert list(reader.get_field_vectors(["body/1"])["body/1"]) == [0, 2.5]


def test_reader_closes_cursor_when_execute_fails():
    cursor = _NativeJsonCursor(error=RuntimeError("query failed"))
    storage = SqlStorage.from_conn(_SingleCursorConnection(cursor), "docs")
    reader = V2SqlIndexReader(storage, "generation")

    with pytest.raises(RuntimeError, match="query failed"):
        reader.get_postings(["green"])

    assert cursor.closed


def test_writer_closes_validation_cursor_when_execute_fails():
    cursor = _NativeJsonCursor(error=RuntimeError("query failed"))
    storage = SimpleNamespace(
        conn=_SingleCursorConnection(cursor),
        index_name="docs",
        dialect=get_dialect("sqlite"),
    )

    with pytest.raises(RuntimeError, match="query failed"):
        V2SqlIndexWriter(storage, "generation")

    assert cursor.closed
