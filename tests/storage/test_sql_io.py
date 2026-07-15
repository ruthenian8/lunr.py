import sqlite3

import pytest

from lunr.storage.sql import SqlStorage
from lunr.storage.sql.schema import activate_generation, begin_generation, ensure_schema


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
    writer.finalize_terms([("green", 0), ("study", 1), ("машина", 2)])
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
    statements = []
    populated_storage.conn.set_trace_callback(statements.append)

    reader = populated_storage.reader()
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
    generation = populated_storage._v2_generation
    writer = populated_storage.writer(generation)
    writer.finalize_terms([("100%real", 3), ("100_percent", 4), ("100xother", 5)])

    reader = populated_storage.reader()

    assert reader.expand_terms(["100%*"]) == ["100%real"]
    assert reader.expand_terms(["100_*"]) == ["100_percent"]


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
    statements = []
    connection.set_trace_callback(statements.append)
    try:
        assert len(storage.reader().expand_terms([term for term, _ in rows])) == 600
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
    activate_generation(connection, populated_storage.dialect, "docs", generation)

    reopened = SqlStorage.from_conn(connection, "docs").reader()

    assert pinned.expand_terms(["green", "replacement"]) == ["green"]
    assert reopened.expand_terms(["green", "replacement"]) == ["replacement"]
