import sqlite3
import warnings
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from lunr import get_default_builder, lunr
from lunr.storage.sql import SqlStorage
from lunr.storage.sql.indexer import SqlIndexer, _bounded_map
from lunr.storage.sql.schema import get_active_generation
from lunr.storage.sql.writer import SqlIndexWriter


@pytest.fixture
def sqlite_storage():
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "docs")
    try:
        yield storage
    finally:
        connection.close()


def test_sql_build_consumes_generator_once_without_raw_document_buffer(sqlite_storage):
    consumed = []

    def documents():
        for index in range(50):
            consumed.append(index)
            yield {
                "id": str(index),
                "title": f"title {index}",
                "body": "common machine",
            }

    index = lunr("id", ("title", "body"), documents(), storage=sqlite_storage)

    assert consumed == list(range(50))
    assert [hit["ref"] for hit in index.search("machine")]
    assert get_active_generation(
        sqlite_storage.conn, sqlite_storage.dialect, "docs"
    ) is not None


def test_failed_rebuild_keeps_previous_results(sqlite_storage):
    lunr("id", ("title",), [{"id": "1", "title": "first"}], storage=sqlite_storage)

    with pytest.raises(KeyError):
        lunr("id", ("title",), [{"id": "2"}], storage=sqlite_storage)

    reopened = sqlite_storage.open_index()
    assert [hit["ref"] for hit in reopened.search("first")] == ["1"]


@pytest.mark.parametrize("backend", [None, "thread", "process"])
def test_sql_scoring_matches_memory(documents, backend):
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "parity")
    try:
        memory = lunr("id", ("title", "body"), documents)
        with warnings.catch_warnings(record=True) as caught:
            sql = lunr(
                "id",
                ("title", "body"),
                iter(documents),
                storage=storage,
                workers=2 if backend else None,
                parallel_backend=backend or "thread",
            )
        assert not [
            warning for warning in caught if "falling back" in str(warning.message)
        ]
        assert [result["ref"] for result in sql.search("green study")] == [
            result["ref"] for result in memory.search("green study")
        ]
    finally:
        connection.close()


def test_sql_build_preserves_custom_builder_scoring_and_extractor(sqlite_storage):
    documents = [
        {"id": "1", "nested": {"text": "green green"}},
        {"id": "2", "nested": {"text": "green"}},
    ]

    def configure(builder):
        builder.b(0.2)
        builder.k1(2.0)
        return builder

    memory_builder = configure(get_default_builder())
    sql_builder = configure(get_default_builder())
    field = {
        "field_name": "title",
        "boost": 3,
        "extractor": lambda document: document["nested"]["text"],
    }
    memory = lunr("id", (field,), documents, builder=memory_builder)
    sql = lunr(
        "id",
        (field,),
        iter(documents),
        builder=sql_builder,
        storage=sqlite_storage,
        workers=2,
        parallel_backend="thread",
    )

    assert [(result["ref"], result["score"]) for result in sql.search("green")] == [
        (result["ref"], result["score"]) for result in memory.search("green")
    ]


def test_custom_builder_preconfigured_storage_and_threshold_are_preserved(
    sqlite_storage,
):
    builder = get_default_builder()
    builder.storage(sqlite_storage)
    builder.df_threshold(2)

    index = lunr(
        "id",
        ("title",),
        iter(
            [
                {"id": "1", "title": "common first"},
                {"id": "2", "title": "common second"},
            ]
        ),
        builder=builder,
    )

    assert index.search("common") == []
    assert [hit["ref"] for hit in index.search("first")] == ["1"]


def test_russian_process_workers_reconstruct_default_pipeline(sqlite_storage):
    pytest.importorskip("pymorphy3")
    documents = (
        {"id": str(index), "body": "общая машина"} for index in range(4)
    )

    with warnings.catch_warnings(record=True) as caught:
        index = lunr(
            "id",
            ("body",),
            documents,
            languages=["ru"],
            storage=sqlite_storage,
            workers=2,
            parallel_backend="process",
        )

    assert [hit["ref"] for hit in index.search("машиной")]
    assert not [warning for warning in caught if "falling back" in str(warning.message)]


def test_parallel_submission_is_bounded():
    submitted = []

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class Executor:
        def submit(self, function, value):
            submitted.append(value)
            return Future(function(value))

    results = _bounded_map(Executor(), lambda value: value * 2, iter(range(20)), 3)
    assert next(results) == 0
    assert submitted == [0, 1, 2]
    assert list(results) == [value * 2 for value in range(1, 20)]


def test_cleanup_failure_after_activation_keeps_new_index_active(
    sqlite_storage, monkeypatch
):
    lunr("id", ("title",), [{"id": "1", "title": "old"}], storage=sqlite_storage)

    def fail_cleanup(*args, **kwargs):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr("lunr.storage.sql.indexer.cleanup_generation", fail_cleanup)
    with pytest.warns(RuntimeWarning, match="activated.*cleanup failed"):
        index = lunr(
            "id", ("title",), [{"id": "2", "title": "new"}], storage=sqlite_storage
        )

    assert [hit["ref"] for hit in index.search("new")] == ["2"]
    assert sqlite_storage.open_index().search("old") == []


def test_empty_vectors_flush_in_bounded_batches(sqlite_storage, monkeypatch):
    sizes = []
    original = SqlIndexWriter.write_vectors

    def recording_write_vectors(self, rows):
        sizes.append(len(rows))
        return original(self, rows)

    monkeypatch.setattr(SqlIndexWriter, "write_vectors", recording_write_vectors)
    builder = get_default_builder()
    builder.sql_flush(row_batch_size=5)
    lunr(
        "id",
        ("body",),
        ({"id": str(index), "body": ""} for index in range(23)),
        builder=builder,
        storage=sqlite_storage,
    )

    assert len(sizes) >= 5
    assert max(sizes) <= 5


def test_string_language_is_stored_as_one_language(sqlite_storage):
    builder = get_default_builder()
    indexer = SqlIndexer(sqlite_storage)
    indexer.build(
        [{"id": "1", "body": "green"}],
        "id",
        [("body", 1, None)],
        {"pipeline": builder.pipeline, "languages": "ru"},
        [],
        search_pipeline=builder.search_pipeline,
    )

    assert get_active_generation(
        sqlite_storage.conn, sqlite_storage.dialect, "docs"
    ).languages == ["ru"]


def test_average_lengths_casts_decimal_values_to_float():
    class Cursor:
        def execute(self, sql, params):
            pass

        def fetchall(self):
            return [("body", Decimal("1.25"))]

        def close(self):
            pass

    connection = SimpleNamespace(cursor=lambda: Cursor())
    storage = SimpleNamespace(
        conn=connection,
        dialect=SimpleNamespace(placeholder="?"),
        index_name="docs",
    )

    assert SqlIndexer(storage)._average_lengths("generation") == {"body": 1.25}


def test_invalid_backend_is_rejected(sqlite_storage):
    with pytest.raises(ValueError, match="backend"):
        lunr(
            "id",
            ("body",),
            [{"id": "1", "body": "green"}],
            storage=sqlite_storage,
            parallel_backend="invalid",
        )


def test_direct_sql_builder_build_is_explicitly_rejected(sqlite_storage):
    builder = get_default_builder()
    builder.storage(sqlite_storage)
    builder.ref("id")
    builder.field("body")
    builder.add({"id": "1", "body": "green"})

    with pytest.raises(RuntimeError, match="lunr"):
        builder.build()


def test_bm25_document_frequency_counts_docs_not_field_postings(sqlite_storage):
    lunr(
        "id",
        ("title", "body"),
        [
            {"id": "1", "title": "common", "body": "common"},
            {"id": "2", "title": "common", "body": "other"},
        ],
        storage=sqlite_storage,
    )
    generation = sqlite_storage.reader().generation
    term_index = sqlite_storage.conn.execute(
        "SELECT term_index FROM lunr_v2_terms WHERE index_name=? "
        "AND generation=? AND term='common'",
        ("docs", generation),
    ).fetchone()[0]
    elements = json.loads(
        sqlite_storage.conn.execute(
            "SELECT elements FROM lunr_v2_field_vectors WHERE index_name=? "
            "AND generation=? AND field_ref='title/1'",
            ("docs", generation),
        ).fetchone()[0]
    )

    assert elements[elements.index(term_index) + 1] == 0.182
