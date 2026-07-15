import sqlite3
import warnings

import pytest

from lunr import get_default_builder, lunr
from lunr.storage.sql import SqlStorage
from lunr.storage.sql.schema import get_active_generation


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
        assert [
            (result["ref"], result["score"])
            for result in sql.search("green study")
        ] == [
            (result["ref"], result["score"])
            for result in memory.search("green study")
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
