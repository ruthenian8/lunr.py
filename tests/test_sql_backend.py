import sqlite3

import pytest

from lunr import get_default_builder, lunr
from lunr.exceptions import BaseLunrException
from lunr.query import QueryPresence
from lunr.storage.sql import SqlStorage


@pytest.fixture
def sql_storage():
    conn = sqlite3.connect(":memory:")
    return SqlStorage.from_conn(conn, index_name="test_idx", dialect="sqlite")


def _build_sql_index(documents, storage):
    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage)
    for document in documents:
        builder.add(document)
    return builder.build()


def test_sql_backend_matches_memory_for_positive_queries(documents, sql_storage):
    mem_idx = lunr(ref="id", fields=("title", "body"), documents=documents)
    sql_idx = _build_sql_index(documents, sql_storage)

    query = "green study"
    mem_refs = [result["ref"] for result in mem_idx.search(query)]
    sql_refs = [result["ref"] for result in sql_idx.search(query)]

    assert sql_refs == mem_refs


def test_sql_backend_wildcard_expansion(documents, sql_storage):
    idx = _build_sql_index(documents, sql_storage)

    starts_with = {result["ref"] for result in idx.search("pl*")}
    ends_with = {result["ref"] for result in idx.search("*reen")}

    assert starts_with == {"b", "c"}
    assert ends_with == {"a", "b", "c"}


def test_sql_backend_disables_prohibited_and_negated_queries(documents, sql_storage):
    idx = _build_sql_index(documents, sql_storage)

    query = idx.create_query()
    query.term("green", presence=QueryPresence.PROHIBITED)
    query.term("study", presence=QueryPresence.OPTIONAL)
    with pytest.raises(BaseLunrException, match="Prohibited clauses"):
        idx.query(query)

    with pytest.raises(BaseLunrException, match="Negated queries"):
        idx.search("-green")


def test_sql_backend_disables_edit_distance(documents, sql_storage):
    idx = _build_sql_index(documents, sql_storage)

    query = idx.create_query()
    query.term("gren", edit_distance=1)

    with pytest.raises(BaseLunrException, match="Edit distance"):
        idx.query(query)


def test_sql_backend_not_serializable(documents, sql_storage):
    idx = _build_sql_index(documents, sql_storage)

    with pytest.raises(BaseLunrException, match="cannot be serialized"):
        idx.serialize()


def test_sql_backend_parallel_matches_single_worker(documents):
    single_storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="single",
        dialect="sqlite",
    )
    parallel_storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="parallel",
        dialect="sqlite",
    )

    single_idx = _build_sql_index(documents, single_storage)

    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(parallel_storage)
    builder.parallel(workers=4, backend="thread")
    for document in documents:
        builder.add(document)
    parallel_idx = builder.build()

    query = "green study"
    assert [result["ref"] for result in parallel_idx.search(query)] == [
        result["ref"] for result in single_idx.search(query)
    ]


def test_lunr_workers_kwarg_for_sql_storage(documents):
    storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="workers",
        dialect="sqlite",
    )
    idx = lunr(
        ref="id",
        fields=("title", "body"),
        documents=documents,
        storage=storage,
        workers=2,
        parallel_backend="thread",
    )

    assert [result["ref"] for result in idx.search("green study")]


def test_sql_backend_positions_metadata_matches_in_memory():
    docs = [
        {"id": "1", "test": "hello world hello"},
        {"id": "2", "test": "world hello"},
    ]

    memory_builder = get_default_builder()
    memory_builder.metadata_whitelist.append("position")
    memory_idx = lunr(
        ref="id",
        fields=["id", "test"],
        documents=docs,
        builder=memory_builder,
    )

    sql_builder = get_default_builder()
    sql_builder.metadata_whitelist.append("position")
    sql_storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="pos",
        dialect="sqlite",
    )
    sql_idx = lunr(
        ref="id",
        fields=["id", "test"],
        documents=docs,
        builder=sql_builder,
        storage=sql_storage,
    )

    mem_posting = memory_idx.inverted_index["hello"]
    sql_posting = sql_idx.inverted_index["hello"]

    assert sql_posting["test"] == mem_posting["test"]
    assert sql_posting["test"]["1"]["position"] == [[0, 5], [12, 5]]


def test_sql_backend_parallel_positions_metadata_parity_with_single_worker():
    docs = [
        {"id": "1", "test": "hello world hello"},
        {"id": "2", "test": "world hello"},
        {"id": "3", "test": "hello hello hello"},
    ]

    single_builder = get_default_builder()
    single_builder.metadata_whitelist.append("position")
    single_storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="single-pos",
        dialect="sqlite",
    )
    single_idx = lunr(
        ref="id",
        fields=["id", "test"],
        documents=docs,
        builder=single_builder,
        storage=single_storage,
    )

    parallel_builder = get_default_builder()
    parallel_builder.metadata_whitelist.append("position")
    parallel_storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="parallel-pos",
        dialect="sqlite",
    )
    parallel_idx = lunr(
        ref="id",
        fields=["id", "test"],
        documents=docs,
        builder=parallel_builder,
        storage=parallel_storage,
        workers=4,
        parallel_backend="thread",
    )

    assert (
        parallel_idx.inverted_index["hello"]["test"]
        == single_idx.inverted_index["hello"]["test"]
    )
    assert [result["ref"] for result in parallel_idx.search("hello")] == [
        result["ref"] for result in single_idx.search("hello")
    ]


def test_sql_backend_parallel_process_backend_matches_thread_backend(documents):
    process_storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="process",
        dialect="sqlite",
    )
    thread_storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="thread",
        dialect="sqlite",
    )

    process_idx = lunr(
        ref="id",
        fields=("title", "body"),
        documents=documents,
        storage=process_storage,
        workers=2,
        parallel_backend="process",
    )
    thread_idx = lunr(
        ref="id",
        fields=("title", "body"),
        documents=documents,
        storage=thread_storage,
        workers=2,
        parallel_backend="thread",
    )

    assert [result["ref"] for result in process_idx.search("green study")] == [
        result["ref"] for result in thread_idx.search("green study")
    ]


def test_sql_backend_process_backend_falls_back_when_payload_is_unpicklable():
    docs = [{"id": "1", "title": "hello", "body": "hello world"}]
    storage = SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name="fallback",
        dialect="sqlite",
    )

    idx = lunr(
        ref="id",
        fields=[
            "id",
            {"field_name": "title", "extractor": lambda doc: doc["title"]},
            "body",
        ],
        documents=docs,
        storage=storage,
        workers=2,
        parallel_backend="process",
    )

    assert [result["ref"] for result in idx.search("hello")] == ["1"]
