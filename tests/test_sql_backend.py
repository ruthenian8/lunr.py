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
