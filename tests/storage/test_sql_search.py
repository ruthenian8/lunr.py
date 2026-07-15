import sqlite3

import pytest

from lunr import lunr
from lunr.exceptions import BaseLunrException
from lunr.storage.sql import SqlStorage
from lunr.stop_word_filter import stop_word_filter


@pytest.fixture
def indexes(documents):
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "search")
    sql = lunr("id", ("title", "body"), iter(documents), storage=storage)
    memory = lunr("id", ("title", "body"), documents)
    try:
        yield sql, memory
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("query", "expected_selects"),
    [("green study", 3), ("green st* pl*", 4)],
)
def test_positive_query_has_bounded_round_trips(indexes, query, expected_selects):
    sql_index, _ = indexes
    statements = []
    sql_index.storage_reader.conn.set_trace_callback(statements.append)

    assert [result["ref"] for result in sql_index.search(query)]

    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) == expected_selects


def test_sql_query_limitations_remain_explicit(indexes):
    sql_index, _ = indexes

    with pytest.raises(BaseLunrException, match="Prohibited"):
        sql_index.search("green -study")
    with pytest.raises(BaseLunrException, match="Negated"):
        sql_index.search("-green")
    with pytest.raises(BaseLunrException, match="Edit distance"):
        sql_index.query(lambda query: query.term("gren", edit_distance=1))


@pytest.mark.parametrize(
    "query",
    ["green study", "+green +study", "pl*", "title:green"],
)
def test_sql_query_results_and_scores_match_memory(indexes, query):
    sql_index, memory_index = indexes

    assert [
        (result["ref"], result["score"]) for result in sql_index.search(query)
    ] == [
        (result["ref"], result["score"]) for result in memory_index.search(query)
    ]


def test_sql_query_does_not_mutate_clause_terms(indexes):
    sql_index, _ = indexes
    query = sql_index.create_query()
    query.term("plants")
    original = query.clauses[0].term

    sql_index.query(query)

    assert query.clauses[0].term == original


def test_required_stop_word_returns_no_results(indexes):
    sql_index, memory_index = indexes
    sql_index.pipeline.add(stop_word_filter)
    memory_index.pipeline.add(stop_word_filter)

    assert sql_index.search("+the") == memory_index.search("+the") == []


def test_equal_scores_are_sorted_by_reference():
    documents = [
        {"id": "z", "body": "identical"},
        {"id": "a", "body": "identical"},
    ]
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(connection, "ties")
    try:
        sql_index = lunr("id", ("body",), iter(documents), storage=storage)
        memory_index = lunr("id", ("body",), documents)

        sql_results = sql_index.search("identical")
        memory_results = memory_index.search("identical")

        assert [result["ref"] for result in sql_results] == ["a", "z"]
        assert [
            (result["ref"], result["score"]) for result in sql_results
        ] == [
            (result["ref"], result["score"]) for result in memory_results
        ]
    finally:
        connection.close()
