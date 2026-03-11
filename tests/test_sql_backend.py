import importlib.util
import os
import sqlite3
import uuid

import pytest

from lunr import get_default_builder, lunr
from lunr.exceptions import BaseLunrException
from lunr.query import QueryPresence
from lunr.storage.sql import SqlStorage


@pytest.fixture
def sql_storage():
    return _sqlite_storage("test_idx")


def _sqlite_storage(index_name):
    return SqlStorage.from_conn(
        sqlite3.connect(":memory:"),
        index_name=index_name,
        dialect="sqlite",
    )


def _refs(idx, query):
    return [result["ref"] for result in idx.search(query)]


def _parse_mysql_user_and_db(raw_value):
    separators = (":", "/", ",")
    for separator in separators:
        if separator in raw_value:
            user, database = raw_value.split(separator, 1)
            user = user.strip()
            database = database.strip()
            if not user or not database:
                break
            return user, database

    value = raw_value.strip()
    if not value:
        raise ValueError("MAINDB must not be empty")
    return value, value


def _connect_mysql(host, user, password, database, port):
    if importlib.util.find_spec("pymysql") is not None:
        import pymysql

        return pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
        )

    if importlib.util.find_spec("MySQLdb") is not None:
        import MySQLdb

        return MySQLdb.connect(
            host=host,
            user=user,
            passwd=password,
            db=database,
            port=port,
        )

    raise ModuleNotFoundError("No MySQL Python driver installed (pymysql or MySQLdb)")


@pytest.fixture
def mysql_storage():
    user_and_db = os.getenv("MAINDB")
    password = os.getenv("PASSWDDB")

    if not user_and_db or password is None:
        pytest.skip("MySQL tests require MAINDB and PASSWDDB to be set")

    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    index_name = f"mysql_idx_{uuid.uuid4().hex}"

    try:
        user, database = _parse_mysql_user_and_db(user_and_db)
    except ValueError as exc:
        pytest.skip(f"Invalid MAINDB value: {exc}")

    try:
        conn = _connect_mysql(host, user, password, database, port)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))
    except Exception as exc:
        pytest.skip(f"Unable to connect to MySQL with local credentials: {exc}")

    storage = SqlStorage.from_conn(conn, index_name=index_name, dialect="mysql")

    cursor = None
    try:
        cursor = storage.conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
    except Exception as exc:
        pytest.skip(f"MySQL connection check failed: {exc}")
    finally:
        if cursor is not None:
            cursor.close()

    return storage


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
    mem_refs = _refs(mem_idx, query)
    sql_refs = _refs(sql_idx, query)

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
    single_storage = _sqlite_storage("single")
    parallel_storage = _sqlite_storage("parallel")

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
    assert _refs(parallel_idx, query) == _refs(single_idx, query)


def test_lunr_workers_kwarg_for_sql_storage(documents):
    storage = _sqlite_storage("workers")
    idx = lunr(
        ref="id",
        fields=("title", "body"),
        documents=documents,
        storage=storage,
        workers=2,
        parallel_backend="thread",
    )

    assert _refs(idx, "green study")


def test_lunr_workers_without_storage_warns_and_keeps_in_memory_behavior(documents):
    default_idx = lunr(ref="id", fields=("title", "body"), documents=documents)

    with pytest.warns(RuntimeWarning, match="requires a SQL storage backend"):
        workers_idx = lunr(
            ref="id",
            fields=("title", "body"),
            documents=documents,
            workers=2,
            parallel_backend="thread",
        )

    assert _refs(workers_idx, "green study") == _refs(default_idx, "green study")


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
    sql_storage = _sqlite_storage("pos")
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
    single_storage = _sqlite_storage("single-pos")
    single_idx = lunr(
        ref="id",
        fields=["id", "test"],
        documents=docs,
        builder=single_builder,
        storage=single_storage,
    )

    parallel_builder = get_default_builder()
    parallel_builder.metadata_whitelist.append("position")
    parallel_storage = _sqlite_storage("parallel-pos")
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
    assert _refs(parallel_idx, "hello") == _refs(single_idx, "hello")


def test_sql_backend_parallel_process_backend_matches_thread_backend(documents):
    process_storage = _sqlite_storage("process")
    thread_storage = _sqlite_storage("thread")

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

    assert _refs(process_idx, "green study") == _refs(thread_idx, "green study")


def test_sql_backend_process_backend_falls_back_when_payload_is_unpicklable():
    docs = [{"id": "1", "title": "hello", "body": "hello world"}]
    storage = _sqlite_storage("fallback")

    with pytest.warns(RuntimeWarning, match="falling back to 'thread'"):
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

    assert _refs(idx, "hello") == ["1"]


def test_sql_backend_incremental_flush_matches_standard(documents):
    standard_idx = _build_sql_index(documents, _sqlite_storage("standard"))

    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(_sqlite_storage("incremental"))
    builder.sql_flush(enabled=True, doc_batch_size=1, row_batch_size=2)
    for document in documents:
        builder.add(document)

    incremental_idx = builder.build()

    assert _refs(incremental_idx, "green study") == _refs(standard_idx, "green study")


def test_sql_backend_parallel_incremental_flush_matches_standard(documents):
    standard_idx = _build_sql_index(documents, _sqlite_storage("standard-parallel"))

    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(_sqlite_storage("incremental-parallel"))
    builder.parallel(workers=2, backend="thread")
    builder.sql_flush(enabled=True, doc_batch_size=1, row_batch_size=2)
    builder.sql_commit_every(docs=1, rows=2)
    for document in documents:
        builder.add(document)

    incremental_idx = builder.build()

    assert _refs(incremental_idx, "green study") == _refs(standard_idx, "green study")


# ---------------------------------------------------------------------------
# df_threshold tests
# ---------------------------------------------------------------------------


def test_df_threshold_standard_sql_build_removes_high_df_terms(documents):
    storage = _sqlite_storage("df-standard")
    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage)
    builder.df_threshold(3)
    for doc in documents:
        builder.add(doc)
    idx = builder.build()

    # "green" appears in all 3 docs (df=3) -> removed
    assert _refs(idx, "green") == []
    # "plant" has df=2, still searchable
    assert _refs(idx, "plant") != []


def test_df_threshold_incremental_sql_purges_from_database(documents):
    storage = _sqlite_storage("df-incremental")
    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage)
    builder.sql_flush(enabled=True, doc_batch_size=1, row_batch_size=2)
    builder.df_threshold(3)
    for doc in documents:
        builder.add(doc)
    idx = builder.build()

    assert _refs(idx, "green") == []
    assert _refs(idx, "plant") != []


def test_df_threshold_parallel_sql_build_removes_high_df_terms(documents):
    storage = _sqlite_storage("df-parallel")
    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage)
    builder.parallel(workers=2, backend="thread")
    builder.df_threshold(3)
    for doc in documents:
        builder.add(doc)
    idx = builder.build()

    assert _refs(idx, "green") == []
    assert _refs(idx, "plant") != []


def test_df_threshold_parallel_incremental_purges_from_database(documents):
    storage = _sqlite_storage("df-par-inc")
    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage)
    builder.parallel(workers=2, backend="thread")
    builder.sql_flush(enabled=True, doc_batch_size=1, row_batch_size=2)
    builder.df_threshold(3)
    for doc in documents:
        builder.add(doc)
    idx = builder.build()

    assert _refs(idx, "green") == []
    assert _refs(idx, "plant") != []


def test_df_threshold_no_effect_when_no_terms_exceed(documents):
    storage_no = _sqlite_storage("df-no-effect")
    storage_hi = _sqlite_storage("df-high")
    baseline = _build_sql_index(documents, storage_no)

    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage_hi)
    builder.df_threshold(100)
    for doc in documents:
        builder.add(doc)
    idx = builder.build()

    assert _refs(idx, "green study") == _refs(baseline, "green study")


def test_df_threshold_via_lunr_convenience(documents):
    storage = _sqlite_storage("df-lunr")
    idx = lunr(
        ref="id",
        fields=("title", "body"),
        documents=documents,
        storage=storage,
        df_threshold=3,
    )

    assert _refs(idx, "green") == []
    assert _refs(idx, "plant") != []


def test_df_threshold_removes_term_rows_from_sql_tables(documents):
    storage = _sqlite_storage("df-rows")
    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage)
    builder.sql_flush(enabled=True, doc_batch_size=10, row_batch_size=100)
    builder.df_threshold(3)
    for doc in documents:
        builder.add(doc)
    builder.build()

    c = storage.conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM lunr_terms WHERE index_name = ? AND term = ?",
        (storage.index_name, "green"),
    )
    assert c.fetchone()[0] == 0

    c.execute(
        "SELECT COUNT(*) FROM lunr_postings WHERE index_name = ? AND term = ?",
        (storage.index_name, "green"),
    )
    assert c.fetchone()[0] == 0

    c.execute(
        "SELECT COUNT(*) FROM lunr_term_frequencies WHERE index_name = ? AND term = ?",
        (storage.index_name, "green"),
    )
    assert c.fetchone()[0] == 0


def test_df_threshold_adjusts_doc_field_lengths_in_sql(documents):
    """After purge, lunr_doc_fields.length reflects only surviving terms."""
    storage = _sqlite_storage("df-lengths")
    builder = get_default_builder()
    builder.ref("id")
    builder.field("title")
    builder.field("body")
    builder.storage(storage)
    builder.sql_flush(enabled=True, doc_batch_size=10, row_batch_size=100)
    builder.df_threshold(3)
    for doc in documents:
        builder.add(doc)
    builder.build()

    c = storage.conn.cursor()
    # For every field_ref, stored length must equal sum of surviving tf values.
    c.execute(
        "SELECT d.field_ref, d.length, COALESCE(t.total, 0) "
        "FROM lunr_doc_fields d "
        "LEFT JOIN ("
        "  SELECT field_ref, SUM(tf) AS total "
        "  FROM lunr_term_frequencies WHERE index_name = ? GROUP BY field_ref"
        ") t ON d.field_ref = t.field_ref "
        "WHERE d.index_name = ?",
        (storage.index_name, storage.index_name),
    )
    for field_ref, stored_len, tf_sum in c.fetchall():
        assert stored_len == tf_sum, (
            f"{field_ref}: stored length {stored_len} != tf sum {tf_sum}"
        )


def test_df_threshold_vectors_agree_across_all_build_paths(documents):
    """Standard, parallel, incremental, and parallel-incremental must produce
    the same field vectors when the same df_threshold is applied."""
    import json

    def _build(label, parallel=False, incremental=False):
        st = _sqlite_storage(label)
        b = get_default_builder()
        b.ref("id")
        b.field("title")
        b.field("body")
        b.storage(st)
        b.df_threshold(3)
        if parallel:
            b.parallel(workers=2, backend="thread")
        if incremental:
            b.sql_flush(enabled=True, doc_batch_size=1, row_batch_size=2)
        for d in documents:
            b.add(d)
        b.build()
        # Read back all field vectors from DB so we can compare.
        c = st.conn.cursor()
        c.execute(
            "SELECT field_ref, elements FROM lunr_field_vectors "
            "WHERE index_name = ? ORDER BY field_ref",
            (st.index_name,),
        )
        return {fr: json.loads(elems) for fr, elems in c.fetchall()}

    standard = _build("std")
    parallel = _build("par", parallel=True)
    incremental = _build("inc", incremental=True)
    par_inc = _build("pi", parallel=True, incremental=True)

    assert standard == parallel
    assert standard == incremental
    assert standard == par_inc
def test_mysql_backend_matches_memory_for_positive_queries(documents, mysql_storage):
    mem_idx = lunr(ref="id", fields=("title", "body"), documents=documents)
    sql_idx = _build_sql_index(documents, mysql_storage)

    query = "green study"
    mem_refs = [result["ref"] for result in mem_idx.search(query)]
    sql_refs = [result["ref"] for result in sql_idx.search(query)]

    assert sql_refs == mem_refs


@pytest.mark.mysql
def test_mysql_backend_wildcard_expansion(documents, mysql_storage):
    idx = _build_sql_index(documents, mysql_storage)

    starts_with = {result["ref"] for result in idx.search("pl*")}
    ends_with = {result["ref"] for result in idx.search("*reen")}

    assert starts_with == {"b", "c"}
    assert ends_with == {"a", "b", "c"}


@pytest.mark.mysql
def test_mysql_backend_disables_prohibited_and_negated_queries(
    documents, mysql_storage
):
    idx = _build_sql_index(documents, mysql_storage)

    query = idx.create_query()
    query.term("green", presence=QueryPresence.PROHIBITED)
    query.term("study", presence=QueryPresence.OPTIONAL)
    with pytest.raises(BaseLunrException, match="Prohibited clauses"):
        idx.query(query)

    with pytest.raises(BaseLunrException, match="Negated queries"):
        idx.search("-green")


@pytest.mark.mysql
def test_mysql_backend_disables_edit_distance(documents, mysql_storage):
    idx = _build_sql_index(documents, mysql_storage)

    query = idx.create_query()
    query.term("gren", edit_distance=1)

    with pytest.raises(BaseLunrException, match="Edit distance"):
        idx.query(query)


@pytest.mark.mysql
def test_mysql_backend_not_serializable(documents, mysql_storage):
    idx = _build_sql_index(documents, mysql_storage)

    with pytest.raises(BaseLunrException, match="cannot be serialized"):
        idx.serialize()


@pytest.mark.mysql
def test_mysql_storage_uses_mysql_dialect(mysql_storage):
    assert mysql_storage.dialect.name == "mysql"
    assert mysql_storage.dialect.placeholder == "%s"
    assert mysql_storage.dialect.upsert == "duplicate"


@pytest.mark.parametrize("raw_value", ["user:db", "user/db", "user,db"])
def test_parse_mysql_user_and_db_supports_compound_values(raw_value):
    assert _parse_mysql_user_and_db(raw_value) == ("user", "db")


def test_parse_mysql_user_and_db_defaults_database_to_user():
    assert _parse_mysql_user_and_db("onlyvalue") == ("onlyvalue", "onlyvalue")


def test_parse_mysql_user_and_db_rejects_empty_value():
    with pytest.raises(ValueError, match="must not be empty"):
        _parse_mysql_user_and_db("   ")


def test_connect_mysql_requires_driver(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(ModuleNotFoundError, match="No MySQL Python driver"):
        _connect_mysql("127.0.0.1", "user", "pass", "db", 3306)
