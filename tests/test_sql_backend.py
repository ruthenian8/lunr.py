import sqlite3

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


def _build_sql_index(documents, storage):
    return lunr(
        ref="id",
        fields=("title", "body"),
        documents=iter(documents),
        storage=storage,
    )


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
    builder.parallel(workers=4, backend="thread")
    parallel_idx = lunr(
        "id",
        ("title", "body"),
        iter(documents),
        builder=builder,
        storage=parallel_storage,
    )

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
    builder.sql_flush(row_batch_size=2)
    incremental_idx = lunr(
        "id",
        ("title", "body"),
        iter(documents),
        builder=builder,
        storage=_sqlite_storage("incremental"),
    )

    assert _refs(incremental_idx, "green study") == _refs(standard_idx, "green study")


def test_sql_backend_parallel_incremental_flush_matches_standard(documents):
    standard_idx = _build_sql_index(documents, _sqlite_storage("standard-parallel"))

    builder = get_default_builder()
    builder.parallel(workers=2, backend="thread")
    builder.sql_flush(row_batch_size=2)
    incremental_idx = lunr(
        "id",
        ("title", "body"),
        iter(documents),
        builder=builder,
        storage=_sqlite_storage("incremental-parallel"),
    )

    assert _refs(incremental_idx, "green study") == _refs(standard_idx, "green study")


# ---------------------------------------------------------------------------
# df_threshold tests
# ---------------------------------------------------------------------------


def test_df_threshold_standard_sql_build_removes_high_df_terms(documents):
    storage = _sqlite_storage("df-standard")
    builder = get_default_builder()
    builder.df_threshold(3)
    idx = lunr(
        "id", ("title", "body"), iter(documents), builder=builder, storage=storage
    )

    # "green" appears in all 3 docs (df=3) -> removed
    assert _refs(idx, "green") == []
    # "plant" has df=2, still searchable
    assert _refs(idx, "plant") != []


def test_df_threshold_incremental_sql_purges_from_database(documents):
    storage = _sqlite_storage("df-incremental")
    builder = get_default_builder()
    builder.sql_flush(row_batch_size=2)
    builder.df_threshold(3)
    idx = lunr(
        "id", ("title", "body"), iter(documents), builder=builder, storage=storage
    )

    assert _refs(idx, "green") == []
    assert _refs(idx, "plant") != []


def test_df_threshold_parallel_sql_build_removes_high_df_terms(documents):
    storage = _sqlite_storage("df-parallel")
    builder = get_default_builder()
    builder.parallel(workers=2, backend="thread")
    builder.df_threshold(3)
    idx = lunr(
        "id", ("title", "body"), iter(documents), builder=builder, storage=storage
    )

    assert _refs(idx, "green") == []
    assert _refs(idx, "plant") != []


def test_df_threshold_parallel_incremental_purges_from_database(documents):
    storage = _sqlite_storage("df-par-inc")
    builder = get_default_builder()
    builder.parallel(workers=2, backend="thread")
    builder.sql_flush(row_batch_size=2)
    builder.df_threshold(3)
    idx = lunr(
        "id", ("title", "body"), iter(documents), builder=builder, storage=storage
    )

    assert _refs(idx, "green") == []
    assert _refs(idx, "plant") != []


def test_df_threshold_no_effect_when_no_terms_exceed(documents):
    storage_no = _sqlite_storage("df-no-effect")
    storage_hi = _sqlite_storage("df-high")
    baseline = _build_sql_index(documents, storage_no)

    builder = get_default_builder()
    builder.df_threshold(100)
    idx = lunr(
        "id",
        ("title", "body"),
        iter(documents),
        builder=builder,
        storage=storage_hi,
    )

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
    lunr("id", ("title", "body"), iter(documents), storage=storage, df_threshold=3)
    generation = storage.reader().generation

    c = storage.conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM lunr_v2_terms "
        "WHERE index_name = ? AND generation = ? AND term = ?",
        (storage.index_name, generation, "green"),
    )
    assert c.fetchone()[0] == 0

    c.execute(
        "SELECT COUNT(*) FROM lunr_v2_postings "
        "WHERE index_name = ? AND generation = ? AND term = ?",
        (storage.index_name, generation, "green"),
    )
    assert c.fetchone()[0] == 0

    c.execute(
        "SELECT COUNT(*) FROM lunr_v2_term_frequencies "
        "WHERE index_name = ? AND generation = ? AND term = ?",
        (storage.index_name, generation, "green"),
    )
    assert c.fetchone()[0] == 0


def test_sql_indexer_cleans_build_time_rows_after_activation(documents):
    storage = _sqlite_storage("df-lengths")
    lunr("id", ("title", "body"), iter(documents), storage=storage, df_threshold=3)
    generation = storage.reader().generation

    c = storage.conn.cursor()
    for table in ("lunr_v2_doc_fields", "lunr_v2_term_frequencies"):
        c.execute(
            f"SELECT COUNT(*) FROM {table} WHERE index_name=? AND generation=?",
            (storage.index_name, generation),
        )
        assert c.fetchone()[0] == 0


def test_df_threshold_vectors_agree_across_all_build_paths(documents):
    """Sequential, thread, and process indexers produce identical vectors."""
    import json

    def _build(label, backend=None):
        st = _sqlite_storage(label)
        lunr(
            "id",
            ("title", "body"),
            iter(documents),
            storage=st,
            df_threshold=3,
            workers=2 if backend else None,
            parallel_backend=backend or "thread",
        )
        generation = st.reader().generation
        c = st.conn.cursor()
        c.execute(
            "SELECT field_ref, elements FROM lunr_v2_field_vectors "
            "WHERE index_name = ? AND generation = ? ORDER BY field_ref",
            (st.index_name, generation),
        )
        return {fr: json.loads(elems) for fr, elems in c.fetchall()}

    standard = _build("std")
    thread = _build("thread", backend="thread")
    process = _build("process", backend="process")

    assert standard == thread
    assert standard == process


def test_df_threshold_creates_empty_vectors_for_fully_purged_fields():
    docs = [
        {"id": "a", "title": "common", "body": "unique alpha text"},
        {"id": "b", "title": "common", "body": "unique beta text"},
        {"id": "c", "title": "common", "body": "unique gamma text"},
    ]

    def _build(label):
        st = _sqlite_storage(label)
        lunr("id", ("title", "body"), iter(docs), storage=st, df_threshold=3)
        generation = st.reader().generation
        c = st.conn.cursor()
        c.execute(
            "SELECT field_ref, elements FROM lunr_v2_field_vectors "
            "WHERE index_name=? AND generation=? AND field_ref LIKE 'title/%'",
            (st.index_name, generation),
        )
        return c.fetchall()

    for field_ref, elements in _build("zero"):
        assert elements == "[]", field_ref


def test_sql_backend_cursors_are_closed_after_operations(documents):
    """Verify that database cursors created during index and search operations
    are properly closed, preventing resource leaks."""
    raw_conn = sqlite3.connect(":memory:")
    cursors_created = []

    class _TrackingCursor:
        def __init__(self, real_cursor):
            self._real = real_cursor
            self.closed = False
            cursors_created.append(self)

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            self.closed = True
            self._real.close()

    class _TrackingConnection:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def cursor(self):
            return _TrackingCursor(self._conn.cursor())

        def commit(self):
            self._conn.commit()

    tracking_conn = _TrackingConnection(raw_conn)

    storage = SqlStorage.from_conn(
        tracking_conn, index_name="cursor_test", dialect="sqlite"
    )
    idx = _build_sql_index(documents, storage)

    # Cursors created during indexing should all be closed
    indexing_cursors = list(cursors_created)
    assert len(indexing_cursors) > 0
    assert all(
        c.closed for c in indexing_cursors
    ), "Some cursors created during indexing were not closed"

    cursors_created.clear()
    _refs(idx, "green study")

    # Cursors created during search should all be closed
    search_cursors = list(cursors_created)
    assert len(search_cursors) > 0
    assert all(
        c.closed for c in search_cursors
    ), "Some cursors created during search were not closed"
