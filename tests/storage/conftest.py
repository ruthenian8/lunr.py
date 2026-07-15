import os
import sqlite3
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from lunr import get_default_builder, lunr
from lunr.storage.sql import SqlStorage
from lunr.storage.sql.indexer import SqlIndexer
from lunr.storage.sql.schema import get_active_generation


_TABLES_IN_DELETE_ORDER = (
    "lunr_v2_postings",
    "lunr_v2_field_vectors",
    "lunr_v2_doc_fields",
    "lunr_v2_term_frequencies",
    "lunr_v2_terms",
    "lunr_v2_generations",
    "lunr_v2_indexes",
)


@dataclass
class BackendHarness:
    storage: SqlStorage
    url: str = None
    index_names: set = field(default_factory=set)

    def __post_init__(self):
        self.index_names.add(self.storage.index_name)

    def storage_for(self, suffix):
        index_name = f"{self.storage.index_name}_{suffix}"
        self.index_names.add(index_name)
        return SqlStorage(
            self.storage.conn,
            index_name,
            self.storage.dialect,
            owns_connection=False,
        )

    def reopened_storage(self):
        if self.url is not None:
            return SqlStorage.from_url(self.url, self.storage.index_name)
        return SqlStorage(
            self.storage.conn,
            self.storage.index_name,
            self.storage.dialect,
            owns_connection=False,
        )


def _cleanup(harness):
    connection = harness.storage.conn
    try:
        connection.rollback()
    except Exception:
        pass
    cursor = connection.cursor()
    try:
        for table in _TABLES_IN_DELETE_ORDER:
            for index_name in harness.index_names:
                cursor.execute(
                    f"DELETE FROM {table} WHERE index_name="
                    f"{harness.storage.dialect.placeholder}",
                    (index_name,),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def _backend_harness(environment_variable, backend):
    url = os.environ.get(environment_variable)
    if not url:
        pytest.skip(
            f"{environment_variable} is not set; skipping live {backend} contract"
        )
    index_name = f"lunr_t9_{backend}_{uuid4().hex[:12]}"
    storage = SqlStorage.from_url(url, index_name)
    harness = BackendHarness(storage, url=url)
    try:
        yield harness
    finally:
        try:
            _cleanup(harness)
        finally:
            storage.close()


@pytest.fixture
def postgresql_storage():
    yield from _backend_harness("LUNR_TEST_POSTGRESQL_URL", "postgresql")


@pytest.fixture
def mysql_storage():
    yield from _backend_harness("LUNR_TEST_MYSQL_URL", "mysql")


@pytest.fixture
def sqlite_backend_storage():
    connection = sqlite3.connect(":memory:")
    storage = SqlStorage.from_conn(
        connection, f"lunr_t9_sqlite_{uuid4().hex[:12]}"
    )
    harness = BackendHarness(storage)
    try:
        yield harness
    finally:
        try:
            _cleanup(harness)
        finally:
            connection.close()


@pytest.fixture
def backend_contract():
    return assert_backend_contract


@pytest.fixture
def backend_harness_factory():
    return _backend_harness


def _result_scores(index, query):
    return [(result["ref"], result["score"]) for result in index.search(query)]


def assert_backend_contract(harness, documents):
    primary = harness.storage
    contract_documents = list(documents) + [
        {
            "id": "literal",
            "title": "Literal patterns",
            "body": "100%real 100_percent bang!token",
        }
    ]
    memory = lunr("id", ("title", "body"), contract_documents)
    default_builder = get_default_builder()
    pinned = SqlIndexer(primary).build(
        iter(contract_documents),
        "id",
        [("title", 1, None), ("body", 1, None)],
        {"languages": []},
        ["position"],
        search_pipeline=default_builder.search_pipeline,
    )

    active = get_active_generation(primary.conn, primary.dialect, primary.index_name)
    assert active is not None
    assert active.fields == ["title", "body"]
    assert active.languages == []
    assert active.build_metadata == {"pipeline": "default"}
    assert _result_scores(pinned, "green study") == _result_scores(
        memory, "green study"
    )

    posting = primary.reader().get_postings(["green"])["green"]
    assert posting["title"]["a"]["position"]
    assert primary.reader().expand_terms("100%*") == ["100%real"]
    assert primary.reader().expand_terms("100_*") == ["100_percent"]
    assert primary.reader().expand_terms("bang!*") == ["bang!token"]

    fresh = harness.reopened_storage()
    try:
        assert fresh.open_index().fields == ["title", "body"]
        assert _result_scores(
            fresh.open_index(), "green study"
        ) == _result_scores(memory, "green study")
    finally:
        fresh.close()

    secondary = harness.storage_for("other")
    lunr(
        "id",
        ("title", "body"),
        [{"id": "other", "title": "Independent", "body": "unchanged"}],
        storage=secondary,
    )

    with pytest.raises(KeyError):
        lunr(
            "id",
            ("title", "body"),
            [{"id": "broken", "title": "missing body"}],
            storage=primary,
        )
    assert _result_scores(primary.open_index(), "green study") == _result_scores(
        memory, "green study"
    )

    replacement = [
        {"id": "new", "title": "Replacement", "body": "new generation"}
    ]
    current = lunr(
        "id", ("title", "body"), iter(replacement), storage=primary
    )
    assert [result["ref"] for result in current.search("replacement")] == ["new"]
    assert pinned.search("green")
    assert primary.open_index().search("green") == []
    assert [
        result["ref"] for result in secondary.open_index().search("unchanged")
    ] == ["other"]
