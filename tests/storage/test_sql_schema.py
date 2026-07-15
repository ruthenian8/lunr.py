import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from lunr.storage.sql.dialects import get_dialect, json_load
from lunr.storage.sql.schema import (
    SCHEMA_VERSION,
    SqlRebuildRequiredError,
    activate_generation,
    begin_generation,
    cleanup_generation,
    ensure_schema,
    fail_generation,
    get_active_generation,
)


@pytest.fixture
def sqlite_connection():
    connection = sqlite3.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


class RecordingCursor:
    def __init__(self, statements):
        self.statements = statements

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    def close(self):
        pass


class RecordingConnection:
    def __init__(self):
        self.statements = []

    def cursor(self):
        return RecordingCursor(self.statements)

    def commit(self):
        pass

    def rollback(self):
        pass


def test_ensure_schema_creates_complete_v2_schema(sqlite_connection):
    ensure_schema(sqlite_connection, get_dialect("sqlite"))

    tables = {
        row[0]
        for row in sqlite_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'lunr_v2_%'"
        )
    }
    assert tables == {
        "lunr_v2_indexes",
        "lunr_v2_generations",
        "lunr_v2_terms",
        "lunr_v2_postings",
        "lunr_v2_field_vectors",
        "lunr_v2_doc_fields",
        "lunr_v2_term_frequencies",
    }

    for table in tables - {"lunr_v2_indexes"}:
        primary_key = {
            row[1]
            for row in sqlite_connection.execute(f"PRAGMA table_info({table})")
            if row[5]
        }
        assert {"index_name", "generation"} <= primary_key


def test_sqlite_schema_columns_types_constraints_and_defaults(sqlite_connection):
    ensure_schema(sqlite_connection, get_dialect("sqlite"))
    expected = {
        "lunr_v2_indexes": [
            ("index_name", "TEXT", 1, None, 1),
            ("schema_version", "INTEGER", 1, None, 0),
            ("active_generation", "TEXT", 0, None, 0),
            ("fields", "TEXT", 0, None, 0),
            ("languages", "TEXT", 0, None, 0),
            ("build_metadata", "TEXT", 0, None, 0),
        ],
        "lunr_v2_generations": [
            ("index_name", "TEXT", 1, None, 1),
            ("generation", "TEXT", 1, None, 2),
            ("state", "TEXT", 1, None, 0),
            ("fields", "TEXT", 1, None, 0),
            ("languages", "TEXT", 1, None, 0),
            ("error", "TEXT", 0, None, 0),
            ("document_count", "INTEGER", 1, "0", 0),
            ("term_count", "INTEGER", 1, "0", 0),
            ("vector_count", "INTEGER", 1, "0", 0),
        ],
        "lunr_v2_terms": [
            ("index_name", "TEXT", 1, None, 1),
            ("generation", "TEXT", 1, None, 2),
            ("term", "TEXT", 1, None, 3),
            ("term_index", "INTEGER", 1, None, 0),
        ],
        "lunr_v2_postings": [
            ("index_name", "TEXT", 1, None, 1),
            ("generation", "TEXT", 1, None, 2),
            ("term", "TEXT", 1, None, 3),
            ("field", "TEXT", 1, None, 4),
            ("doc_ref", "TEXT", 1, None, 5),
            ("metadata", "TEXT", 1, None, 0),
        ],
        "lunr_v2_field_vectors": [
            ("index_name", "TEXT", 1, None, 1),
            ("generation", "TEXT", 1, None, 2),
            ("field_ref", "TEXT", 1, None, 3),
            ("field", "TEXT", 1, None, 0),
            ("doc_ref", "TEXT", 1, None, 0),
            ("elements", "TEXT", 1, None, 0),
            ("magnitude", "REAL", 1, None, 0),
        ],
        "lunr_v2_doc_fields": [
            ("index_name", "TEXT", 1, None, 1),
            ("generation", "TEXT", 1, None, 2),
            ("field_ref", "TEXT", 1, None, 3),
            ("field", "TEXT", 1, None, 0),
            ("doc_ref", "TEXT", 1, None, 0),
            ("length", "INTEGER", 1, None, 0),
            ("boost", "REAL", 1, "1", 0),
        ],
        "lunr_v2_term_frequencies": [
            ("index_name", "TEXT", 1, None, 1),
            ("generation", "TEXT", 1, None, 2),
            ("field_ref", "TEXT", 1, None, 3),
            ("term", "TEXT", 1, None, 4),
            ("tf", "INTEGER", 1, None, 0),
        ],
    }
    actual = {
        table: [
            (row[1], row[2], row[3], row[4], row[5])
            for row in sqlite_connection.execute(f"PRAGMA table_info({table})")
        ]
        for table in expected
    }
    assert actual == expected


def test_mysql_postings_primary_key_fits_utf8mb4_index_limit():
    connection = RecordingConnection()

    ensure_schema(connection, get_dialect("mysql"))

    posting_ddl = next(
        sql for sql, _ in connection.statements if "lunr_v2_postings" in sql
    )
    expected_widths = {
        "index_name": 64,
        "generation": 32,
        "term": 255,
        "field": 64,
        "doc_ref": 255,
    }
    actual_widths = {
        column: int(re.search(rf"\b{column}\s+VARCHAR\((\d+)\)", posting_ddl).group(1))
        for column in expected_widths
    }
    assert actual_widths == expected_widths
    assert sum(actual_widths.values()) * 4 <= 3072


def test_begin_generation_stores_build_metadata(sqlite_connection):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)

    generation = begin_generation(
        sqlite_connection, dialect, "docs", ["title", "body"], ["en", "ru"]
    )

    assert len(generation) == 32
    int(generation, 16)
    row = sqlite_connection.execute(
        "SELECT state, fields, languages FROM lunr_v2_generations "
        "WHERE index_name=? AND generation=?",
        ("docs", generation),
    ).fetchone()
    assert row[0] == "building"
    assert json_load(row[1]) == ["title", "body"]
    assert json_load(row[2]) == ["en", "ru"]


def test_failed_generation_preserves_active_generation(sqlite_connection):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    first = begin_generation(sqlite_connection, dialect, "docs", ["title"], ["ru"])
    activate_generation(sqlite_connection, dialect, "docs", first)
    second = begin_generation(sqlite_connection, dialect, "docs", ["title"], ["ru"])
    fail_generation(sqlite_connection, dialect, "docs", second, "tokenization failed")

    active = get_active_generation(sqlite_connection, dialect, "docs")
    assert active.generation == first
    assert active.fields == ["title"]
    assert active.languages == ["ru"]
    assert sqlite_connection.execute(
        "SELECT state, error FROM lunr_v2_generations "
        "WHERE index_name=? AND generation=?",
        ("docs", second),
    ).fetchone() == ("failed", "tokenization failed")


def test_two_indexes_have_independent_active_generations(sqlite_connection):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    docs = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    users = begin_generation(sqlite_connection, dialect, "users", ["bio"], [])
    activate_generation(sqlite_connection, dialect, "docs", docs)
    activate_generation(sqlite_connection, dialect, "users", users)

    assert get_active_generation(sqlite_connection, dialect, "docs").generation == docs
    assert (
        get_active_generation(sqlite_connection, dialect, "users").generation == users
    )


def test_successful_replacement_leaves_exactly_one_active_generation(
    sqlite_connection,
):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    first = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    second = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    activate_generation(sqlite_connection, dialect, "docs", first)

    activate_generation(sqlite_connection, dialect, "docs", second)

    assert sqlite_connection.execute(
        "SELECT generation FROM lunr_v2_generations "
        "WHERE index_name=? AND state='active'",
        ("docs",),
    ).fetchall() == [(second,)]
    assert (
        get_active_generation(sqlite_connection, dialect, "docs").generation == second
    )


def test_activation_locks_logical_index_before_state_changes(sqlite_connection):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    generation = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    statements = []
    sqlite_connection.set_trace_callback(statements.append)

    activate_generation(sqlite_connection, dialect, "docs", generation)

    normalized = [" ".join(statement.split()) for statement in statements]
    begin_position = normalized.index("BEGIN IMMEDIATE")
    lock_position = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("SELECT active_generation FROM lunr_v2_indexes")
    )
    state_position = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("UPDATE lunr_v2_generations SET state='ready'")
    )
    assert begin_position < lock_position < state_position


def test_concurrent_sqlite_activations_leave_one_active_generation(tmp_path):
    database = str(tmp_path / "activation.db")
    setup = sqlite3.connect(database)
    dialect = get_dialect("sqlite")
    ensure_schema(setup, dialect)
    first = begin_generation(setup, dialect, "docs", ["body"], [])
    second = begin_generation(setup, dialect, "docs", ["body"], [])
    setup.close()
    barrier = threading.Barrier(2)

    def activate(generation):
        connection = sqlite3.connect(database, timeout=5)
        try:
            barrier.wait()
            activate_generation(connection, dialect, "docs", generation)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(activate, (first, second)))

    check = sqlite3.connect(database)
    try:
        pointer = get_active_generation(check, dialect, "docs").generation
        active = check.execute(
            "SELECT generation FROM lunr_v2_generations "
            "WHERE index_name=? AND state='active'",
            ("docs",),
        ).fetchall()
        assert active == [(pointer,)]
    finally:
        check.close()


def test_activation_rolls_back_all_state_when_final_transition_fails(
    sqlite_connection,
):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    generation = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    sqlite_connection.execute(
        "CREATE TRIGGER reject_active BEFORE UPDATE OF state ON lunr_v2_generations "
        "WHEN NEW.state='active' BEGIN SELECT RAISE(ABORT, 'reject active'); END"
    )
    sqlite_connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="reject active"):
        activate_generation(sqlite_connection, dialect, "docs", generation)

    assert sqlite_connection.execute(
        "SELECT state FROM lunr_v2_generations WHERE index_name=? AND generation=?",
        ("docs", generation),
    ).fetchone() == ("building",)
    assert sqlite_connection.execute(
        "SELECT COUNT(*) FROM lunr_v2_indexes WHERE index_name=?", ("docs",)
    ).fetchone() == (0,)


def test_cleanup_removes_only_the_selected_generation(sqlite_connection):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    keep = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    remove = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    sqlite_connection.execute(
        "INSERT INTO lunr_v2_terms "
        "(index_name, generation, term, term_index) VALUES (?, ?, ?, ?)",
        ("docs", remove, "discard", 0),
    )
    sqlite_connection.commit()

    cleanup_generation(sqlite_connection, dialect, "docs", remove)

    remaining = sqlite_connection.execute(
        "SELECT generation FROM lunr_v2_generations WHERE index_name=?",
        ("docs",),
    ).fetchall()
    assert remaining == [(keep,)]
    assert sqlite_connection.execute(
        "SELECT COUNT(*) FROM lunr_v2_terms WHERE index_name=? AND generation=?",
        ("docs", remove),
    ).fetchone() == (0,)


def test_cleanup_locks_logical_index_before_pointer_check_and_delete(
    sqlite_connection,
):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    generation = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    statements = []
    sqlite_connection.set_trace_callback(statements.append)

    cleanup_generation(sqlite_connection, dialect, "docs", generation)

    normalized = [" ".join(statement.split()) for statement in statements]
    begin_position = normalized.index("BEGIN IMMEDIATE")
    lock_position = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("SELECT active_generation FROM lunr_v2_indexes")
    )
    delete_position = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("DELETE FROM lunr_v2_postings")
    )
    assert begin_position < lock_position < delete_position


@pytest.mark.parametrize(
    "legacy_table",
    [
        "lunr_terms",
        "lunr_postings",
        "lunr_field_vectors",
        "lunr_doc_fields",
        "lunr_term_frequencies",
    ],
)
def test_v1_only_schema_requires_an_explicit_rebuild(sqlite_connection, legacy_table):
    sqlite_connection.execute(f"CREATE TABLE {legacy_table} (marker INTEGER)")
    sqlite_connection.commit()

    with pytest.raises(SqlRebuildRequiredError, match="rebuild"):
        get_active_generation(sqlite_connection, get_dialect("sqlite"), "docs")


def test_fresh_schema_has_no_active_generation(sqlite_connection):
    ensure_schema(sqlite_connection, get_dialect("sqlite"))

    assert (
        get_active_generation(sqlite_connection, get_dialect("sqlite"), "missing")
        is None
    )
    assert SCHEMA_VERSION == 2


def test_active_generation_probe_propagates_sqlite_permission_errors(
    sqlite_connection,
):
    sqlite_connection.set_authorizer(
        lambda action, *_: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_SELECT
            else sqlite3.SQLITE_OK
        )
    )

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        get_active_generation(sqlite_connection, get_dialect("sqlite"), "docs")
