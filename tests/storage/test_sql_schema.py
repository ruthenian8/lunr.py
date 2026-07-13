import sqlite3

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
