from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

from .dialects import json_dump, json_load


SCHEMA_VERSION = 2

_V1_TABLES = (
    "lunr_terms",
    "lunr_postings",
    "lunr_field_vectors",
    "lunr_doc_fields",
    "lunr_term_frequencies",
)


class SqlRebuildRequiredError(RuntimeError):
    """Raised when a database contains only the incompatible V1 schema."""


@dataclass(frozen=True)
class Generation:
    index_name: str
    generation: str
    state: str
    fields: List[str]
    languages: List[str]
    build_metadata: dict
    error: Optional[str]


def ensure_schema(conn, dialect) -> None:
    """Create the complete V2 schema without modifying V1 tables."""
    key = dialect.key_type
    index_name = dialect.index_name_type
    generation = dialect.generation_type
    term = dialect.term_type
    field = dialect.field_name_type
    reference = dialect.reference_type
    json_type = dialect.json_type
    real = dialect.real_type
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_v2_indexes (
                index_name {index_name} NOT NULL,
                schema_version INTEGER NOT NULL,
                active_generation {generation},
                fields {json_type},
                languages {json_type},
                build_metadata {json_type},
                PRIMARY KEY (index_name)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_v2_generations (
                index_name {index_name} NOT NULL,
                generation {generation} NOT NULL,
                state {key} NOT NULL,
                fields {json_type} NOT NULL,
                languages {json_type} NOT NULL,
                error TEXT,
                document_count INTEGER NOT NULL DEFAULT 0,
                term_count INTEGER NOT NULL DEFAULT 0,
                vector_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (index_name, generation)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_v2_terms (
                index_name {index_name} NOT NULL,
                generation {generation} NOT NULL,
                term {term} NOT NULL,
                term_index INTEGER NOT NULL,
                PRIMARY KEY (index_name, generation, term)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_v2_postings (
                index_name {index_name} NOT NULL,
                generation {generation} NOT NULL,
                term {term} NOT NULL,
                field {field} NOT NULL,
                doc_ref {reference} NOT NULL,
                metadata {json_type} NOT NULL,
                PRIMARY KEY (index_name, generation, term, field, doc_ref)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_v2_field_vectors (
                index_name {index_name} NOT NULL,
                generation {generation} NOT NULL,
                field_ref {reference} NOT NULL,
                field {field} NOT NULL,
                doc_ref {reference} NOT NULL,
                elements {json_type} NOT NULL,
                magnitude {real} NOT NULL,
                PRIMARY KEY (index_name, generation, field_ref)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_v2_doc_fields (
                index_name {index_name} NOT NULL,
                generation {generation} NOT NULL,
                field_ref {reference} NOT NULL,
                field {field} NOT NULL,
                doc_ref {reference} NOT NULL,
                length INTEGER NOT NULL,
                boost {real} NOT NULL DEFAULT 1,
                PRIMARY KEY (index_name, generation, field_ref)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS lunr_v2_term_frequencies (
                index_name {index_name} NOT NULL,
                generation {generation} NOT NULL,
                field_ref {reference} NOT NULL,
                term {term} NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (index_name, generation, field_ref, term)
            )
            """
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def begin_generation(conn, dialect, index_name, fields, languages) -> str:
    """Create an invisible generation in the building state."""
    generation = uuid4().hex
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO lunr_v2_generations "
            "(index_name, generation, state, fields, languages) "
            f"VALUES ({dialect.placeholders(5)})",
            (
                index_name,
                generation,
                "building",
                json_dump(list(fields)),
                json_dump(list(languages)),
            ),
        )
        conn.commit()
        return generation
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def activate_generation(
    conn, dialect, index_name, generation, build_metadata=None
) -> None:
    """Atomically make a completed generation visible to new readers."""
    placeholder = dialect.placeholder
    cursor = conn.cursor()
    try:
        _lock_logical_index(cursor, dialect, index_name)
        cursor.execute(
            "UPDATE lunr_v2_generations SET state='ready' "
            f"WHERE index_name={placeholder} AND generation={placeholder} "
            "AND state='building'",
            (index_name, generation),
        )
        if cursor.rowcount != 1:
            raise ValueError("Generation is missing or is not building")
        cursor.execute(
            "UPDATE lunr_v2_generations SET state='ready' "
            f"WHERE index_name={placeholder} AND state='active'",
            (index_name,),
        )
        cursor.execute(
            "UPDATE lunr_v2_indexes SET "
            f"schema_version={placeholder}, active_generation={placeholder}, "
            f"build_metadata={placeholder} "
            f"WHERE index_name={placeholder}",
            (
                SCHEMA_VERSION,
                generation,
                json_dump(build_metadata or {}),
                index_name,
            ),
        )
        cursor.execute(
            "UPDATE lunr_v2_generations SET state='active' "
            f"WHERE index_name={placeholder} AND generation={placeholder}",
            (index_name, generation),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def fail_generation(conn, dialect, index_name, generation, error) -> None:
    """Mark a non-active generation failed without changing the active pointer."""
    placeholder = dialect.placeholder
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE lunr_v2_generations SET state='failed', "
            f"error={placeholder} WHERE index_name={placeholder} "
            f"AND generation={placeholder} AND state<>'active'",
            (str(error), index_name, generation),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def get_active_generation(conn, dialect, index_name) -> Optional[Generation]:
    """Return stored metadata for the active generation, if there is one."""
    if not _table_exists(conn, dialect, "lunr_v2_indexes"):
        if any(_table_exists(conn, dialect, table) for table in _V1_TABLES):
            raise SqlRebuildRequiredError(
                "This database contains a V1 Lunr SQL index; rebuild it for V2"
            )
        return None

    placeholder = dialect.placeholder
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT g.index_name, g.generation, g.state, g.fields, "
            "g.languages, i.build_metadata, g.error FROM lunr_v2_indexes i "
            "JOIN lunr_v2_generations g ON "
            "g.index_name=i.index_name AND g.generation=i.active_generation "
            f"WHERE i.index_name={placeholder} AND i.schema_version={placeholder}",
            (index_name, SCHEMA_VERSION),
        )
        row = cursor.fetchone()
    finally:
        cursor.close()
    if row is None:
        return None
    return Generation(
        index_name=row[0],
        generation=row[1],
        state=row[2],
        fields=list(json_load(row[3])),
        languages=list(json_load(row[4])),
        build_metadata=dict(json_load(row[5]) or {}),
        error=row[6],
    )


def cleanup_generation(conn, dialect, index_name, generation) -> None:
    """Delete all rows belonging to one inactive generation."""
    placeholder = dialect.placeholder
    cursor = conn.cursor()
    try:
        active_generation = _lock_logical_index(cursor, dialect, index_name)
        if active_generation == generation:
            raise ValueError("Cannot clean up the active generation")
        for table in (
            "lunr_v2_postings",
            "lunr_v2_field_vectors",
            "lunr_v2_doc_fields",
            "lunr_v2_term_frequencies",
            "lunr_v2_terms",
            "lunr_v2_generations",
        ):
            cursor.execute(
                f"DELETE FROM {table} WHERE index_name={placeholder} "
                f"AND generation={placeholder}",
                (index_name, generation),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def _lock_logical_index(cursor, dialect, index_name):
    """Create and lock one logical-index row for a lifecycle transaction."""
    if dialect.begin_write_sql is not None:
        cursor.execute(dialect.begin_write_sql)
    cursor.execute(
        dialect.upsert_sql(
            "lunr_v2_indexes",
            ["index_name", "schema_version", "active_generation"],
            ["index_name"],
            ["schema_version"],
        ),
        (index_name, SCHEMA_VERSION, None),
    )
    cursor.execute(
        "SELECT active_generation FROM lunr_v2_indexes "
        f"WHERE index_name={dialect.placeholder}{dialect.row_lock_suffix}",
        (index_name,),
    )
    row = cursor.fetchone()
    return None if row is None else row[0]


def _table_exists(conn, dialect, table_name) -> bool:
    """Probe a fixed internal table name without leaving a failed transaction."""
    cursor = conn.cursor()
    try:
        cursor.execute("SAVEPOINT lunr_v2_schema_probe")
        try:
            cursor.execute(f"SELECT 1 FROM {table_name} WHERE 1=0")
        except Exception as error:
            cursor.execute("ROLLBACK TO SAVEPOINT lunr_v2_schema_probe")
            cursor.execute("RELEASE SAVEPOINT lunr_v2_schema_probe")
            if dialect.is_undefined_table_error(error):
                return False
            raise
        else:
            cursor.execute("RELEASE SAVEPOINT lunr_v2_schema_probe")
            return True
    finally:
        cursor.close()
