# Versioned SQL Storage Redesign

## Objective

Replace the existing monolithic SQL backend with a versioned, generation-based
storage subsystem that keeps SQLite, PostgreSQL, and MySQL as first-class
backends. Existing V1 SQL indexes are not migrated or read; users must rebuild
them into the V2 schema.

The in-memory Lunr implementation remains unchanged. Existing public SQL entry
points remain available:

- `lunr(..., storage=storage, workers=..., parallel_backend=...)`
- `SqlStorage.from_url(...)`
- `SqlStorage.from_conn(...)`
- `build_or_rebuild_index(...)`
- `sql_lunr_index(...)`

## Architecture

Each logical index has an active generation recorded in V2 metadata. A rebuild
writes a separate, invisible generation, validates it, and then changes the
active-generation pointer in one transaction. Searches pin the active generation
when an `Index` is opened, so a concurrent rebuild cannot change the data used
mid-query. A failed rebuild never deletes or replaces the active generation.

SQL construction uses one pipeline for sequential, threaded, and process-based
operation:

```text
document iterator
    -> optional parallel tokenization
    -> batched staging rows in SQL
    -> SQL aggregation for document frequency and field statistics
    -> streamed vector construction
    -> generation validation
    -> atomic activation
```

This replaces the standard, parallel, incremental, and parallel-incremental SQL
Builder branches. Documents are processed from an iterator and are not retained
as a complete `_raw_documents` collection.

## V2 Schema

V2 uses new `lunr_v2_*` table names so schema creation cannot silently reuse V1
tables:

- `lunr_v2_indexes`: logical index name, schema version, active generation,
  fields, language configuration, and build metadata.
- `lunr_v2_generations`: generation identity and `building`, `ready`, `active`,
  or `failed` state.
- `lunr_v2_terms`: generation-scoped term and term index.
- `lunr_v2_postings`: generation-scoped term, field, document reference, and
  metadata.
- `lunr_v2_field_vectors`: generation-scoped field vector data.
- `lunr_v2_doc_fields`: generation-scoped build-time field lengths.
- `lunr_v2_term_frequencies`: generation-scoped build-time term frequencies.

Build-time rows may be removed after activation. Old inactive generations are
removed only after the new generation is active. Every search table key includes
the logical index and generation, permitting multiple indexes in one database.

Opening a database containing only V1 tables raises a clear rebuild-required
error. No automatic data migration is attempted.

## Component Boundaries

The existing `lunr/storage/sql.py` module becomes a `lunr/storage/sql/` package:

- `dialects.py` owns placeholders, DDL types, bulk-upsert syntax, wildcard
  escaping, and JSON adaptation for SQLite, PostgreSQL, and MySQL.
- `schema.py` owns V2 DDL and generation lifecycle operations.
- `writer.py` owns staging and persistent bulk writes.
- `reader.py` owns generation-pinned, batched search reads.
- `indexer.py` owns the single streaming build pipeline.
- `__init__.py` exposes the stable `SqlStorage` facade and compatibility imports.

Higher layers do not branch on backend names. The dialect adapter is the only
component that contains database-specific SQL.

The existing in-memory `Builder` continues to construct in-memory indexes. When
SQL storage is configured, it delegates document iteration and configuration to
the SQL indexer rather than selecting one of four internal build methods.

## Search Data Flow

An SQL reader pins the active generation at creation. Query execution resolves
all exact or wildcard terms once, loads their postings in bulk, computes matching
field references in Python using existing Lunr semantics, and loads all required
field vectors in one bulk operation.

The redesigned SQL mode retains its documented limitations: fuzzy/edit-distance
expansion, prohibited clauses, fully negated queries, and serialization remain
unsupported. Leading wildcard queries remain potentially expensive and are
documented accordingly.

Russian language configuration is stored in index metadata. Reopening an index
reconstructs the appropriate search pipeline without requiring callers to repeat
`languages=["ru"]`. If languages are explicitly supplied, they must match stored
configuration. Custom pipeline reopening requires an explicitly supplied
compatible pipeline or builder.

## Parallel and Streaming Indexing

Default language pipelines are reconstructed inside process workers from a
serializable language configuration. Generated closure functions are not sent to
workers. Custom extractors or pipelines that cannot be represented safely fall
back to threads with a `RuntimeWarning`.

The input document iterable is consumed incrementally. Tokenized document-field
records are written in configurable batches. Vocabulary maps and SQL staging data
may grow with corpus vocabulary, but the original documents and full posting tree
are not retained in memory.

`df_threshold` remains available for compatibility, is never hard-coded by Flask,
and rejects invalid values instead of silently clamping them. SQL aggregation
applies the threshold before terms and vectors are finalized.

## Failure and Transaction Semantics

Large builds may commit staging batches because the generation is not visible to
searches. Activation occurs in a short transaction after generation validation.
If indexing or vector finalization fails, the generation is marked `failed`, its
rows are cleaned up, and the previous active generation remains unchanged.

Connections passed to `SqlStorage.from_conn` remain caller-owned. Connections
created by `from_url` are storage-owned and gain explicit context-manager/close
behavior. Flask helpers acquire and close one raw connection per context.

JSON readers accept either encoded JSON strings/bytes or native lists/dicts
returned by database drivers. Writers use the dialect adapter so psycopg,
PyMySQL/mysqlclient, and SQLite receive supported parameter types.

SQLite URL semantics match the conventional forms:

- `sqlite:///:memory:` opens an in-memory database.
- `sqlite:///relative.db` opens a relative path.
- `sqlite:////absolute/path.db` opens an absolute path.

## Flask Integration

`build_or_rebuild_index` forwards its document iterable directly to the SQL
indexer. It does not delete active rows before a build. `sql_lunr_index` reads
fields and languages from V2 metadata and pins the active generation.

Search responses convert Lunr results to JSON-safe dictionaries containing
`ref`, `score`, and `match_data.metadata`. The example reindex endpoint streams
ORM rows, has no hard-coded DF threshold, and documents that production mutation
routes require authentication.

## Packaging and Documentation

The project adds optional dependency groups for Flask, PostgreSQL, MySQL, and a
combined SQL installation. `tqdm` is not a core dependency; progress reporting
uses standard logging or a caller callback.

Documentation includes the V1 rebuild requirement, URL semantics, supported SQL
query features, process/thread behavior, generation lifecycle, optional extras,
and PostgreSQL/MySQL configuration.

## Verification

Automated tests cover:

- V2 schema creation and explicit V1 rebuild-required errors.
- Atomic rebuild success and failure with the previous generation remaining live.
- Multiple logical indexes in one database.
- Relative, absolute, and in-memory SQLite URLs.
- Native and string JSON driver values.
- Batched term, posting, and vector reads with bounded statement counts.
- Generator-based streaming builds.
- Default and Russian process-worker configuration.
- Flask test-client search and rebuild round trips with JSON-safe results.
- Russian morphology through in-memory, SQLite, and Flask paths.
- PostgreSQL and MySQL integration suites using explicit connection URLs and CI
  service containers.
- Optional-extra import and packaging behavior.

PostgreSQL/MySQL tests skip only when their explicit integration connection URLs
are absent. SQLite tests always run.

## Compatibility

Python-level SQL entry points remain stable. V1 SQL data is intentionally
incompatible and must be rebuilt. In-memory indexes and Lunr.js serialization are
unaffected. SQL scoring and result ordering must retain parity with the in-memory
implementation for supported positive queries.
