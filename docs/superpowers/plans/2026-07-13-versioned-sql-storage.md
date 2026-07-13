# Versioned SQL Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the V1 SQL backend with a streaming, generation-based V2 backend for SQLite, PostgreSQL, and MySQL while preserving the public Python API and Russian morphological search.

**Architecture:** SQL builds write an invisible generation through one streaming indexer, compute vectors from staged SQL rows, and atomically activate the completed generation. A dialect boundary owns all backend differences; a generation-pinned reader batches search data and the Flask adapter exposes JSON-safe results.

**Tech Stack:** Python DB-API 2.0, SQLite, psycopg/psycopg2, PyMySQL/mysqlclient, Flask-SQLAlchemy, pytest, pymorphy3.

## Global Constraints

- SQLite, PostgreSQL, and MySQL remain first-class supported backends.
- V1 SQL indexes are not migrated; opening V1-only storage raises a rebuild-required error.
- Existing top-level SQL and Flask Python entry points remain import-compatible.
- In-memory indexing and Lunr.js serialization behavior must not change.
- Supported SQL queries retain in-memory scoring and ordering parity.
- SQL mode continues to reject fuzzy, prohibited, fully negated, and serialized-index operations.
- Every behavior change follows red-green-refactor TDD.
- Preserve user-owned changes until the replacement behavior is covered by a failing test.

---

## File Structure

- Replace `lunr/storage/sql.py` with a package in one filesystem-safe change,
  temporarily preserving its implementation as `lunr/storage/sql/legacy.py`
  until V2 parity is complete.
- Create `lunr/storage/sql/__init__.py`: `SqlStorage` facade and public re-exports.
- Create `lunr/storage/sql/dialects.py`: URL parsing, DB connections, SQL syntax, and JSON adaptation.
- Create `lunr/storage/sql/schema.py`: V2 DDL and generation lifecycle.
- Create `lunr/storage/sql/writer.py`: batched staging writes.
- Create `lunr/storage/sql/reader.py`: generation-pinned batched search reads.
- Create `lunr/storage/sql/indexer.py`: single sequential/thread/process build pipeline.
- Modify `lunr/builder.py`: retain in-memory behavior and delegate SQL builds.
- Modify `lunr/index.py`: use reader bulk APIs without changing in-memory query behavior.
- Modify `lunr/lunr.py`: pass document iterables and language configuration to the SQL indexer.
- Modify `lunr/integrations/flask.py`: atomic rebuild helper, metadata-based reopen, and JSON-safe example responses.
- Modify `pyproject.toml`, `README.md`, `docs/indices.md`, `docs/usage.md`, and `docs/languages.md`: extras, V2 rebuild requirement, backend configuration, and operational behavior.
- Replace `tests/test_sql_backend.py` with focused V2 unit/integration coverage split across `tests/storage/`.
- Extend `tests/test_flask_integration.py` and `tests/test_language_support_russian.py`.

---

### Task 1: Dialect Boundary, URL Semantics, and JSON Adaptation

**Files:**
- Delete: `lunr/storage/sql.py`
- Create: `lunr/storage/sql/__init__.py`
- Create: `lunr/storage/sql/legacy.py` as an unchanged compatibility copy of the deleted module
- Create: `lunr/storage/sql/dialects.py`
- Create: `tests/storage/test_sql_dialects.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `SqlDialect`, `get_dialect(name)`, `connect_url(url)`, `json_dump(value)`, and `json_load(value)`.
- `connect_url(url: str) -> tuple[connection, SqlDialect]` owns connections it creates.
- `json_load` accepts `str`, `bytes`, `dict`, `list`, numeric values, and `None`.

- [ ] **Step 1: Write failing URL and JSON tests**

```python
def test_sqlite_url_forms(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    relative, dialect = connect_url("sqlite:///relative.db")
    assert dialect.name == "sqlite"
    relative.execute("CREATE TABLE marker (id INTEGER)")
    relative.close()
    assert (tmp_path / "relative.db").exists()

    memory, _ = connect_url("sqlite:///:memory:")
    assert memory.execute("PRAGMA database_list").fetchone()[2] == ""
    memory.close()

    absolute_path = tmp_path / "absolute.db"
    absolute, _ = connect_url(f"sqlite:////{str(absolute_path).lstrip('/')}")
    absolute.close()
    assert absolute_path.exists()


@pytest.mark.parametrize(
    "value, expected",
    [
        ('{"a": 1}', {"a": 1}),
        (b"[1, 2]", [1, 2]),
        ({"a": 1}, {"a": 1}),
        ([1, 2], [1, 2]),
        (None, None),
    ],
)
def test_json_load_accepts_driver_native_and_encoded_values(value, expected):
    assert json_load(value) == expected
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/storage/test_sql_dialects.py`

Expected: collection fails because `lunr.storage.sql.dialects` does not exist.

- [ ] **Step 3: Convert the module to a package without changing legacy behavior**

In one patch, copy the complete current `lunr/storage/sql.py` implementation to
`lunr/storage/sql/legacy.py`, delete `lunr/storage/sql.py`, and add:

```python
# lunr/storage/sql/__init__.py
from .legacy import (
    DIALECTS,
    SqlDialect,
    SqlFieldVectorsProxy,
    SqlIndexReader,
    SqlIndexWriter,
    SqlInvertedIndexProxy,
    SqlStorage,
)

__all__ = [
    "DIALECTS",
    "SqlDialect",
    "SqlFieldVectorsProxy",
    "SqlIndexReader",
    "SqlIndexWriter",
    "SqlInvertedIndexProxy",
    "SqlStorage",
]
```

Run: `pytest -q tests/test_sql_backend.py`

Expected: legacy SQL tests have the same result as before the file-to-package
conversion.

- [ ] **Step 4: Implement the dialect boundary**

```python
@dataclass(frozen=True)
class SqlDialect:
    name: str
    placeholder: str
    json_type: str
    real_type: str
    key_type: str

    def placeholders(self, count: int) -> str:
        return ", ".join([self.placeholder] * count)

    def upsert_sql(self, table, columns, keys, updates):
        values = self.placeholders(len(columns))
        base = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values})"
        if self.name == "sqlite":
            return base + f" ON CONFLICT ({', '.join(keys)}) DO UPDATE SET " + ", ".join(
                f"{column}=excluded.{column}" for column in updates
            )
        if self.name == "postgresql":
            return base + f" ON CONFLICT ({', '.join(keys)}) DO UPDATE SET " + ", ".join(
                f"{column}=EXCLUDED.{column}" for column in updates
            )
        return base + " ON DUPLICATE KEY UPDATE " + ", ".join(
            f"{column}=VALUES({column})" for column in updates
        )


def json_dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_load(value):
    if value is None or isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)
```

Implement `connect_url` using `urlsplit`: remove one leading slash for relative SQLite paths and retain the leading slash only when the URL has four slashes. Keep psycopg→psycopg2 and PyMySQL→MySQLdb driver fallbacks.

- [ ] **Step 5: Verify GREEN and package extras**

Run: `pytest -q tests/storage/test_sql_dialects.py`

Expected: all tests pass.

Add these exact extras to `pyproject.toml`:

```toml
flask = ["Flask", "Flask-SQLAlchemy"]
postgresql = ["psycopg[binary]>=3"]
mysql = ["PyMySQL>=1"]
sql = ["lunr[flask,postgresql,mysql]"]
```

- [ ] **Step 6: Commit**

```bash
git add lunr/storage/sql.py lunr/storage/sql tests/storage/test_sql_dialects.py pyproject.toml
git commit -m "feat: add portable SQL dialect boundary"
```

### Task 2: V2 Schema and Atomic Generation Lifecycle

**Files:**
- Create: `lunr/storage/sql/schema.py`
- Create: `tests/storage/test_sql_schema.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION = 2`, `ensure_schema`, `begin_generation`, `activate_generation`, `fail_generation`, `get_active_generation`, and `cleanup_generation`.
- Generation identifiers are UUID hex strings stored as text.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_failed_generation_preserves_active_generation(sqlite_connection):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    first = begin_generation(sqlite_connection, dialect, "docs", ["title"], ["ru"])
    activate_generation(sqlite_connection, dialect, "docs", first)
    second = begin_generation(sqlite_connection, dialect, "docs", ["title"], ["ru"])
    fail_generation(sqlite_connection, dialect, "docs", second, "tokenization failed")
    assert get_active_generation(sqlite_connection, dialect, "docs").generation == first


def test_two_indexes_have_independent_active_generations(sqlite_connection):
    dialect = get_dialect("sqlite")
    ensure_schema(sqlite_connection, dialect)
    docs = begin_generation(sqlite_connection, dialect, "docs", ["body"], [])
    users = begin_generation(sqlite_connection, dialect, "users", ["bio"], [])
    activate_generation(sqlite_connection, dialect, "docs", docs)
    activate_generation(sqlite_connection, dialect, "users", users)
    assert get_active_generation(sqlite_connection, dialect, "docs").generation == docs
    assert get_active_generation(sqlite_connection, dialect, "users").generation == users
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/storage/test_sql_schema.py`

Expected: import failure for the missing schema module.

- [ ] **Step 3: Implement V2 DDL and generation state transitions**

Create all seven `lunr_v2_*` tables from the approved design. Use composite keys containing `index_name` and `generation`. Implement activation as one transaction:

```python
def activate_generation(conn, dialect, index_name, generation):
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"UPDATE lunr_v2_generations SET state='ready' "
            f"WHERE index_name={dialect.placeholder} AND generation={dialect.placeholder}",
            (index_name, generation),
        )
        cursor.execute(
            dialect.upsert_sql(
                "lunr_v2_indexes",
                ["index_name", "schema_version", "active_generation"],
                ["index_name"],
                ["schema_version", "active_generation"],
            ),
            (index_name, SCHEMA_VERSION, generation),
        )
        cursor.execute(
            f"UPDATE lunr_v2_generations SET state='active' "
            f"WHERE index_name={dialect.placeholder} AND generation={dialect.placeholder}",
            (index_name, generation),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
```

Store fields and languages on the generation row before activation. Detect V1-only tables and raise `SqlRebuildRequiredError` when opening rather than building.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/storage/test_sql_schema.py`

Expected: all lifecycle and isolation tests pass.

- [ ] **Step 5: Commit**

```bash
git add lunr/storage/sql/schema.py tests/storage/test_sql_schema.py
git commit -m "feat: add atomic SQL index generations"
```

### Task 3: Stable SqlStorage Facade and V1 Rebuild Error

**Files:**
- Modify: `lunr/storage/sql/__init__.py`
- Create: `tests/storage/test_sql_storage.py`
- Modify: `lunr/storage/sql/legacy.py`

**Interfaces:**
- Produces: import-compatible `SqlStorage.from_url`, `SqlStorage.from_conn`, `writer`, `reader`, `close`, `__enter__`, and `__exit__`.
- `from_conn` never closes its caller-owned connection; `from_url` does.

- [ ] **Step 1: Write failing facade tests**

```python
def test_storage_context_closes_only_owned_connection(tmp_path):
    with SqlStorage.from_url("sqlite:///owned.db", "docs") as owned:
        conn = owned.conn
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")

    external = sqlite3.connect(":memory:")
    with SqlStorage.from_conn(external, "docs"):
        pass
    assert external.execute("SELECT 1").fetchone() == (1,)


def test_v1_only_database_requires_rebuild(v1_connection):
    storage = SqlStorage.from_conn(v1_connection, "docs")
    with pytest.raises(SqlRebuildRequiredError, match="rebuild"):
        storage.reader()
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/storage/test_sql_storage.py`

Expected: missing ownership behavior and rebuild exception.

- [ ] **Step 3: Replace legacy re-exports with the V2 package facade**

```python
class SqlStorage:
    def __init__(self, conn, index_name, dialect, owns_connection=False):
        self.conn = conn
        self.index_name = index_name
        self.dialect = get_dialect(dialect) if isinstance(dialect, str) else dialect
        self.owns_connection = owns_connection

    @classmethod
    def from_url(cls, url, index_name):
        conn, dialect = connect_url(url)
        return cls(conn, index_name, dialect, owns_connection=True)

    @classmethod
    def from_conn(cls, conn, index_name, dialect="sqlite"):
        return cls(conn, index_name, dialect, owns_connection=False)

    def close(self):
        if self.owns_connection:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
```

Re-export legacy reader/writer proxy names temporarily so all existing imports
continue to resolve while later tasks replace their implementations. Keep
`legacy.py` until Task 5 removes the old Builder paths.

- [ ] **Step 4: Verify GREEN and existing imports**

Run: `pytest -q tests/storage/test_sql_storage.py tests/test_sql_backend.py --collect-only`

Expected: facade tests pass and legacy tests collect.

- [ ] **Step 5: Commit**

```bash
git add lunr/storage/sql/__init__.py lunr/storage/sql/legacy.py tests/storage/test_sql_storage.py
git commit -m "refactor: replace SQL module with V2 facade"
```

### Task 4: Batched Writer and Generation-Pinned Reader

**Files:**
- Create: `lunr/storage/sql/writer.py`
- Create: `lunr/storage/sql/reader.py`
- Create: `tests/storage/test_sql_io.py`

**Interfaces:**
- Produces: `SqlIndexWriter.write_document_fields`, `write_term_frequencies`, `write_postings`, `finalize_terms`, and `write_vectors`.
- Produces: `SqlIndexReader.expand_terms`, `get_postings`, `get_field_vectors`, `iter_doc_fields`, and `iter_term_frequencies`.
- `get_postings(terms)` and `get_field_vectors(refs)` return mappings and issue batched `IN` queries.

- [ ] **Step 1: Write failing native-JSON and bounded-query tests**

```python
def test_writer_reader_json_roundtrip(populated_storage):
    reader = populated_storage.reader()
    posting = reader.get_postings(["машина"])["машина"]
    assert posting["body"]["1"] == {"position": [[0, 6]]}


def test_two_term_query_data_uses_three_selects(populated_storage):
    statements = []
    populated_storage.conn.set_trace_callback(statements.append)
    reader = populated_storage.reader()
    terms = reader.expand_terms(["green", "study"])
    postings = reader.get_postings(terms)
    refs = {f"{field}/{doc}" for posting in postings.values() for field in ("title", "body") for doc in posting.get(field, {})}
    reader.get_field_vectors(refs)
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) <= 3
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/storage/test_sql_io.py`

Expected: missing reader/writer APIs.

- [ ] **Step 3: Implement generic bulk writes and chunked reads**

Use a single writer helper:

```python
def _upsert_many(self, table, columns, keys, rows):
    if not rows:
        return
    updates = [column for column in columns if column not in keys]
    sql = self.dialect.upsert_sql(table, columns, keys, updates)
    cursor = self.conn.cursor()
    try:
        cursor.executemany(sql, rows)
    finally:
        cursor.close()
```

Implement reader chunking with a backend-safe maximum of 500 parameters per
chunk. `expand_terms` accepts all clause terms together, performs one exact `IN`
query for non-wildcard terms, and performs separately escaped `LIKE` queries only
for wildcard patterns. Decode every JSON value through `json_load`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/storage/test_sql_io.py`

Expected: all tests pass; the trace assertion reports at most three selects.

- [ ] **Step 5: Commit**

```bash
git add lunr/storage/sql/writer.py lunr/storage/sql/reader.py tests/storage/test_sql_io.py
git commit -m "feat: batch SQL index reads and writes"
```

### Task 5: Single Streaming SQL Indexer

**Files:**
- Create: `lunr/storage/sql/indexer.py`
- Create: `tests/storage/test_sql_indexer.py`
- Modify: `lunr/builder.py`
- Modify: `lunr/lunr.py`

**Interfaces:**
- Produces: `SqlIndexer.build(documents, ref, fields, pipeline_config, metadata_whitelist, workers, backend, batch_sizes, df_threshold) -> Index`.
- `Builder.add` remains unchanged for in-memory use; SQL construction is driven by an iterable passed to `SqlIndexer`.

- [ ] **Step 1: Write failing streaming, atomic-failure, and parity tests**

```python
def test_sql_build_consumes_generator_without_raw_document_buffer(sqlite_storage):
    consumed = []
    def documents():
        for i in range(50):
            consumed.append(i)
            yield {"id": str(i), "title": f"title {i}", "body": "общая машина"}
    idx = lunr("id", ("title", "body"), documents(), languages=["ru"], storage=sqlite_storage)
    assert len(consumed) == 50
    assert [hit["ref"] for hit in idx.search("машиной")]


def test_failed_rebuild_keeps_previous_results(sqlite_storage):
    first = lunr("id", ("title",), [{"id": "1", "title": "first"}], storage=sqlite_storage)
    with pytest.raises(KeyError):
        lunr("id", ("title",), [{"id": "2"}], storage=sqlite_storage)
    reopened = sqlite_storage.open_index()
    assert [hit["ref"] for hit in reopened.search("first")] == ["1"]


@pytest.mark.parametrize("backend", [None, "thread", "process"])
def test_sql_scoring_matches_memory(documents, backend):
    memory = lunr("id", ("title", "body"), documents)
    sql = lunr("id", ("title", "body"), iter(documents), storage=storage, workers=2 if backend else None, parallel_backend=backend or "thread")
    assert [(r["ref"], r["score"]) for r in sql.search("green study")] == [(r["ref"], r["score"]) for r in memory.search("green study")]
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/storage/test_sql_indexer.py`

Expected: generator path uses the old Builder buffer or V2 generation APIs are missing.

- [ ] **Step 3: Implement one document-record worker**

Create a module-level `_process_document(payload)` that reconstructs the default
pipeline from `languages`, tokenizes each configured field, and returns only
serializable field records:

```python
FieldRecord = namedtuple(
    "FieldRecord", "doc_ref field field_ref length term_counts metadata_by_term boost"
)
```

For custom extractors, test `pickle.dumps` on the worker configuration and fall
back to `ThreadPoolExecutor` with a `RuntimeWarning`. Use `executor.map` directly
over the input iterator; never materialize worker payloads.

- [ ] **Step 4: Implement staging, aggregation, vector finalization, and activation**

For each `FieldRecord`, batch-write `doc_fields`, `term_frequencies`, and
`postings`. After input exhaustion:

1. Aggregate distinct document frequency by term in SQL.
2. Exclude terms meeting `df_threshold`.
3. Assign deterministic term indexes ordered by term.
4. Aggregate average field lengths from surviving term frequencies.
5. Iterate term frequencies ordered by field reference and construct each
   `Vector` using the existing BM25 formula and rounding.
6. Validate that term and vector counts match generation metadata.
7. Activate the generation.
8. Clean build-time rows and the prior generation.

Wrap the flow in:

```python
generation = begin_generation(...)
try:
    self._stage_documents(...)
    self._finalize_generation(...)
    activate_generation(...)
except Exception as exc:
    fail_generation(..., str(exc))
    cleanup_generation(...)
    raise
```

- [ ] **Step 5: Delegate SQL builds and remove four Builder paths**

Change `lunr()` so SQL storage receives the document iterable directly. Preserve
the existing Builder route when `storage is None`. Remove `_build_sql_parallel`,
`_build_sql_incremental`, `_build_sql_parallel_incremental`, SQL batching helpers,
`_raw_documents`, debug `print`, and the undeclared `tqdm` import after parity
tests are green.

- [ ] **Step 6: Verify GREEN**

Run: `pytest -q tests/storage/test_sql_indexer.py tests/test_builder.py tests/test_search.py`

Expected: streaming, failure isolation, all backends, and in-memory regressions pass without process-fallback warnings for default or Russian pipelines.

- [ ] **Step 7: Commit**

```bash
git add lunr/storage/sql/indexer.py lunr/builder.py lunr/lunr.py tests/storage/test_sql_indexer.py
git commit -m "refactor: unify SQL indexing in streaming generations"
```

### Task 6: Batched Index Query Integration

**Files:**
- Modify: `lunr/index.py`
- Modify: `lunr/storage/sql/reader.py`
- Create: `tests/storage/test_sql_search.py`

**Interfaces:**
- Produces: `SqlIndexReader.prepare_query(clauses) -> QueryData` containing expanded terms, postings, and lazy bulk vector loading.
- In-memory `Index.query` remains on its existing TokenSet path.

- [ ] **Step 1: Write failing end-to-end query-count and feature tests**

```python
def test_positive_query_has_bounded_round_trips(sql_index):
    statements = []
    sql_index.storage_reader.conn.set_trace_callback(statements.append)
    assert [r["ref"] for r in sql_index.search("green study")]
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) <= 4


def test_sql_query_limitations_remain_explicit(sql_index):
    with pytest.raises(BaseLunrException, match="Prohibited"):
        sql_index.search("green -study")
    with pytest.raises(BaseLunrException, match="Negated"):
        sql_index.search("-green")
    with pytest.raises(BaseLunrException, match="Edit distance"):
        sql_index.query(lambda query: query.term("gren", edit_distance=1))
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/storage/test_sql_search.py`

Expected: query-count test exceeds the bound.

- [ ] **Step 3: Refactor only the SQL branch of `Index.query`**

Collect processed clause terms first, call `reader.expand_terms` and
`reader.get_postings` once, then execute existing required/optional matching in
Python. Collect matching field references and call `reader.get_field_vectors`
once before scoring. Do not mutate `Clause.term`; retain processed terms in local
variables.

- [ ] **Step 4: Verify GREEN and full query parity**

Run: `pytest -q tests/storage/test_sql_search.py tests/test_search.py tests/test_query.py tests/test_query_parser.py`

Expected: query count is bounded and all in-memory behavior still passes.

- [ ] **Step 5: Commit**

```bash
git add lunr/index.py lunr/storage/sql/reader.py tests/storage/test_sql_search.py
git commit -m "perf: batch SQL search reads"
```

### Task 7: Flask Rebuild and JSON-Safe Search

**Files:**
- Modify: `lunr/integrations/flask.py`
- Modify: `tests/test_flask_integration.py`

**Interfaces:**
- `build_or_rebuild_index` passes iterables to V2 indexing without deleting active rows.
- `sql_lunr_index` loads stored fields/languages and pins one generation.
- Produces: `serialize_search_result(result) -> dict`.

- [ ] **Step 1: Write failing Flask regressions**

```python
def test_rebuild_failure_preserves_searchable_index(db):
    build_or_rebuild_index(db, "site", [{"id": "1", "title": "first", "body": ""}])
    with pytest.raises(KeyError):
        build_or_rebuild_index(db, "site", [{"id": "2", "title": "missing body"}])
    with sql_lunr_index(db, "site") as index:
        assert [r["ref"] for r in index.search("first")] == ["1"]


def test_search_endpoint_returns_json_result(app):
    response = app.test_client().get("/search?q=машиной")
    assert response.status_code == 200
    assert response.get_json()[0].keys() == {"ref", "score", "match_data"}
    assert isinstance(response.get_json()[0]["match_data"], dict)


def test_rebuilding_one_index_does_not_delete_another(db):
    build_or_rebuild_index(db, "a", docs_a)
    build_or_rebuild_index(db, "b", docs_b)
    build_or_rebuild_index(db, "a", replacement_a)
    with sql_lunr_index(db, "b") as index:
        assert index.search("unchanged")
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_flask_integration.py`

Expected: current `TRUNCATE` fails on SQLite and JSON serialization raises.

- [ ] **Step 3: Implement minimal safe Flask adapter**

Remove all table deletion and debug output. Pass `documents` directly to the SQL
indexer. Store and reload languages through metadata. Add:

```python
def serialize_search_result(result):
    match_data = result.get("match_data")
    return {
        "ref": result["ref"],
        "score": result["score"],
        "match_data": match_data.metadata if match_data is not None else {},
    }
```

Use this function before `jsonify`. Replace `Document.query.all()` with
`Document.query.yield_per(doc_batch_size)` and remove the hard-coded threshold.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_flask_integration.py`

Expected: all Flask tests pass, including real test-client JSON serialization.

- [ ] **Step 5: Commit**

```bash
git add lunr/integrations/flask.py tests/test_flask_integration.py
git commit -m "fix: make Flask SQL rebuilds atomic and JSON safe"
```

### Task 8: Russian Metadata Reopening and Process Support

**Files:**
- Modify: `lunr/languages/__init__.py`
- Modify: `lunr/storage/sql/indexer.py`
- Modify: `tests/test_language_support_russian.py`

**Interfaces:**
- Stored language lists are normalized, ordered lists.
- Reopening without `languages` uses metadata; explicitly conflicting languages raise `BaseLunrException`.

- [ ] **Step 1: Write failing Russian integration tests**

```python
@pytest.mark.skipif(not HAS_PYMORPHY3, reason="pymorphy3 is not installed")
def test_sql_reopen_uses_stored_russian_pipeline(sqlite_storage):
    lunr("id", ("title", "body"), RUSSIAN_DOCS, languages=["ru"], storage=sqlite_storage)
    reopened = sqlite_storage.open_index()
    for query in ("машина", "машины", "машиной", "читать", "читала"):
        assert [r["ref"] for r in reopened.search(query)] == ["1"]


def test_reopen_rejects_conflicting_languages(sqlite_storage):
    lunr("id", ("body",), [{"id": "1", "body": "машина"}], languages=["ru"], storage=sqlite_storage)
    with pytest.raises(BaseLunrException, match="languages"):
        sqlite_storage.open_index(languages=["en"])
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_language_support_russian.py`

Expected: reopen lacks stored pipeline configuration.

- [ ] **Step 3: Implement language metadata reconstruction**

Normalize languages once in `lunr()` and persist them on the generation. In
`open_index`, call `get_default_builder(stored_languages).search_pipeline`.
Generate process workers from the stored language list rather than pickling
pipeline closures.

- [ ] **Step 4: Verify GREEN in an environment with pymorphy3**

Run: `pytest -q tests/test_language_support_russian.py tests/storage/test_sql_indexer.py -k 'russian or process'`

Expected: all selected tests pass without process fallback warnings.

- [ ] **Step 5: Commit**

```bash
git add lunr/languages/__init__.py lunr/storage/sql/indexer.py tests/test_language_support_russian.py
git commit -m "feat: persist Russian SQL search configuration"
```

### Task 9: PostgreSQL and MySQL First-Class Integration Suites

**Files:**
- Create: `tests/storage/conftest.py`
- Create: `tests/storage/test_sql_postgresql.py`
- Create: `tests/storage/test_sql_mysql.py`
- Modify: `.github/workflows/ci.yml` if present; otherwise create `.github/workflows/sql.yml`
- Modify: `setup.cfg`

**Interfaces:**
- Integration URLs: `LUNR_TEST_POSTGRESQL_URL` and `LUNR_TEST_MYSQL_URL`.
- Tests skip only when the corresponding URL is absent.

- [ ] **Step 1: Write backend contract tests**

Parameterize a shared contract covering schema creation, rebuild activation,
native JSON reads, wildcard escaping, positive-query parity, two-index isolation,
and failed-build preservation. Backend modules invoke the contract with
`SqlStorage.from_url(os.environ[...])` and unique index names.

```python
@pytest.mark.postgresql
def test_postgresql_contract(postgresql_storage, documents):
    assert_backend_contract(postgresql_storage, documents)


@pytest.mark.mysql
def test_mysql_contract(mysql_storage, documents):
    assert_backend_contract(mysql_storage, documents)
```

- [ ] **Step 2: Verify local skip behavior**

Run: `pytest -q tests/storage/test_sql_postgresql.py tests/storage/test_sql_mysql.py`

Expected without URLs: exactly two backend fixtures skip with explicit URL messages.

- [ ] **Step 3: Configure CI services**

Add PostgreSQL 16 and MySQL 8 services with health checks, pass both URLs to the
test job, and install `.[tests,sql]`. Register `postgresql` and `mysql` markers in
`setup.cfg`.

- [ ] **Step 4: Run each available backend contract**

Run:

```bash
pytest -q tests/storage/test_sql_postgresql.py
pytest -q tests/storage/test_sql_mysql.py
```

Expected: all contract tests pass when URLs are configured.

- [ ] **Step 5: Commit**

```bash
git add tests/storage/conftest.py tests/storage/test_sql_postgresql.py tests/storage/test_sql_mysql.py .github setup.cfg
git commit -m "test: verify PostgreSQL and MySQL storage contracts"
```

### Task 10: Documentation, Compatibility Cleanup, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/indices.md`
- Modify: `docs/usage.md`
- Modify: `docs/languages.md`
- Modify: `CHANGELOG.md`
- Modify/Delete: legacy SQL tests and compatibility code superseded by V2 coverage.

**Interfaces:**
- Documentation uses the supported extras and correct SQLite URL examples.
- Changelog explicitly states that V1 SQL indexes require rebuilding.

- [ ] **Step 1: Update documentation**

Document:

- `pip install lunr[sql]`, backend-specific extras, and Flask extra.
- V1 rebuild requirement and `lunr_v2_*` schema.
- Atomic generations and failed-build behavior.
- Correct relative/absolute SQLite forms.
- Streaming limitations: vocabulary and staging rows still scale with corpus.
- Bounded SQL search round trips and expensive leading wildcards.
- Process behavior for default/Russian versus custom pipelines.
- SQL query feature limitations.
- Authentication requirement for production reindex endpoints.

- [ ] **Step 2: Remove superseded compatibility code and tests**

Delete old writer/proxy implementations, four-path Builder helpers, environment-
specific MySQL credential parsing, debug output, and tests that assert V1 table
names. Retain public import aliases and explicit query-limitation tests.

- [ ] **Step 3: Run formatting and static checks**

Run:

```bash
black --check lunr tests
flake8 lunr tests
mypy lunr
git diff --check
```

Expected: all commands exit zero. If the repository's existing configuration
excludes files, use those configured exclusions without broadening them.

- [ ] **Step 4: Run focused verification**

Run:

```bash
pytest -q tests/storage tests/test_flask_integration.py tests/test_language_support_russian.py
```

Expected: all installed-dependency tests pass; only explicit missing-backend or
missing-pymorphy3 skips remain.

- [ ] **Step 5: Run the non-acceptance suite**

Run: `pytest -q -m "not acceptance"`

Expected: zero failures.

- [ ] **Step 6: Run acceptance tests with JavaScript dependencies installed**

Run:

```bash
npm ci --prefix tests/acceptance_tests/javascript
pytest -q -m acceptance
```

Expected: zero failures.

- [ ] **Step 7: Inspect final scope and commit**

Run: `git status --short && git diff --stat HEAD~10..HEAD`

Confirm only SQL storage, Builder delegation, Flask integration, Russian metadata,
packaging, tests, and documentation changed.

```bash
git add README.md docs CHANGELOG.md lunr tests pyproject.toml setup.cfg .github
git commit -m "docs: complete V2 SQL storage migration"
```
