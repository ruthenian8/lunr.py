from __future__ import annotations

import math
import pickle
import warnings
from collections import Counter, defaultdict, deque, namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from lunr.field_ref import FieldRef
from lunr.languages import normalize_languages
from lunr.tokenizer import Tokenizer
from lunr.vector import Vector

from .schema import (
    activate_generation,
    begin_generation,
    cleanup_generation,
    ensure_schema,
    fail_generation,
)


FieldRecord = namedtuple(
    "FieldRecord",
    "doc_ref field field_ref length term_counts metadata_by_term boost",
)


def _bounded_map(executor, function, iterable, max_in_flight):
    """Yield ordered results without eagerly submitting the whole iterable."""
    iterator = iter(iterable)
    pending = deque()
    for _ in range(max_in_flight):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def _process_document(payload):
    document_item, ref, fields, pipeline_config, metadata_whitelist = payload
    if isinstance(document_item, (tuple, list)):
        document, attributes = document_item
    else:
        document, attributes = document_item, {}
    doc_ref = str(document[ref])
    pipeline = pipeline_config.get("pipeline")
    if pipeline is None:
        from lunr.lunr import get_default_builder

        pipeline = get_default_builder(
            pipeline_config.get("languages") or None
        ).pipeline
    records = []
    for field_name, field_boost, extractor in fields:
        value = document[field_name] if extractor is None else extractor(document)
        terms = pipeline.run(Tokenizer(value), field_name)
        term_counts = Counter(str(term) for term in terms)
        metadata = defaultdict(lambda: defaultdict(list))
        for term in terms:
            for key in metadata_whitelist:
                metadata[str(term)][key].append(term.metadata[key])
        records.append(
            FieldRecord(
                doc_ref,
                field_name,
                str(FieldRef(doc_ref, field_name)),
                len(terms),
                dict(term_counts),
                {
                    term: {key: list(values) for key, values in values_by_key.items()}
                    for term, values_by_key in metadata.items()
                },
                field_boost * (attributes or {}).get("boost", 1),
            )
        )
    return records


class SqlIndexer:
    def __init__(self, storage, b=0.75, k1=1.2) -> None:
        self.storage = storage
        self.conn = storage.conn
        self.dialect = storage.dialect
        self.index_name = storage.index_name
        self.b = b
        self.k1 = k1

    def build(
        self,
        documents,
        ref,
        fields,
        pipeline_config,
        metadata_whitelist,
        workers=None,
        backend="process",
        batch_sizes=None,
        df_threshold=None,
        search_pipeline=None,
    ):
        batch_sizes = batch_sizes or {}
        row_batch_size = max(1, int(batch_sizes.get("rows", 5000)))
        languages = normalize_languages(pipeline_config.get("languages"))
        pipeline_config = dict(pipeline_config, languages=languages)
        if backend not in {"process", "thread"}:
            raise ValueError("backend must be either 'process' or 'thread'")
        field_names = [field[0] for field in fields]
        ensure_schema(self.conn, self.dialect)
        generation = begin_generation(
            self.conn, self.dialect, self.index_name, field_names, languages
        )
        writer = None
        try:
            writer = self.storage.writer(generation)
            document_count = self._stage_documents(
                writer,
                documents,
                ref,
                fields,
                pipeline_config,
                metadata_whitelist,
                workers,
                backend,
                row_batch_size,
            )
            term_count, vector_count = self._finalize_generation(
                writer, document_count, row_batch_size, df_threshold
            )
            self._record_and_validate_counts(
                generation, document_count, term_count, vector_count
            )
            self.conn.commit()
            writer = None
            activate_generation(
                self.conn,
                self.dialect,
                self.index_name,
                generation,
                build_metadata={
                    "pipeline": (
                        "custom"
                        if pipeline_config.get("pipeline") is not None
                        else "default"
                    )
                },
            )
        except Exception as error:
            writer = None
            self.conn.rollback()
            fail_generation(
                self.conn, self.dialect, self.index_name, generation, str(error)
            )
            cleanup_generation(self.conn, self.dialect, self.index_name, generation)
            raise

        try:
            self._clean_after_activation(generation)
        except Exception as error:
            self.conn.rollback()
            warnings.warn(
                f"SQL index activated but post-activation cleanup failed: {error}",
                RuntimeWarning,
            )

        self.storage._index_fields = field_names
        self.storage._search_pipeline = search_pipeline
        return self.storage.open_index()

    def _payloads(self, documents, ref, fields, pipeline_config, metadata_whitelist):
        for document in documents:
            yield (document, ref, fields, pipeline_config, metadata_whitelist)

    def _records(
        self,
        documents,
        ref,
        fields,
        pipeline_config,
        metadata_whitelist,
        workers,
        backend,
    ):
        payloads = self._payloads(
            documents, ref, fields, pipeline_config, metadata_whitelist
        )
        if not workers or int(workers) <= 1:
            return map(_process_document, payloads)
        selected_backend = backend
        if backend == "process":
            shared = (None, ref, fields, pipeline_config, metadata_whitelist)
            try:
                pickle.dumps(shared)
            except Exception as error:
                warnings.warn(
                    "Parallel backend 'process' unavailable "
                    f"({error!r}); falling back to 'thread'.",
                    RuntimeWarning,
                )
                selected_backend = "thread"
        executor_class = (
            ProcessPoolExecutor if selected_backend == "process" else ThreadPoolExecutor
        )
        max_workers = max(1, min(int(workers), 10))
        executor = executor_class(max_workers=max_workers)
        records = _bounded_map(
            executor, _process_document, payloads, max_workers * 2
        )
        return _ExecutorRecords(executor, records)

    def _stage_documents(
        self,
        writer,
        documents,
        ref,
        fields,
        pipeline_config,
        metadata_whitelist,
        workers,
        backend,
        row_batch_size,
    ):
        batches = {"documents": [], "frequencies": [], "postings": []}
        document_count = 0
        records = self._records(
            documents,
            ref,
            fields,
            pipeline_config,
            metadata_whitelist,
            workers,
            backend,
        )
        try:
            for document_records in records:
                document_count += 1
                for record in document_records:
                    batches["documents"].append(
                        (
                            record.field_ref,
                            record.field,
                            record.doc_ref,
                            record.length,
                            record.boost,
                        )
                    )
                    for term, frequency in record.term_counts.items():
                        batches["frequencies"].append(
                            (record.field_ref, term, frequency)
                        )
                        batches["postings"].append(
                            (
                                term,
                                record.field,
                                record.doc_ref,
                                record.metadata_by_term.get(term, {}),
                            )
                        )
                if any(len(batch) >= row_batch_size for batch in batches.values()):
                    self._flush(writer, batches)
            self._flush(writer, batches)
        finally:
            close = getattr(records, "close", None)
            if close is not None:
                close()
        return document_count

    def _flush(self, writer, batches):
        writer.write_document_fields(batches["documents"])
        writer.write_term_frequencies(batches["frequencies"])
        writer.write_postings(batches["postings"])
        for batch in batches.values():
            batch.clear()
        self.conn.commit()

    def _finalize_generation(self, writer, document_count, row_batch_size, threshold):
        placeholder = self.dialect.placeholder
        if threshold is not None:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    "SELECT term FROM lunr_v2_postings "
                    f"WHERE index_name={placeholder} AND generation={placeholder} "
                    "GROUP BY term HAVING COUNT(DISTINCT doc_ref) >= "
                    f"{placeholder}",
                    (self.index_name, writer.generation, int(threshold)),
                )
                excluded = [row[0] for row in cursor.fetchall()]
            finally:
                cursor.close()
            self._delete_excluded(writer.generation, excluded)
        self._recompute_lengths(writer.generation)

        cursor = self.conn.cursor()
        term_count = 0
        try:
            cursor.execute(
                "SELECT DISTINCT term FROM lunr_v2_term_frequencies "
                f"WHERE index_name={placeholder} AND generation={placeholder} "
                "ORDER BY term",
                (self.index_name, writer.generation),
            )
            while True:
                terms = cursor.fetchmany(row_batch_size)
                if not terms:
                    break
                writer.finalize_terms(
                    [
                        (term, term_count + index)
                        for index, (term,) in enumerate(terms)
                    ]
                )
                term_count += len(terms)
        finally:
            cursor.close()

        averages = self._average_lengths(writer.generation)
        vector_count = self._write_vectors(
            writer, document_count, averages, row_batch_size
        )
        return term_count, vector_count

    def _delete_excluded(self, generation, terms):
        for offset in range(0, len(terms), 498):
            chunk = terms[offset : offset + 498]
            placeholders = self.dialect.placeholders(len(chunk))
            cursor = self.conn.cursor()
            try:
                for table in ("lunr_v2_postings", "lunr_v2_term_frequencies"):
                    cursor.execute(
                        f"DELETE FROM {table} WHERE "
                        f"index_name={self.dialect.placeholder} "
                        f"AND generation={self.dialect.placeholder} "
                        f"AND term IN ({placeholders})",
                        (self.index_name, generation, *chunk),
                    )
            finally:
                cursor.close()

    def _recompute_lengths(self, generation):
        cursor = self.conn.cursor()
        read_cursor = self.conn.cursor()
        try:
            cursor.execute(
                "UPDATE lunr_v2_doc_fields SET length=0 "
                f"WHERE index_name={self.dialect.placeholder} "
                f"AND generation={self.dialect.placeholder}",
                (self.index_name, generation),
            )
            read_cursor.execute(
                "SELECT field_ref, SUM(tf) FROM lunr_v2_term_frequencies "
                f"WHERE index_name={self.dialect.placeholder} "
                f"AND generation={self.dialect.placeholder} GROUP BY field_ref",
                (self.index_name, generation),
            )
            while True:
                lengths = read_cursor.fetchmany(500)
                if not lengths:
                    break
                cursor.executemany(
                    "UPDATE lunr_v2_doc_fields SET length="
                    f"{self.dialect.placeholder} WHERE index_name="
                    f"{self.dialect.placeholder} AND generation="
                    f"{self.dialect.placeholder} AND field_ref="
                    f"{self.dialect.placeholder}",
                    [
                        (length, self.index_name, generation, field_ref)
                        for field_ref, length in lengths
                    ],
                )
        finally:
            read_cursor.close()
            cursor.close()

    def _average_lengths(self, generation):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT field, AVG(length) FROM lunr_v2_doc_fields "
                f"WHERE index_name={self.dialect.placeholder} "
                f"AND generation={self.dialect.placeholder} GROUP BY field",
                (self.index_name, generation),
            )
            return {field: float(length) for field, length in cursor.fetchall()}
        finally:
            cursor.close()

    def _write_vectors(
        self, writer, document_count, averages, row_batch_size
    ):
        cursor = self.conn.cursor()
        rows = []
        vector_count = 0
        try:
            cursor.execute(
                "SELECT d.field_ref, d.field, d.doc_ref, d.length, d.boost, "
                "t.term, t.tf, x.term_index, o.occurrences "
                "FROM lunr_v2_doc_fields d "
                "LEFT JOIN lunr_v2_term_frequencies t ON "
                "t.index_name=d.index_name AND t.generation=d.generation "
                "AND t.field_ref=d.field_ref LEFT JOIN lunr_v2_terms x ON "
                "x.index_name=t.index_name AND x.generation=t.generation "
                "AND x.term=t.term LEFT JOIN (SELECT term, COUNT(*) occurrences "
                "FROM lunr_v2_postings WHERE index_name="
                f"{self.dialect.placeholder} AND generation="
                f"{self.dialect.placeholder} GROUP BY term) o ON o.term=t.term "
                f"WHERE d.index_name={self.dialect.placeholder} "
                f"AND d.generation={self.dialect.placeholder} "
                "ORDER BY d.field_ref, x.term_index",
                (
                    self.index_name,
                    writer.generation,
                    self.index_name,
                    writer.generation,
                ),
            )
            current = None
            vector = None
            identity = None
            for (
                field_ref,
                field,
                doc_ref,
                length,
                boost,
                term,
                tf,
                index,
                count,
            ) in iter(cursor.fetchone, None):
                if field_ref != current:
                    if current is not None:
                        rows.append((*identity, vector.serialize(), vector.magnitude))
                        vector_count += 1
                        if len(rows) >= row_batch_size:
                            writer.write_vectors(rows)
                            rows.clear()
                    current = field_ref
                    identity = (field_ref, field, doc_ref)
                    vector = Vector()
                if term is None:
                    continue
                value = math.log(
                    1 + abs((document_count - count + 0.5) / (count + 0.5))
                )
                average = averages[field] or 1
                score = value * ((self.k1 + 1) * tf) / (
                    self.k1
                    * (1 - self.b + self.b * (length / average))
                    + tf
                )
                vector.insert(index, round(score * boost, 3))
            if current is not None:
                rows.append((*identity, vector.serialize(), vector.magnitude))
                vector_count += 1
            writer.write_vectors(rows)
        finally:
            cursor.close()
        return vector_count

    def _record_and_validate_counts(self, generation, documents, terms, vectors):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "UPDATE lunr_v2_generations SET document_count="
                f"{self.dialect.placeholder}, term_count={self.dialect.placeholder}, "
                f"vector_count={self.dialect.placeholder} WHERE index_name="
                f"{self.dialect.placeholder} AND generation={self.dialect.placeholder}",
                (documents, terms, vectors, self.index_name, generation),
            )
            for table, expected in (
                ("lunr_v2_terms", terms),
                ("lunr_v2_field_vectors", vectors),
            ):
                cursor.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE index_name="
                    f"{self.dialect.placeholder} AND generation="
                    f"{self.dialect.placeholder}",
                    (self.index_name, generation),
                )
                if cursor.fetchone()[0] != expected:
                    raise RuntimeError(f"Generation validation failed for {table}")
            cursor.execute(
                "SELECT document_count, term_count, vector_count "
                "FROM lunr_v2_generations WHERE index_name="
                f"{self.dialect.placeholder} AND generation="
                f"{self.dialect.placeholder}",
                (self.index_name, generation),
            )
            if cursor.fetchone() != (documents, terms, vectors):
                raise RuntimeError("Generation metadata count validation failed")
        finally:
            cursor.close()

    def _clean_after_activation(self, generation):
        cursor = self.conn.cursor()
        try:
            for table in ("lunr_v2_doc_fields", "lunr_v2_term_frequencies"):
                cursor.execute(
                    f"DELETE FROM {table} WHERE index_name={self.dialect.placeholder} "
                    f"AND generation={self.dialect.placeholder}",
                    (self.index_name, generation),
                )
            self.conn.commit()
        finally:
            cursor.close()


class _ExecutorRecords:
    def __init__(self, executor, records) -> None:
        self.executor = executor
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def close(self):
        self.executor.shutdown(wait=True)
