from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
import pickle
import warnings


from lunr.pipeline import Pipeline
from lunr.tokenizer import Tokenizer
from lunr.token_set import TokenSet
from lunr.field_ref import FieldRef
from lunr.index import Index
from lunr.vector import Vector
from lunr.idf import idf as Idf


def _process_document_for_parallel(args):
    doc_ref, doc, fields, pipeline, metadata_whitelist = args
    partial_fields = {}

    for field_name, extractor in fields:
        field_value = doc[field_name] if extractor is None else extractor(doc)
        tokens = Tokenizer(field_value)
        terms = pipeline.run(tokens, field_name)
        term_counts = defaultdict(int)
        metadata_by_term = defaultdict(lambda: defaultdict(list))

        for term in terms:
            term_key = str(term)
            term_counts[term_key] += 1
            for metadata_key in metadata_whitelist:
                metadata = term.metadata[metadata_key]
                metadata_by_term[term_key][metadata_key].append(metadata)

        serializable_metadata = {
            term: {k: list(v) for k, v in values.items()}
            for term, values in metadata_by_term.items()
        }
        partial_fields[field_name] = {
            "length": len(terms),
            "tfs": dict(term_counts),
            "metadata": serializable_metadata,
        }

    return doc_ref, partial_fields


class Field:
    """Represents a field with boost and extractor functions."""

    def __init__(self, field_name, boost=1, extractor=None):
        self.name = field_name
        self.boost = boost
        self.extractor = extractor

    def __repr__(self):
        return '<Field "{0.name}" boost="{0.boost}">'.format(self)

    def __hash__(self):
        return hash(self.name)


class Builder:
    """Performs indexing on a set of documents and returns instances of
    lunr.Index ready for querying.

    All configuration of the index is done via the builder, the fields to
    index, the document reference, the text processing pipeline and document
    scoring parameters are all set on the builder before indexing.
    """

    def __init__(self):
        self._ref = "id"
        self._fields = {}
        self.inverted_index = {}
        self.field_term_frequencies = {}
        self.field_lengths = {}
        self.pipeline = Pipeline()
        self.search_pipeline = Pipeline()
        self._documents = {}
        self.document_count = 0
        self._b = 0.75
        self._k1 = 1.2
        self.term_index = 0
        self.metadata_whitelist = []
        # Optional storage backend. If set by calling .storage(), the builder
        # will persist the index into the backend on build() and return an
        # Index configured to read from it.
        self._storage_backend = None
        self._parallel_workers = 1
        self._parallel_backend = "process"
        self._raw_documents = []
        self._defer_indexing = False
        self._sql_flush_enabled = False
        self._sql_doc_batch_size = 500
        self._sql_row_batch_size = 5000
        self._sql_commit_every_docs = None
        self._sql_commit_every_rows = None

    def ref(self, ref):
        """Sets the document field used as the document reference.

        Every document must have this field. The type of this field in the
        document should be a string, if it is not a string it will be coerced
        into a string by calling `str`.

        The default ref is 'id'. The ref should _not_ be changed during
        indexing, it should be set before any documents are added to the index.
        Changing it during indexing can lead to inconsistent results.

        """
        self._ref = ref

    def field(self, field_name, boost=1, extractor=None):
        """Adds a field to the list of document fields that will be indexed.

        Every document being indexed should have this field. None values for
        this field in indexed documents will not cause errors but will limit
        the chance of that document being retrieved by searches.

        All fields should be added before adding documents to the index. Adding
        fields after a document has been indexed will have no effect on already
        indexed documents.

        Fields can be boosted at build time. This allows terms within that
        field to have more importance on search results. Use a field boost to
        specify that matches within one field are more important that other
        fields.

        Args:
            field_name (str): Name of the field to be added, must not include
                a forward slash '/'.
            boost (int): Optional boost factor to apply to field.
            extractor (callable): Optional function to extract a field from
                the document.

        Raises:
            ValueError: If the field name contains a `/`.
        """
        if "/" in field_name:
            raise ValueError("Field {} contains illegal character `/`")

        self._fields[field_name] = Field(field_name, boost, extractor)

    def b(self, number):
        """A parameter to tune the amount of field length normalisation that is
        applied when calculating relevance scores.

        A value of 0 will completely disable any normalisation and a value of 1
        will fully normalise field lengths. The default is 0.75. Values of b
        will be clamped to the range 0 - 1.
        """
        if number < 0:
            self._b = 0
        elif number > 1:
            self._b = 1
        else:
            self._b = number

    def k1(self, number):
        """A parameter that controls the speed at which a rise in term
        frequency results in term frequency saturation.

        The default value is 1.2. Setting this to a higher value will give
        slower saturation levels, a lower value will result in quicker
        saturation.
        """
        self._k1 = number

    def add(self, doc, attributes=None):
        """Adds a document to the index.

        Before adding documents to the index it should have been fully
        setup, with the document ref and all fields to index already having
        been specified.

        The document must have a field name as specified by the ref (by default
        this is 'id') and it should have all fields defined for indexing,
        though None values will not cause errors.

        Args:
            - doc (dict): The document to be added to the index.
            - attributes (dict, optional): A set of attributes corresponding
            to the document, currently a single `boost` -> int will be
            taken into account.
        """
        doc_ref = str(doc[self._ref])
        self._documents[doc_ref] = attributes or {}
        self.document_count += 1
        if self._parallel_workers > 1 or (
            self._storage_backend is not None and self._sql_flush_enabled
        ):
            self._raw_documents.append((doc_ref, doc))
            self._defer_indexing = True

        if self._parallel_workers > 1 or (
            self._storage_backend is not None and self._sql_flush_enabled
        ):
            return

        self._index_document(doc_ref, doc)

    def _index_document(self, doc_ref, doc):
        for field_name, field in self._fields.items():
            extractor = field.extractor
            field_value = doc[field_name] if extractor is None else extractor(doc)
            tokens = Tokenizer(field_value)
            terms = self.pipeline.run(tokens, field_name)
            field_ref = FieldRef(doc_ref, field_name)
            field_terms = defaultdict(int)

            # TODO: field_refs are casted to strings in JS, should we allow
            # FieldRef as keys?
            self.field_term_frequencies[str(field_ref)] = field_terms
            self.field_lengths[str(field_ref)] = len(terms)

            for term in terms:
                # TODO: term is a Token, should we allow Tokens as keys?
                term_key = str(term)

                field_terms[term_key] += 1
                if term_key not in self.inverted_index:
                    posting = {_field_name: {} for _field_name in self._fields}
                    posting["_index"] = self.term_index
                    self.term_index += 1
                    self.inverted_index[term_key] = posting

                if doc_ref not in self.inverted_index[term_key][field_name]:
                    self.inverted_index[term_key][field_name][doc_ref] = defaultdict(
                        list
                    )

                for metadata_key in self.metadata_whitelist:
                    metadata = term.metadata[metadata_key]
                    self.inverted_index[term_key][field_name][doc_ref][
                        metadata_key
                    ].append(metadata)

    def parallel(self, workers=2, backend="process"):
        """Enable parallel indexing.

        Parallel indexing is currently supported for SQL-backed builds and is
        opt-in. The worker count is clamped to the range 1..10.
        """
        self._parallel_workers = max(1, min(int(workers), 10))
        if backend not in {"process", "thread"}:
            raise ValueError("backend must be either 'process' or 'thread'")
        self._parallel_backend = backend
        return self

    # -------------------------------------------------------------------
    # Storage configuration
    # -------------------------------------------------------------------
    def storage(self, storage_backend):
        """Configure a storage backend for persisting the index.

        If called before ``build()`` this will cause the builder to
        write the constructed index into the provided backend and return
        an Index that queries from it. The storage backend must conform
        to the ``SqlStorage`` interface, exposing ``writer()`` and
        ``reader()`` methods returning objects with the appropriate methods.

        Parameters
        ----------
        storage_backend : object
            A storage backend instance created from ``lunr.storage.sql.SqlStorage``.
        """
        self._storage_backend = storage_backend
        return self

    def sql_flush(self, enabled=True, doc_batch_size=500, row_batch_size=5000):
        """Configure bounded-memory SQL flushing.

        When enabled with a SQL backend, postings and term frequencies are
        streamed to SQL in chunks instead of keeping the entire posting tree in
        memory.
        """
        self._sql_flush_enabled = bool(enabled)
        self._sql_doc_batch_size = max(1, int(doc_batch_size))
        self._sql_row_batch_size = max(1, int(row_batch_size))
        return self

    def sql_commit_every(self, docs=None, rows=None):
        """Configure optional commit cadence for SQL writes."""
        self._sql_commit_every_docs = None if docs is None else max(1, int(docs))
        self._sql_commit_every_rows = None if rows is None else max(1, int(rows))
        return self

    def build(self):
        """Builds the index, creating an instance of `lunr.Index`.

        This completes the indexing process and should only be called once all
        documents have been added to the index.
        """
        if (
            self._storage_backend is not None
            and self._parallel_workers > 1
            and self._sql_flush_enabled
        ):
            return self._build_sql_parallel_incremental()

        if self._storage_backend is not None and self._sql_flush_enabled:
            return self._build_sql_incremental()

        if self._storage_backend is not None and self._parallel_workers > 1:
            return self._build_sql_parallel()

        if self._defer_indexing and self._raw_documents and not self.inverted_index:
            for doc_ref, doc in self._raw_documents:
                self._index_document(doc_ref, doc)
            self._raw_documents = []
            self._defer_indexing = False

        # Calculate average field lengths and construct field vectors in all
        # modes. These operations populate self.field_vectors and
        # self.field_lengths used by the scoring algorithm.
        self._calculate_average_field_lengths()
        self._create_field_vectors()
        # Determine whether we are operating with a storage backend. If not,
        # build and return an in‑memory index as before.
        if self._storage_backend is None:
            # Create a token set from all terms for wildcard/fuzzy expansion.
            self._create_token_set()
            return Index(
                inverted_index=self.inverted_index,
                field_vectors=self.field_vectors,
                token_set=self.token_set,
                fields=list(self._fields.keys()),
                pipeline=self.search_pipeline,
            )

        # With a storage backend configured, persist the index to the
        # backend and return an Index configured to use the SQL reader. We
        # explicitly do *not* build a TokenSet because term expansion will be
        # performed by the database via LIKE.
        writer = self._storage_backend.writer()
        # Persist terms and postings. The inverted_index has structure:
        # { term : { field_name : { doc_ref : metadata_dict }, '_index': idx } }
        for term, posting in self.inverted_index.items():
            term_index = posting["_index"]
            # Write term index
            writer.upsert_term(term, term_index)
            for field_name in self._fields:
                field_postings = posting.get(field_name, {})
                for doc_ref, metadata in field_postings.items():
                    # metadata is defaultdict(list) -> convert to normal dict
                    md = {k: list(v) for k, v in metadata.items()}
                    writer.upsert_posting(term, field_name, doc_ref, md)
        # Persist field vectors. Keys are fieldRef strings; we need to know
        # the constituent field and doc_ref. A fieldRef takes the form
        # 'docRef/fieldName'.
        for field_ref, vector in self.field_vectors.items():
            parsed_ref = FieldRef.from_string(field_ref)
            writer.upsert_field_vector(
                field_ref, parsed_ref.field_name, parsed_ref.doc_ref, vector
            )
        # Commit writes to the database.
        writer.commit()
        # Create reader and proxies for the Index. No token_set is provided.
        reader = self._storage_backend.reader()
        from lunr.storage.sql import SqlInvertedIndexProxy, SqlFieldVectorsProxy

        inverted_index_proxy = SqlInvertedIndexProxy(reader)
        field_vectors_proxy = SqlFieldVectorsProxy(reader)
        return Index(
            inverted_index=inverted_index_proxy,
            field_vectors=field_vectors_proxy,
            token_set=None,
            fields=list(self._fields.keys()),
            pipeline=self.search_pipeline,
            storage_reader=reader,
        )

    def _build_sql_parallel(self):
        self.inverted_index = {}
        self.field_term_frequencies = {}
        self.field_lengths = {}
        self.term_index = 0

        fields = [(name, field.extractor) for name, field in self._fields.items()]
        worker_payloads = [
            (doc_ref, doc, fields, self.pipeline, self.metadata_whitelist)
            for doc_ref, doc in self._raw_documents
        ]

        backend = self._parallel_backend
        if backend == "process":
            try:
                for payload in worker_payloads:
                    pickle.dumps(payload)
            except Exception as exc:
                backend = "thread"
                self._warn_process_fallback(exc)

        executor_cls = (
            ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
        )
        with executor_cls(max_workers=self._parallel_workers) as executor:
            for doc_ref, partial_fields in executor.map(
                _process_document_for_parallel, worker_payloads
            ):
                self._merge_partial(doc_ref, partial_fields)

        self._calculate_average_field_lengths()
        self._create_field_vectors()

        writer = self._storage_backend.writer()
        batch_size = 1000
        terms_batch = []
        postings_batch = []
        vectors_batch = []
        for term, posting in self.inverted_index.items():
            terms_batch.append((term, posting["_index"]))
            if len(terms_batch) >= batch_size:
                writer.upsert_terms_bulk(terms_batch)
                terms_batch = []

            for field_name in self._fields:
                for doc_ref, metadata in posting.get(field_name, {}).items():
                    postings_batch.append(
                        (
                            term,
                            field_name,
                            doc_ref,
                            {k: list(v) for k, v in metadata.items()},
                        )
                    )
                    if len(postings_batch) >= batch_size:
                        writer.upsert_postings_bulk(postings_batch)
                        postings_batch = []

        for field_ref, vector in self.field_vectors.items():
            parsed_ref = FieldRef.from_string(field_ref)
            vectors_batch.append(
                (field_ref, parsed_ref.field_name, parsed_ref.doc_ref, vector)
            )
            if len(vectors_batch) >= batch_size:
                writer.upsert_field_vectors_bulk(vectors_batch)
                vectors_batch = []

        if terms_batch:
            writer.upsert_terms_bulk(terms_batch)
        if postings_batch:
            writer.upsert_postings_bulk(postings_batch)
        if vectors_batch:
            writer.upsert_field_vectors_bulk(vectors_batch)
        writer.commit()
        self._raw_documents = []
        self._defer_indexing = False

        reader = self._storage_backend.reader()
        from lunr.storage.sql import SqlInvertedIndexProxy, SqlFieldVectorsProxy

        return Index(
            inverted_index=SqlInvertedIndexProxy(reader),
            field_vectors=SqlFieldVectorsProxy(reader),
            token_set=None,
            fields=list(self._fields.keys()),
            pipeline=self.search_pipeline,
            storage_reader=reader,
        )

    def _warn_process_fallback(self, exc):
        warnings.warn(
            "Parallel backend 'process' unavailable "
            f"({exc!r}); falling back to 'thread'.",
            RuntimeWarning,
        )

    def _flush_incremental_batches(self, writer, batches):
        rows_written = 0
        if batches["terms"]:
            writer.upsert_terms_bulk(batches["terms"])
            rows_written += len(batches["terms"])
            batches["terms"] = []
        if batches["postings"]:
            writer.upsert_postings_bulk(batches["postings"])
            rows_written += len(batches["postings"])
            batches["postings"] = []
        if batches["doc_fields"]:
            writer.upsert_doc_fields_bulk(batches["doc_fields"])
            rows_written += len(batches["doc_fields"])
            batches["doc_fields"] = []
        if batches["term_frequencies"]:
            writer.upsert_term_frequencies_bulk(batches["term_frequencies"])
            rows_written += len(batches["term_frequencies"])
            batches["term_frequencies"] = []
        return rows_written

    def _build_sql_incremental(self):
        writer = self._storage_backend.writer()
        batches = {
            "terms": [],
            "postings": [],
            "doc_fields": [],
            "term_frequencies": [],
        }
        term_to_index = {}
        postings_count_by_term = defaultdict(int)
        field_length_sum = defaultdict(int)
        field_doc_count = defaultdict(int)
        rows_since_commit = 0
        docs_since_commit = 0

        for doc_ref, doc in self._raw_documents:
            for field_name, field in self._fields.items():
                extractor = field.extractor
                field_value = doc[field_name] if extractor is None else extractor(doc)
                terms = self.pipeline.run(Tokenizer(field_value), field_name)
                field_ref = str(FieldRef(doc_ref, field_name))
                term_counts = defaultdict(int)
                metadata_by_term = defaultdict(lambda: defaultdict(list))
                for term in terms:
                    term_key = str(term)
                    term_counts[term_key] += 1
                    for metadata_key in self.metadata_whitelist:
                        metadata_by_term[term_key][metadata_key].append(
                            term.metadata[metadata_key]
                        )
                field_length_sum[field_name] += len(terms)
                field_doc_count[field_name] += 1
                batches["doc_fields"].append(
                    (field_ref, field_name, doc_ref, len(terms))
                )
                for term_key, tf in term_counts.items():
                    if term_key not in term_to_index:
                        term_to_index[term_key] = len(term_to_index)
                        batches["terms"].append((term_key, term_to_index[term_key]))
                    batches["term_frequencies"].append((field_ref, term_key, tf))
                    postings_count_by_term[term_key] += 1
                    md = {
                        k: list(v)
                        for k, v in metadata_by_term.get(term_key, {}).items()
                    }
                    batches["postings"].append((term_key, field_name, doc_ref, md))
            docs_since_commit += 1
            if docs_since_commit >= self._sql_doc_batch_size or any(
                len(v) >= self._sql_row_batch_size for v in batches.values()
            ):
                rows_since_commit += self._flush_incremental_batches(writer, batches)
                if (
                    self._sql_commit_every_docs
                    and docs_since_commit >= self._sql_commit_every_docs
                ) or (
                    self._sql_commit_every_rows
                    and rows_since_commit >= self._sql_commit_every_rows
                ):
                    writer.commit()
                    docs_since_commit = 0
                    rows_since_commit = 0

        self.average_field_length = defaultdict(float)
        for field_name in self._fields:
            count = field_doc_count[field_name] or 1
            self.average_field_length[field_name] = field_length_sum[field_name] / count

        rows_since_commit += self._flush_incremental_batches(writer, batches)

        reader = self._storage_backend.reader()
        vectors_batch = []
        current_field_ref = None
        current_vector = None
        current_field_name = None
        current_doc_ref = None
        current_length = 0

        doc_field_lengths = {
            field_ref: (field, doc_ref, length)
            for field_ref, field, doc_ref, length in reader.iter_doc_fields()
        }

        for field_ref, term, tf in reader.iter_term_frequencies():
            if field_ref != current_field_ref:
                if current_field_ref is not None:
                    vectors_batch.append(
                        (
                            current_field_ref,
                            current_field_name,
                            current_doc_ref,
                            current_vector,
                        )
                    )
                current_field_ref = field_ref
                current_field_name, current_doc_ref, current_length = doc_field_lengths[
                    field_ref
                ]
                current_vector = Vector()

            documents_with_term = postings_count_by_term[term]
            x = (self.document_count - documents_with_term + 0.5) / (
                documents_with_term + 0.5
            )
            idf = math.log(1 + abs(x))
            score = (
                idf
                * ((self._k1 + 1) * tf)
                / (
                    self._k1
                    * (
                        1
                        - self._b
                        + self._b
                        * (
                            current_length
                            / self.average_field_length[current_field_name]
                        )
                    )
                    + tf
                )
            )
            score *= self._fields[current_field_name].boost
            score *= self._documents[current_doc_ref].get("boost", 1)
            current_vector.insert(term_to_index[term], round(score, 3))

            if len(vectors_batch) >= self._sql_row_batch_size:
                writer.upsert_field_vectors_bulk(vectors_batch)
                rows_since_commit += len(vectors_batch)
                vectors_batch = []

        if current_field_ref is not None:
            vectors_batch.append(
                (current_field_ref, current_field_name, current_doc_ref, current_vector)
            )
        if vectors_batch:
            writer.upsert_field_vectors_bulk(vectors_batch)

        writer.commit()
        self.inverted_index = {}
        self.field_term_frequencies = {}
        self.field_lengths = {}
        self._raw_documents = []
        self._defer_indexing = False

        from lunr.storage.sql import SqlFieldVectorsProxy, SqlInvertedIndexProxy

        return Index(
            inverted_index=SqlInvertedIndexProxy(reader),
            field_vectors=SqlFieldVectorsProxy(reader),
            token_set=None,
            fields=list(self._fields.keys()),
            pipeline=self.search_pipeline,
            storage_reader=reader,
        )

    def _build_sql_parallel_incremental(self):
        fields = [(name, field.extractor) for name, field in self._fields.items()]
        worker_payloads = [
            (doc_ref, doc, fields, self.pipeline, self.metadata_whitelist)
            for doc_ref, doc in self._raw_documents
        ]
        backend = self._parallel_backend
        if backend == "process":
            try:
                for payload in worker_payloads:
                    pickle.dumps(payload)
            except Exception as exc:
                backend = "thread"
                self._warn_process_fallback(exc)

        writer = self._storage_backend.writer()
        batches = {
            "terms": [],
            "postings": [],
            "doc_fields": [],
            "term_frequencies": [],
        }
        term_to_index = {}
        postings_count_by_term = defaultdict(int)
        field_length_sum = defaultdict(int)
        field_doc_count = defaultdict(int)

        executor_cls = (
            ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
        )
        with executor_cls(max_workers=self._parallel_workers) as executor:
            for doc_ref, partial_fields in executor.map(
                _process_document_for_parallel, worker_payloads
            ):
                for field_name, field_data in partial_fields.items():
                    field_ref = str(FieldRef(doc_ref, field_name))
                    field_length_sum[field_name] += field_data["length"]
                    field_doc_count[field_name] += 1
                    batches["doc_fields"].append(
                        (field_ref, field_name, doc_ref, field_data["length"])
                    )
                    for term_key, tf in field_data["tfs"].items():
                        if term_key not in term_to_index:
                            term_to_index[term_key] = len(term_to_index)
                            batches["terms"].append((term_key, term_to_index[term_key]))
                        batches["term_frequencies"].append((field_ref, term_key, tf))
                        postings_count_by_term[term_key] += 1
                        md = {
                            k: list(v)
                            for k, v in field_data["metadata"].get(term_key, {}).items()
                        }
                        batches["postings"].append((term_key, field_name, doc_ref, md))
                if any(len(v) >= self._sql_row_batch_size for v in batches.values()):
                    self._flush_incremental_batches(writer, batches)

        self._flush_incremental_batches(writer, batches)
        self.average_field_length = defaultdict(float)
        for field_name in self._fields:
            count = field_doc_count[field_name] or 1
            self.average_field_length[field_name] = field_length_sum[field_name] / count

        reader = self._storage_backend.reader()
        vectors_batch = []
        doc_field_lengths = {
            field_ref: (field, doc_ref, length)
            for field_ref, field, doc_ref, length in reader.iter_doc_fields()
        }
        current_field_ref = None
        current_vector = None
        current_field_name = None
        current_doc_ref = None
        current_length = 0
        for field_ref, term, tf in reader.iter_term_frequencies():
            if field_ref != current_field_ref:
                if current_field_ref is not None:
                    vectors_batch.append(
                        (
                            current_field_ref,
                            current_field_name,
                            current_doc_ref,
                            current_vector,
                        )
                    )
                current_field_ref = field_ref
                current_field_name, current_doc_ref, current_length = doc_field_lengths[
                    field_ref
                ]
                current_vector = Vector()
            documents_with_term = postings_count_by_term[term]
            x = (self.document_count - documents_with_term + 0.5) / (
                documents_with_term + 0.5
            )
            idf = math.log(1 + abs(x))
            score = (
                idf
                * ((self._k1 + 1) * tf)
                / (
                    self._k1
                    * (
                        1
                        - self._b
                        + self._b
                        * (
                            current_length
                            / self.average_field_length[current_field_name]
                        )
                    )
                    + tf
                )
            )
            score *= self._fields[current_field_name].boost
            score *= self._documents[current_doc_ref].get("boost", 1)
            current_vector.insert(term_to_index[term], round(score, 3))

            if len(vectors_batch) >= self._sql_row_batch_size:
                writer.upsert_field_vectors_bulk(vectors_batch)
                vectors_batch = []

        if current_field_ref is not None:
            vectors_batch.append(
                (current_field_ref, current_field_name, current_doc_ref, current_vector)
            )
        if vectors_batch:
            writer.upsert_field_vectors_bulk(vectors_batch)
        writer.commit()

        self.inverted_index = {}
        self.field_term_frequencies = {}
        self.field_lengths = {}
        self._raw_documents = []
        self._defer_indexing = False

        from lunr.storage.sql import SqlFieldVectorsProxy, SqlInvertedIndexProxy

        return Index(
            inverted_index=SqlInvertedIndexProxy(reader),
            field_vectors=SqlFieldVectorsProxy(reader),
            token_set=None,
            fields=list(self._fields.keys()),
            pipeline=self.search_pipeline,
            storage_reader=reader,
        )

    def _merge_partial(self, doc_ref, partial_fields):
        for field_name, field_data in partial_fields.items():
            field_ref = str(FieldRef(doc_ref, field_name))
            self.field_lengths[field_ref] = field_data["length"]
            self.field_term_frequencies[field_ref] = field_data["tfs"]

            for term_key, _ in field_data["tfs"].items():
                if term_key not in self.inverted_index:
                    posting = {_field_name: {} for _field_name in self._fields}
                    posting["_index"] = self.term_index
                    self.term_index += 1
                    self.inverted_index[term_key] = posting
                if doc_ref not in self.inverted_index[term_key][field_name]:
                    self.inverted_index[term_key][field_name][doc_ref] = defaultdict(
                        list
                    )

                metadata_for_term = field_data["metadata"].get(term_key, {})
                for metadata_key, values in metadata_for_term.items():
                    self.inverted_index[term_key][field_name][doc_ref][
                        metadata_key
                    ].extend(values)

    def _create_token_set(self):
        """Creates a token set of all tokens in the index using `lunr.TokenSet`"""
        self.token_set = TokenSet.from_list(sorted(list(self.inverted_index.keys())))

    def _calculate_average_field_lengths(self):
        """Calculates the average document length for this index"""
        accumulator = defaultdict(int)
        documents_with_field = defaultdict(int)

        for field_ref, length in self.field_lengths.items():
            _field_ref = FieldRef.from_string(field_ref)
            field = _field_ref.field_name

            documents_with_field[field] += 1
            accumulator[field] += length

        for field_name in self._fields:
            accumulator[field_name] /= documents_with_field[field_name]

        self.average_field_length = accumulator

    def _create_field_vectors(self):
        """Builds a vector space model of every document using lunr.Vector."""
        field_vectors = {}
        term_idf_cache = {}

        for field_ref, term_frequencies in self.field_term_frequencies.items():
            _field_ref = FieldRef.from_string(field_ref)
            field_name = _field_ref.field_name
            field_length = self.field_lengths[field_ref]
            field_vector = Vector()
            field_boost = self._fields[field_name].boost
            doc_boost = self._documents[_field_ref.doc_ref].get("boost", 1)

            for term, tf in term_frequencies.items():
                term_index = self.inverted_index[term]["_index"]

                if term not in term_idf_cache:
                    idf = Idf(self.inverted_index[term], self.document_count)
                    term_idf_cache[term] = idf
                else:
                    idf = term_idf_cache[term]

                score = (
                    idf
                    * ((self._k1 + 1) * tf)
                    / (
                        self._k1
                        * (
                            1
                            - self._b
                            + self._b
                            * (field_length / self.average_field_length[field_name])
                        )
                        + tf
                    )
                )
                score *= field_boost
                score *= doc_boost
                score_with_precision = round(score, 3)

                field_vector.insert(term_index, score_with_precision)

            field_vectors[field_ref] = field_vector

        self.field_vectors = field_vectors

    def use(self, fn, *args, **kwargs):
        """Applies a plugin to the index builder.

        A plugin is a function that is called with the index builder as its
        context. Plugins can be used to customise or extend the behaviour of
        the index in some way.

        A plugin is just a function, that encapsulated the custom behaviour
        that should be applied when building the index. The plugin function
        will be called with the index builder as its argument, additional
        arguments can also be passed when calling use.
        """
        fn(self, *args, **kwargs)
