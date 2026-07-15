import warnings

from lunr import languages as lang
from lunr.builder import Builder
from lunr.stemmer import stemmer
from lunr.trimmer import trimmer
from lunr.stop_word_filter import stop_word_filter


def lunr(
    ref,
    fields,
    documents,
    languages=None,
    builder=None,
    storage=None,
    workers=None,
    parallel_backend="process",
    df_threshold=None,
):
    """A convenience function to configure and construct a lunr.Index.

    Args:
        ref (str): The key in the documents to be used a the reference.
        fields (list): A list of strings defining fields in the documents to
            index. Optionally a list of dictionaries with three keys:
            `field_name` defining the document's field, `boost` an integer
            defining a boost to be applied to the field, and `extractor`
            a callable taking the document as a single argument and returning
            a string located in the document in a particular way.
        documents (list): The list of dictonaries representing the documents
            to index. Optionally a 2-tuple of dicts, the first one being
            the document and the second the associated attributes to it.
        languages (str or list, optional): The languages to use for the
            language pipeline. If NLTK is unavailable, only NLTK-independent
            languages (currently ``"ru"``) are allowed; requesting other
            languages raises ``RuntimeError``.
        builder (Builder, optional): A pre-configured builder instance.
        storage (SqlStorage, optional): Optional SQL storage backend.
        workers (int, optional): Number of workers to use for SQL-backed
            parallel builds. If omitted, defaults to single-worker behavior.
        parallel_backend (str, optional): Parallel executor backend,
            either "process" or "thread".
        df_threshold (int, optional): Document frequency threshold for
            SQL-backed builds.  Terms appearing in at least this many
            distinct documents are removed before vectors are computed.

    Returns:
        Index: The populated Index ready to search against.
    """
    normalized_languages = lang.normalize_languages(languages)
    custom_builder = builder is not None
    builder = builder or get_default_builder(normalized_languages or None)
    storage = storage or builder._storage_backend
    if storage is not None and parallel_backend not in {"process", "thread"}:
        raise ValueError("backend must be either 'process' or 'thread'")
    if df_threshold is not None:
        builder.df_threshold(df_threshold)
    if workers is not None and storage is None:
        try:
            if int(workers) > 1:
                warnings.warn(
                    "workers>1 requires a SQL storage backend; ignoring parallel settings.",
                    RuntimeWarning,
                )
        except TypeError:
            pass
    builder.ref(ref)
    for field in fields:
        if isinstance(field, dict):
            builder.field(**field)
        else:
            builder.field(field)

    if storage is not None:
        from lunr.storage.sql.indexer import SqlIndexer

        configured_fields = [
            (name, field.boost, field.extractor)
            for name, field in builder._fields.items()
        ]
        pipeline_config = (
            {"pipeline": builder.pipeline, "languages": normalized_languages}
            if custom_builder
            else {"languages": normalized_languages}
        )
        effective_workers = (
            workers if workers is not None else builder._parallel_workers
        )
        effective_backend = (
            parallel_backend if workers is not None else builder._parallel_backend
        )
        return SqlIndexer(storage, b=builder._b, k1=builder._k1).build(
            documents,
            ref,
            configured_fields,
            pipeline_config,
            list(builder.metadata_whitelist),
            workers=effective_workers,
            backend=effective_backend,
            batch_sizes={"rows": builder._sql_row_batch_size},
            df_threshold=builder._df_threshold,
            search_pipeline=builder.search_pipeline,
        )

    for document in documents:
        if isinstance(document, (tuple, list)):
            builder.add(document[0], attributes=document[1])
        else:
            builder.add(document)

    return builder.build()


def get_default_builder(languages=None):
    """Creates a new pre-configured instance of Builder.

    Useful as a starting point to tweak the defaults.
    """
    languages = lang.normalize_languages(languages)
    if languages:
        requested_languages = set(languages)
        nltk_independent_languages = {"ru"}

        if not lang.LANGUAGE_SUPPORT and not requested_languages.issubset(
            nltk_independent_languages
        ):
            raise RuntimeError(
                "Language support requires NLTK. Install with: pip install lunr[languages]"
            )

        unsupported_languages = requested_languages - set(lang.SUPPORTED_LANGUAGES)
        if unsupported_languages:
            raise RuntimeError(
                "The specified languages {} are not supported, "
                "please choose one of {}".format(
                    ", ".join(unsupported_languages),
                    ", ".join(lang.SUPPORTED_LANGUAGES.keys()),
                )
            )
        if "ru" in languages:
            lang.ru.get_morph_analyzer()
        builder = lang.get_nltk_builder(languages)
    else:
        builder = Builder()
        builder.pipeline.add(trimmer, stop_word_filter, stemmer)
        builder.search_pipeline.add(stemmer)

    return builder
