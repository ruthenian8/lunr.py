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
        languages (str or list, optional): The languages to use if using
            NLTK language support, ignored if NLTK is not available.
        builder (Builder, optional): A pre-configured builder instance.
        storage (SqlStorage, optional): Optional SQL storage backend.
        workers (int, optional): Number of workers to use for SQL-backed
            parallel builds. If omitted, defaults to single-worker behavior.
        parallel_backend (str, optional): Parallel executor backend,
            either "process" or "thread".

    Returns:
        Index: The populated Index ready to search against.
    """
    builder = builder or get_default_builder(languages)
    if storage is not None:
        builder.storage(storage)
    if workers is not None and storage is not None:
        try:
            if (
                int(workers) > 1
                and parallel_backend == "process"
                and len(documents) < 200
            ):
                warnings.warn(
                    "workers>1 on small corpora may be slower due to parallel overhead.",
                    RuntimeWarning,
                )
        except TypeError:
            pass
        builder.parallel(workers=workers, backend=parallel_backend)
    elif workers is not None:
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
    if languages is not None and lang.LANGUAGE_SUPPORT:
        if isinstance(languages, str):
            languages = [languages]

        unsupported_languages = set(languages) - set(lang.SUPPORTED_LANGUAGES)
        if unsupported_languages:
            raise RuntimeError(
                "The specified languages {} are not supported, "
                "please choose one of {}".format(
                    ", ".join(unsupported_languages),
                    ", ".join(lang.SUPPORTED_LANGUAGES.keys()),
                )
            )
        builder = lang.get_nltk_builder(languages)
    else:
        builder = Builder()
        builder.pipeline.add(trimmer, stop_word_filter, stemmer)
        builder.search_pipeline.add(stemmer)

    return builder
