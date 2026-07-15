[![Build Status](https://github.com/yeraydiazdiaz/lunr.py/workflows/CI/badge.svg?branch=master)](https://github.com/yeraydiazdiaz/lunr.py/actions?workflow=CI)
[![codecov](https://codecov.io/gh/yeraydiazdiaz/lunr.py/branch/master/graph/badge.svg)](https://codecov.io/gh/yeraydiazdiaz/lunr.py)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/lunr.svg)](https://pypi.org/project/lunr/)
[![PyPI](https://img.shields.io/pypi/v/lunr.svg)](https://pypi.org/project/lunr/)
[![Read the Docs](https://img.shields.io/readthedocs/lunr.svg)](http://lunr.readthedocs.io/en/latest/)
[![Downloads](http://pepy.tech/badge/lunr)](http://pepy.tech/project/lunr)

# Lunr.py

A Python implementation of [Lunr.js](https://lunrjs.com) by [Oliver Nightingale](https://github.com/olivernn).

> A bit like Solr, but much smaller and not as bright.

This Python version of Lunr.js aims to bring the simple and powerful full text search
capabilities into Python guaranteeing results as close as the original
implementation as possible.

- [Documentation](http://lunr.readthedocs.io/en/latest/)

## What does this even do?

Lunr is a simple full text search solution for situations where deploying a full
scale solution like Elasticsearch isn't possible, viable or you're simply prototyping.
Lunr parses a set of documents and creates an inverted index for quick full text
searches in the same way other more complicated solution.

The trade-off is that Lunr keeps the inverted index in memory and requires you
to recreate or read the index at the start of your application.

## Interoperability with Lunr.js

A core objective of Lunr.py is to provide
[interoperability with the JavaScript version](https://lunr.readthedocs.io/en/latest/lunrjs-interop).

An example can be found in the [MkDocs documentation library](http://www.mkdocs.org/).
MkDocs produces a set of documents from the pages of the documentation and uses
[Lunr.js](https://lunrjs.com) in the frontend to power its built-in searching
engine. This set of documents is in the form of a JSON file which needs to be
fetched and parsed by Lunr.js to create the inverted index at startup of your application.

While this is not a problem for most sites, depending on the size of your document
set, this can take some time.

Lunr.py provides a backend solution, allowing you to parse the documents in Python
of time and create a serialized Lunr.js index you can pass have the browser
version read, minimizing start up time of your application.

Each version of lunr.py
[targets a specific version of lunr.js](https://github.com/yeraydiazdiaz/lunr.py/blob/master/lunr/__init__.py#L12)
and produces the same results for a
[non-trivial corpus of documents](https://github.com/yeraydiazdiaz/lunr.py/blob/master/tests/acceptance_tests/fixtures/mkdocs_index.json).

## Installation

`pip install lunr`

An optional and experimental support for other languages thanks to the
[Natural Language Toolkit](http://www.nltk.org/) stemmers is also available via
`pip install lunr[languages]`. Russian language support additionally uses
[pymorphy3](https://pypi.org/project/pymorphy3/) lemmatization and can be installed
with `pip install lunr[russian]` (or alongside NLTK with
`pip install lunr[languages,russian]`). The usage of the language feature is subject to
[NTLK corpus licensing clauses](https://github.com/nltk/nltk#redistributing).

Please refer to the
[documentation page on languages](https://lunr.readthedocs.io/en/latest/languages.html)
for more information.

## Usage

First, you'll need a list of dicts representing the documents you want to search on.
These documents must have a unique field which will serve as a reference and a
series of fields you'd like to search on.

Lunr provides a convenience `lunr` function to quickly index this set of documents:

```python
>>> from lunr import lunr
>>>
>>> documents = [{
...     'id': 'a',
...     'title': 'Mr. Green kills Colonel Mustard',
...     'body': 'Mr. Green killed Colonel Mustard in the study with the candlestick.',
... }, {
...     'id': 'b',
...     'title': 'Plumb waters plant',
...     'body': 'Professor Plumb has a green plant in his study',
... }]
>>> idx = lunr(
...     ref='id', fields=('title', 'body'), documents=documents
... )
>>> idx.search('kill')
[{'ref': 'a', 'score': 0.6931722372559913, 'match_data': <MatchData "kill">}]
>>> idx.search('study')
[{'ref': 'b', 'score': 0.23576799568081389, 'match_data': <MatchData "studi">}, {'ref': 'a', 'score': 0.2236629211724517, 'match_data': <MatchData "studi">}]
```

Please refer to the [documentation](http://lunr.readthedocs.io/en/latest/)
for more usage examples.

## SQL storage backend (optional)

Install all SQL and Flask integrations with `pip install "lunr[sql]"`, or select
`lunr[postgresql]`, `lunr[mysql]`, or `lunr[flask]`. SQLite uses Python's
built-in driver and needs no backend-specific extra.

`SqlStorage` persists terms, postings, and vectors in the generation-based
`lunr_v2_*` schema while keeping the public indexing and search API unchanged.

```python
from lunr import lunr
from lunr.storage.sql import SqlStorage

storage = SqlStorage.from_url("sqlite:///:memory:", index_name="docs")
idx = lunr(ref="id", fields=("title", "body"), documents=documents, storage=storage)
```

Use `sqlite:///relative.db` for a path relative to the current directory and
`sqlite:////absolute/path/index.db` for an absolute POSIX path. PostgreSQL and
MySQL use normal driver URLs such as `postgresql://user:pass@host/db` and
`mysql://user:pass@host/db`.

V1 databases containing `lunr_terms`, `lunr_postings`, or the other unversioned
tables are not migrated. Opening one raises `SqlRebuildRequiredError`; rebuild
the source documents to create the V2 schema.

Rebuilds stage an invisible generation and atomically activate it only after a
successful build. A failed build leaves the previous generation searchable.
Older generations are retained for readers already pinned to them. After every
such reader is closed, reclaim them explicitly with
`storage.prune_inactive_generations()`. Never prune while a pinned reader may
still be in use.

SQL mode supports exact term search and wildcard (`*`) expansion via SQL `LIKE`.
The following features are intentionally not supported in SQL mode and will raise
an exception:

- prohibited clauses (e.g. `-term`)
- fully negated queries
- fuzzy / edit-distance expansion
- serialization with `Index.serialize()`

Exact queries use a bounded number of SQL reads and wildcard queries add an
expansion read. Leading wildcards such as `*term` can require an expensive scan.
SQL streaming avoids buffering raw documents and the complete posting tree, but
the term vocabulary, final vector state, and staged database rows still scale
with corpus size.

### Multiprocess indexing and search with SQL storage

When using SQL storage, you can parallelize indexing with `parallel_backend="process"`
and then fan out searches across multiple worker processes.

```python
import multiprocessing as mp

from lunr import lunr
from lunr.storage.sql import SqlStorage

documents = [
    {
        "id": "a",
        "title": "Mr. Green kills Colonel Mustard",
        "body": "Mr. Green killed Colonel Mustard in the study with the candlestick.",
    },
    {
        "id": "b",
        "title": "Plumb waters plant",
        "body": "Professor Plumb has a green plant in his study",
    },
]

storage = SqlStorage.from_url("sqlite:///lunr_demo.db", index_name="docs")
idx = lunr(
    ref="id",
    fields=("title", "body"),
    documents=documents,
    storage=storage,
    workers=4,
    parallel_backend="process",
)


def search_in_worker(query: str):
    return query, [hit["ref"] for hit in idx.search(query)]


if __name__ == "__main__":
    # `fork` keeps the built SQL-backed index object available in child processes.
    with mp.get_context("fork").Pool(processes=2) as pool:
        results = dict(pool.map(search_in_worker, ["kill", "green study"]))

    print(results)
    # Example: {'kill': ['a'], 'green study': ['a', 'b']}
```

Default and Russian language pipelines can be reconstructed in worker processes
and from stored language metadata. If process payloads are not picklable, Lunr
emits a `RuntimeWarning` and falls back to threads. Arbitrary custom pipelines
cannot be reconstructed by a fresh `SqlStorage` facade; retain the building
facade or provide application-level pipeline setup.

The optional Flask example exposes a `/reindex` route for demonstration. A
production application must protect any reindex endpoint with authentication
and authorization; do not expose it publicly.
