"""Mapping proxies used by SQL-backed :class:`lunr.index.Index` objects."""

from collections.abc import Mapping
from typing import Any, Dict, Iterator

from lunr.vector import Vector


class SqlInvertedIndexProxy(Mapping):
    def __init__(self, reader) -> None:
        self.reader = reader

    def __getitem__(self, term: str) -> Dict[str, Any]:
        return self.reader.get_posting(term)

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SqlFieldVectorsProxy(Mapping):
    def __init__(self, reader) -> None:
        self.reader = reader

    def __getitem__(self, field_ref: str) -> Vector:
        return self.reader.get_field_vector(field_ref)

    def __iter__(self) -> Iterator[str]:
        return iter(self.reader.iter_all_field_refs())

    def __len__(self) -> int:
        raise NotImplementedError

    def keys(self):
        return set(self.reader.iter_all_field_refs())
