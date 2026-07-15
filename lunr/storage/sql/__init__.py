from __future__ import annotations

from .dialects import DIALECTS, SqlDialect, connect_url, get_dialect
from .legacy import (
    SqlFieldVectorsProxy,
    SqlIndexReader,
    SqlIndexWriter,
    SqlInvertedIndexProxy,
)
from .reader import SqlIndexReader as V2SqlIndexReader
from .schema import SqlRebuildRequiredError, get_active_generation
from .writer import SqlIndexWriter as V2SqlIndexWriter


class SqlStorage:
    def __init__(
        self, conn, index_name: str, dialect, owns_connection: bool = False
    ) -> None:
        self.conn = conn
        self.index_name = index_name
        self.dialect = get_dialect(dialect) if isinstance(dialect, str) else dialect
        self.owns_connection = owns_connection
        self._legacy_write_requested = False
        self._v2_generation = None

    @classmethod
    def from_url(cls, url: str, index_name: str) -> "SqlStorage":
        conn, dialect = connect_url(url)
        return cls(conn, index_name, dialect, owns_connection=True)

    @classmethod
    def from_conn(
        cls, conn, index_name: str, dialect: str = "sqlite"
    ) -> "SqlStorage":
        return cls(conn, index_name, dialect, owns_connection=False)

    def writer(self, generation=None):
        if generation is not None:
            self._v2_generation = generation
            return V2SqlIndexWriter(self, generation)
        self._legacy_write_requested = True
        return SqlIndexWriter(self)

    def reader(self):
        if self._legacy_write_requested:
            return SqlIndexReader(self)
        if self._v2_generation is None:
            active = get_active_generation(self.conn, self.dialect, self.index_name)
            if active is None:
                raise ValueError(f"No active SQL index named {self.index_name!r}")
            self._v2_generation = active.generation
        return V2SqlIndexReader(self, self._v2_generation)

    def close(self) -> None:
        if self.owns_connection:
            self.conn.close()

    def __enter__(self) -> "SqlStorage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "DIALECTS",
    "SqlDialect",
    "SqlFieldVectorsProxy",
    "SqlIndexReader",
    "SqlIndexWriter",
    "SqlInvertedIndexProxy",
    "SqlRebuildRequiredError",
    "SqlStorage",
]
