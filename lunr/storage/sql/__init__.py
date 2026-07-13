from __future__ import annotations

from .dialects import DIALECTS, SqlDialect, connect_url, get_dialect
from .legacy import (
    SqlFieldVectorsProxy,
    SqlIndexReader,
    SqlIndexWriter,
    SqlInvertedIndexProxy,
)
from .schema import SqlRebuildRequiredError, get_active_generation


class SqlStorage:
    def __init__(
        self, conn, index_name: str, dialect, owns_connection: bool = False
    ) -> None:
        self.conn = conn
        self.index_name = index_name
        self.dialect = get_dialect(dialect) if isinstance(dialect, str) else dialect
        self.owns_connection = owns_connection
        self._legacy_write_requested = False

    @classmethod
    def from_url(cls, url: str, index_name: str) -> "SqlStorage":
        conn, dialect = connect_url(url)
        return cls(conn, index_name, dialect, owns_connection=True)

    @classmethod
    def from_conn(
        cls, conn, index_name: str, dialect: str = "sqlite"
    ) -> "SqlStorage":
        return cls(conn, index_name, dialect, owns_connection=False)

    def writer(self) -> SqlIndexWriter:
        self._legacy_write_requested = True
        return SqlIndexWriter(self)

    def reader(self) -> SqlIndexReader:
        if not self._legacy_write_requested:
            get_active_generation(self.conn, self.dialect, self.index_name)
        return SqlIndexReader(self)

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
