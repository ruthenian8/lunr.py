from __future__ import annotations

from .dialects import DIALECTS, SqlDialect, connect_url, get_dialect
from .proxies import SqlFieldVectorsProxy, SqlInvertedIndexProxy
from .reader import SqlIndexReader
from .schema import (
    SqlRebuildRequiredError,
    get_active_generation,
    prune_inactive_generations,
)
from .writer import SqlIndexWriter


class SqlStorage:
    def __init__(
        self, conn, index_name: str, dialect, owns_connection: bool = False
    ) -> None:
        self.conn = conn
        self.index_name = index_name
        self.dialect = get_dialect(dialect) if isinstance(dialect, str) else dialect
        self.owns_connection = owns_connection

    @classmethod
    def from_url(cls, url: str, index_name: str) -> "SqlStorage":
        conn, dialect = connect_url(url)
        return cls(conn, index_name, dialect, owns_connection=True)

    @classmethod
    def from_conn(cls, conn, index_name: str, dialect: str = "sqlite") -> "SqlStorage":
        return cls(conn, index_name, dialect, owns_connection=False)

    def writer(self, generation=None):
        if generation is None:
            raise TypeError("generation is required for a V2 SQL writer")
        return SqlIndexWriter(self, generation)

    def reader(self):
        active = get_active_generation(self.conn, self.dialect, self.index_name)
        if active is None:
            raise ValueError(f"No active SQL index named {self.index_name!r}")
        return SqlIndexReader(self, active.generation)

    def open_index(self, languages=None, generation=None):
        from lunr import get_default_builder
        from lunr.exceptions import BaseLunrException
        from lunr.index import Index
        from lunr.languages import normalize_languages

        active = generation or get_active_generation(
            self.conn, self.dialect, self.index_name
        )
        if active is None:
            raise ValueError(f"No active SQL index named {self.index_name!r}")

        stored_languages = normalize_languages(active.languages)
        if languages is not None:
            requested_languages = normalize_languages(languages)
            if requested_languages != stored_languages:
                raise BaseLunrException(
                    "Requested languages do not match the stored index languages"
                )

        if active.build_metadata.get("pipeline") == "custom":
            if getattr(self, "_search_pipeline", None) is None:
                raise BaseLunrException(
                    "A custom pipeline cannot be reconstructed from SQL metadata"
                )
            search_pipeline = self._search_pipeline
        else:
            search_pipeline = get_default_builder(
                stored_languages or None
            ).search_pipeline

        reader = SqlIndexReader(self, active.generation)
        return Index(
            inverted_index=SqlInvertedIndexProxy(reader),
            field_vectors=SqlFieldVectorsProxy(reader),
            token_set=None,
            fields=active.fields,
            pipeline=search_pipeline,
            storage_reader=reader,
        )

    def prune_inactive_generations(self):
        """Delete retained generations after all pinned readers are closed.

        This operation is never automatic because an older ``Index`` may still
        query its pinned generation. Building generations are not removed.
        """
        return prune_inactive_generations(self.conn, self.dialect, self.index_name)

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
