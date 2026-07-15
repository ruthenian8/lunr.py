import pytest


def test_shared_backend_contract_on_sqlite(
    sqlite_backend_storage, backend_contract, documents
):
    backend_contract(sqlite_backend_storage, documents)


def test_backend_harness_closes_owned_storage_when_cleanup_fails(
    monkeypatch, backend_harness_factory
):
    class TrackingStorage:
        closed = False
        index_name = "tracking"

        def close(self):
            self.closed = True

    storage = TrackingStorage()
    function_globals = backend_harness_factory.__globals__
    monkeypatch.setenv("LUNR_TEST_POSTGRESQL_URL", "postgresql://unused")
    monkeypatch.setattr(
        function_globals["SqlStorage"],
        "from_url",
        lambda url, index_name: storage,
    )

    def fail_cleanup(harness):
        raise RuntimeError("cleanup failed")

    monkeypatch.setitem(function_globals, "_cleanup", fail_cleanup)
    harness_generator = backend_harness_factory(
        "LUNR_TEST_POSTGRESQL_URL", "postgresql"
    )
    next(harness_generator)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        next(harness_generator)

    assert storage.closed
