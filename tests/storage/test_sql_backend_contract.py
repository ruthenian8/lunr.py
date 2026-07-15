def test_shared_backend_contract_on_sqlite(
    sqlite_backend_storage, backend_contract, documents
):
    backend_contract(sqlite_backend_storage, documents)
