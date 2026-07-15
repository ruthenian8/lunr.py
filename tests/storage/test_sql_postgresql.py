import pytest


@pytest.mark.postgresql
def test_postgresql_backend_contract(
    postgresql_storage, backend_contract, documents
):
    backend_contract(postgresql_storage, documents)
