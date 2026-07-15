import pytest


@pytest.mark.mysql
def test_mysql_backend_contract(mysql_storage, backend_contract, documents):
    backend_contract(mysql_storage, documents)
