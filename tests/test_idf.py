import math

from lunr.idf import idf


def test_idf_counts_document_once_across_multiple_fields():
    posting = {
        "_index": 0,
        "title": {"1": {}, "2": {}},
        "body": {"1": {}},
    }

    assert idf(posting, document_count=2) == math.log(1.2)
