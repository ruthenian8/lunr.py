import math

from lunr.idf import idf


def test_idf_matches_lunr_js_field_posting_count():
    posting = {
        "_index": 0,
        "title": {"1": {}, "2": {}},
        "body": {"1": {}},
    }

    assert idf(posting, document_count=2) == math.log(8 / 7)
