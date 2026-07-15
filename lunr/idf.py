import math


def idf(posting, document_count):
    """A function to calculate the inverse document frequency for a posting.
    This is shared between the builder and the index.
    """
    document_refs = set()
    for field_name in posting:
        if field_name == "_index":
            continue
        document_refs.update(posting[field_name].keys())

    documents_with_term = len(document_refs)
    x = (document_count - documents_with_term + 0.5) / (documents_with_term + 0.5)
    return math.log(1 + abs(x))
