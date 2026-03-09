import sys

import pytest

from lunr import get_default_builder, lunr
from lunr.languages import ru


try:
    ru.get_morph_analyzer()
    HAS_PYMORPHY3 = True
except (RuntimeError, AttributeError):
    HAS_PYMORPHY3 = False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("машиной,", "машиной"),
        ("«машина»", "машина"),
        ("ЁЛКА", "ёлка"),
        ("ёлка", "ёлка"),
        ("елка", "елка"),
        ("маши\u0301на", "машина"),
        ("домо\u0301й", "домой"),
        ("тру\u0301дно", "трудно"),
        ("до*м", "дом"),
        ("чцать", "чать"),
        ("чцц", "ч"),
    ],
)
def test_clean_russian_token(raw, expected):
    assert ru.clean_russian_token(raw) == expected


@pytest.mark.skipif(not HAS_PYMORPHY3, reason="pymorphy3 is not installed")
@pytest.mark.parametrize(
    "word,expected",
    [
        ("машины", "машина"),
        ("машиной", "машина"),
        ("книге", "книга"),
        ("читала", "читать"),
        ("читает", "читать"),
    ],
)
def test_russian_morphology_normalization(word, expected):
    assert ru.normalize_russian_word(word) == expected


@pytest.mark.skipif(not HAS_PYMORPHY3, reason="pymorphy3 is not installed")
def test_russian_search_case_and_tense_agnostic():
    docs = [
        {
            "id": "1",
            "title": "В статье говорится о машине",
            "body": "Он читает книгу",
        }
    ]
    idx = lunr(ref="id", fields=("title", "body"), documents=docs, languages=["ru"])

    for query in ("машина", "машины", "машиной", "читать", "читала", "читает"):
        assert [result["ref"] for result in idx.search(query)] == ["1"]


@pytest.mark.skipif(not HAS_PYMORPHY3, reason="pymorphy3 is not installed")
def test_russian_search_cleanup_before_morphology():
    docs = [{"id": "1", "title": "Ёлкой, украшенной.", "body": "праздник"}]
    idx = lunr(ref="id", fields=("title", "body"), documents=docs, languages=["ru"])

    for query in ("ёлка", "ёлкой"):
        assert [result["ref"] for result in idx.search(query)] == ["1"]


@pytest.mark.skipif(not HAS_PYMORPHY3, reason="pymorphy3 is not installed")
def test_russian_stop_words_filtered():
    builder = get_default_builder(["ru"])
    assert builder.pipeline.run_string("и") == []
    assert builder.pipeline.run_string("в") == []
    assert builder.pipeline.run_string("на") == []
    assert builder.pipeline.run_string("что") == []
    assert builder.pipeline.run_string("это") == []
    assert builder.pipeline.run_string("машиной") == ["машина"]


def test_ru_missing_dependency_error(monkeypatch):
    ru.get_morph_analyzer.cache_clear()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pymorphy3":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "pymorphy3", raising=False)

    try:
        with pytest.raises(RuntimeError, match="pymorphy3 and pymorphy3-dicts-ru"):
            get_default_builder(["ru"])
    finally:
        ru.get_morph_analyzer.cache_clear()


@pytest.mark.skipif(not HAS_PYMORPHY3, reason="pymorphy3 is not installed")
def test_mixed_english_russian_does_not_crash(documents):
    docs = documents + [{"id": "ru", "title": "машина", "body": "он читает"}]
    idx = lunr(ref="id", fields=("title", "body"), documents=docs, languages=["en", "ru"])
    assert idx.search("читала")
