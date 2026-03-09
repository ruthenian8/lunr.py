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
        ("дом&amp;сад", "домсад"),
        ("&quot;машина&quot;", "машина"),
        ("текст<br>строка", "текстстрока"),
        ("текст<br/>строка", "текстстрока"),
        ("текст<BR>строка", "текстстрока"),
        ("&amp;", ""),
        ("&quot;", ""),
        ("<br>", ""),
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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("декламирует:]<p>ль", ["декламирует", "ль"]),
        ("надо.<p>[искать", ["надо", "искать"]),
        ("что\u2026[", ["что"]),
        ("знае\u2026дак", ["знае", "дак"]),
        ("заботы\u00bb.<p>обнять", ["заботы", "обнять"]),
        ("зачитывает:]<p>стоять", ["зачитывает", "стоять"]),
        ("показаться?]дак", ["показаться", "дак"]),
        ("годочка!\u00bb<p>тогда", ["годочка", "тогда"]),
        ("жи\u2026ящмень", ["жи", "ящмень"]),
        ("вернуть.<p>только", ["вернуть", "только"]),
        ("золотистые.<p>[этый", ["золотистые", "этый"]),
        ("0&lt;p&gt;&lt;p&gt;lll</p></p></p></p></p></p></p>", ["0", "lll"]),
        ("машиной,", ["машиной"]),
        ("<p></p>", []),
    ],
)
def test_split_and_clean_russian_token(raw, expected):
    assert ru.split_and_clean_russian_token(raw) == expected


def test_russian_cleanup_filter_splits_tokens():
    from lunr.token import Token

    token = Token("вернуть.<p>только", {"position": [0, 17], "index": 0})
    result = ru.russian_cleanup_filter(token)
    assert isinstance(result, list)
    assert [str(t) for t in result] == ["вернуть", "только"]


def test_russian_cleanup_filter_single_token():
    from lunr.token import Token

    token = Token("машиной,", {"position": [0, 8], "index": 0})
    result = ru.russian_cleanup_filter(token)
    assert str(result) == "машиной"


def test_russian_cleanup_filter_empty():
    from lunr.token import Token

    token = Token("<p></p>", {"position": [0, 7], "index": 0})
    result = ru.russian_cleanup_filter(token)
    assert result is None


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
