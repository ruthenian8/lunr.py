import re
import unicodedata
from functools import lru_cache
try:
    from importlib import resources
except ImportError:  # pragma: no cover
    import importlib_resources as resources


from lunr.pipeline import Pipeline

_PUNCTUATION_RE = re.compile(r"^\W+|\W+$")
_CYRILLIC_RE = re.compile(r"[а-яё]")
_EXTRA_ORTH_RE = re.compile("[\u0301*]")
_CHTs_RE = re.compile("чц+")

RUSSIAN_WORD_CHARACTERS = (
    {chr(code) for code in range(ord("а"), ord("я") + 1)}
    | {chr(code) for code in range(ord("А"), ord("Я") + 1)}
    | {"ё", "Ё"}
)


@lru_cache(maxsize=1)
def get_russian_stop_words():
    stopwords = set()
    with resources.open_text("lunr.languages.data", "ru_stopwords.txt", encoding="utf-8") as source:
        for line in source:
            normalized = clean_russian_token(line)
            if normalized:
                stopwords.add(normalized)
    return frozenset(stopwords)


@lru_cache(maxsize=1)
def get_morph_analyzer():
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError as e:
        raise RuntimeError(
            "Russian language support requires pymorphy3 and pymorphy3-dicts-ru. "
            "Install with: pip install lunr[russian]"
        ) from e
    return MorphAnalyzer()


def clean_russian_token(text):
    decomposed = unicodedata.normalize("NFD", text)
    decomposed = _EXTRA_ORTH_RE.sub("", decomposed)
    normalized = unicodedata.normalize("NFC", decomposed).lower().strip()
    normalized = _PUNCTUATION_RE.sub("", normalized)
    normalized = _CHTs_RE.sub("ч", normalized)
    return normalized


def russian_cleanup_filter(token, i=None, tokens=None):
    return token.update(lambda s, m: clean_russian_token(s))


def russian_stop_word_filter(token, i=None, tokens=None):
    if token and str(token) not in get_russian_stop_words():
        return token


@lru_cache(maxsize=50000)
def normalize_russian_word(text):
    cleaned = clean_russian_token(text)
    if not cleaned or cleaned.isnumeric() or _CYRILLIC_RE.search(cleaned) is None:
        return cleaned

    parse = get_morph_analyzer().parse(cleaned)
    if not parse:
        return cleaned
    return parse[0].normal_form


def russian_morphology_filter(token, i=None, tokens=None):
    return token.update(lambda s, m: normalize_russian_word(s))


Pipeline.register_function(russian_cleanup_filter, "russian-cleanup")
Pipeline.register_function(russian_stop_word_filter, "stopWordFilter-ru")
Pipeline.register_function(russian_morphology_filter, "russian-morphology")
