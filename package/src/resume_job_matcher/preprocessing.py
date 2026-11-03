"""Job-description preprocessing utilities used by the matching pipeline."""

import html
import re

from bs4 import BeautifulSoup

_UNICODE_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "-",
        "\u2013": "-",
        "\u2026": "...",
    }
)
_WHITESPACE_RE = re.compile(r"[\s\xa0]+")


def _unescape(raw: str) -> str:
    for _ in range(5):
        decoded = html.unescape(raw)
        if decoded == raw:
            break
        raw = decoded
    return raw


def preprocess_description(raw: str | None) -> str:
    """Convert a raw HTML job description into normalized plain text."""
    if not raw:
        return ""

    text = _unescape(raw)
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(["iframe", "img"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.translate(_UNICODE_TABLE)
