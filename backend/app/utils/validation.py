"""
Input Validation Helpers
"""

import re
from typing import Optional

from app.core.config import settings

_SPANISH_CHARS = set("ñáéíóúüÑÁÉÍÓÚÜ¿¡")

_SPANISH_STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "de",
    "del",
    "que",
    "y",
    "en",
    "es",
    "son",
    "por",
    "para",
    "con",
    "como",
    "pero",
    "más",
    "muy",
    "su",
    "sus",
    "al",
    "lo",
    "se",
    "no",
    "esta",
    "este",
    "estaba",
    "cuando",
    "porque",
    "también",
    "hola",
}

# Words shared with English, dropped to avoid false positives.
_ENGLISH_OVERLAP = {"no", "la", "un", "a", "as", "is", "in", "de", "que", "y"}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_SPANISH_WARNING = (
    "The input looks like it may already be Spanish. This model translates "
    "English to Spanish only, so the result may be unreliable."
)


def contains_alphabetic(text: str) -> bool:
    """Check if the text has at least one alphabetic character."""
    return any(ch.isalpha() for ch in text)


def validate_input_text(text: Optional[str]) -> str:
    """
    Validate and clean translation input.

    Args:
        text: Raw input text

    Returns:
        Cleaned text

    Raises:
        ValueError: If the text breaks one of the input rules
    """
    if text is None:
        raise ValueError("Text cannot be None.")

    cleaned = text.strip()

    if not cleaned:
        raise ValueError("Text cannot be empty or whitespace only.")

    if len(cleaned) < settings.MIN_INPUT_CHARS:
        raise ValueError(
            f"Text must be at least {settings.MIN_INPUT_CHARS} characters long."
        )

    if len(cleaned) > settings.MAX_INPUT_CHARS:
        raise ValueError(
            f"Text must be at most {settings.MAX_INPUT_CHARS} characters long "
            f"(received {len(cleaned)})."
        )

    if not contains_alphabetic(cleaned):
        raise ValueError(
            "Text must contain at least one alphabetic character; "
            "digits, punctuation and symbols alone cannot be translated."
        )

    return cleaned


def spanish_score(text: str) -> float:
    """Score from 0 to 1 for how Spanish the text looks."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return 0.0

    spanish_chars = sum(1 for ch in text if ch in _SPANISH_CHARS)
    char_signal = min(1.0, spanish_chars / max(1, len(words)))

    distinctive = _SPANISH_STOPWORDS - _ENGLISH_OVERLAP
    stopword_hits = sum(1 for w in words if w in distinctive)
    word_signal = stopword_hits / len(words)

    return min(1.0, char_signal + word_signal * 2)


def looks_like_spanish(text: str, threshold: float = 0.15) -> bool:
    """Check if the input is probably already Spanish."""
    if not settings.LANG_CHECK_ENABLED:
        return False
    return spanish_score(text) >= threshold


def language_warning(text: str) -> Optional[str]:
    """Get a warning for Spanish-looking input, or None."""
    if looks_like_spanish(text):
        return _SPANISH_WARNING
    return None
