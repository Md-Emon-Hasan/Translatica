"""
Thread-Safe In-Memory Translation Cache
"""

import functools
import hashlib
import threading
from typing import Any, Callable, Dict, Optional

from cachetools import TTLCache

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger("cache")

# TTLCache is not thread-safe, so every access below goes through this lock.
_lock = threading.Lock()
_cache: TTLCache = TTLCache(
    maxsize=settings.CACHE_MAXSIZE, ttl=settings.CACHE_TTL_SECONDS
)
_hits: int = 0
_misses: int = 0


def is_enabled() -> bool:
    """Check if caching is switched on."""
    return bool(settings.CACHE_ENABLED)


def normalize_text(text: str) -> str:
    """Collapse whitespace runs so equivalent inputs share a key."""
    return " ".join(text.strip().split())


def make_cache_key(
    text: str,
    prefix: str = "req",
    num_beams: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
) -> str:
    """
    Build a cache key from the text and the generation parameters.

    The generation parameters belong in the key: changing num_beams changes
    the decoding strategy, and old results must not leak through the cache.
    """
    normalized = normalize_text(text).lower()
    num_beams = settings.GEN_NUM_BEAMS if num_beams is None else num_beams
    max_new_tokens = (
        settings.GEN_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    )
    length_penalty = (
        settings.GEN_LENGTH_PENALTY if length_penalty is None else length_penalty
    )
    no_repeat_ngram_size = (
        settings.GEN_NO_REPEAT_NGRAM_SIZE
        if no_repeat_ngram_size is None
        else no_repeat_ngram_size
    )

    raw = (
        f"{normalized}|{num_beams}|{max_new_tokens}"
        f"|{length_penalty}|{no_repeat_ngram_size}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def is_cacheable(value: Any) -> bool:
    """Check if a result is worth storing."""
    if not value:
        return False
    if isinstance(value, dict):
        return bool(value.get("translation"))
    return True


def cache_get(key: str) -> Optional[Any]:
    """Look up a key and update the hit/miss counters."""
    global _hits, _misses

    if not is_enabled():
        return None

    with _lock:
        try:
            value = _cache[key]
        except KeyError:
            _misses += 1
            return None
        _hits += 1
        return value


def cache_set(key: str, value: Any) -> bool:
    """Store a value, skipping empty and error results."""
    if not is_enabled() or not is_cacheable(value):
        return False

    with _lock:
        _cache[key] = value
    return True


def clear_cache() -> None:
    """Empty the cache and reset the counters."""
    global _hits, _misses

    with _lock:
        _cache.clear()
        _hits = 0
        _misses = 0


def configure_cache(maxsize: Optional[int] = None, ttl: Optional[float] = None) -> None:
    """Rebuild the cache with a new size and TTL."""
    global _cache, _hits, _misses

    with _lock:
        _cache = TTLCache(
            maxsize=settings.CACHE_MAXSIZE if maxsize is None else maxsize,
            ttl=settings.CACHE_TTL_SECONDS if ttl is None else ttl,
        )
        _hits = 0
        _misses = 0


def cache_stats() -> Dict[str, Any]:
    """Get current cache statistics."""
    with _lock:
        size = len(_cache)
        maxsize = _cache.maxsize
        ttl = _cache.ttl
        hits = _hits
        misses = _misses

    total = hits + misses

    return {
        "size": size,
        "maxsize": maxsize,
        "ttl": ttl,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0.0,
    }


def cached(prefix: str = "req") -> Callable:
    """Cache a ``f(text) -> result`` function, or pass through when disabled."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(text: str, *args: Any, **kwargs: Any) -> Any:
            if not is_enabled():
                return func(text, *args, **kwargs)

            key = make_cache_key(text, prefix=prefix)
            hit = cache_get(key)
            if hit is not None:
                logger.info(f"Cache hit for key {key}")
                return hit

            result = func(text, *args, **kwargs)
            cache_set(key, result)
            return result

        return wrapper

    return decorator
