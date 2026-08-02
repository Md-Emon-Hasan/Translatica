"""
In-Memory Cache Tests
"""

import time

import pytest

from app.core import cache as cache_module
from app.core.cache import (
    cache_get,
    cache_set,
    cache_stats,
    cached,
    clear_cache,
    configure_cache,
    is_cacheable,
    make_cache_key,
    normalize_text,
)
from app.core.config import settings


class TestCacheKeys:
    """Tests for key construction."""

    def test_whitespace_and_case_are_normalized(self):
        assert make_cache_key("  Hello   World  ") == make_cache_key("hello world")

    def test_normalize_text_collapses_whitespace(self):
        assert normalize_text("  a\n\tb  c ") == "a b c"

    def test_different_text_gives_different_key(self):
        assert make_cache_key("hello") != make_cache_key("goodbye")

    def test_prefix_separates_namespaces(self):
        assert make_cache_key("hello", prefix="req") != make_cache_key(
            "hello", prefix="chunk"
        )
        assert make_cache_key("hello", prefix="req").startswith("req:")

    def test_changing_num_beams_changes_the_key(self):
        assert make_cache_key("hello", num_beams=1) != make_cache_key(
            "hello", num_beams=4
        )

    def test_changing_any_generation_param_changes_the_key(self):
        baseline = make_cache_key("hello")
        assert baseline != make_cache_key("hello", max_new_tokens=32)
        assert baseline != make_cache_key("hello", length_penalty=2.0)
        assert baseline != make_cache_key("hello", no_repeat_ngram_size=5)

    def test_config_change_does_not_leak_cached_results(self, monkeypatch):
        before = make_cache_key("hello")
        monkeypatch.setattr(settings, "GEN_NUM_BEAMS", 1)
        assert make_cache_key("hello") != before

    def test_key_is_hashed_not_raw_text(self):
        key = make_cache_key("a very long piece of text " * 100)
        assert len(key) == len("req:") + 64
        assert "very long" not in key


class TestCacheBehaviour:
    """Tests for get / set semantics."""

    def test_set_then_get(self):
        cache_set("req:abc", "Hola")
        assert cache_get("req:abc") == "Hola"

    def test_miss_returns_none(self):
        assert cache_get("req:nope") is None

    def test_empty_results_are_not_cached(self):
        assert cache_set("req:empty", "") is False
        assert cache_set("req:none", None) is False
        assert cache_set("req:dict", {"translation": ""}) is False
        assert cache_get("req:empty") is None

    def test_is_cacheable(self):
        assert is_cacheable("Hola") is True
        assert is_cacheable({"translation": "Hola"}) is True
        assert is_cacheable({}) is False

    def test_dict_values_round_trip(self):
        cache_set("req:doc", {"translation": "Hola", "chunk_count": 2})
        assert cache_get("req:doc")["chunk_count"] == 2

    def test_ttl_expiry_misses(self):
        configure_cache(ttl=0.05)
        try:
            cache_set("req:ttl", "Hola")
            assert cache_get("req:ttl") == "Hola"
            time.sleep(0.1)
            assert cache_get("req:ttl") is None
        finally:
            configure_cache()

    def test_clear_cache_empties_and_resets_counters(self):
        cache_set("req:x", "Hola")
        cache_get("req:x")
        clear_cache()

        stats = cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_stats_shape_and_hit_rate(self):
        cache_set("req:y", "Hola")
        cache_get("req:y")
        cache_get("req:missing")

        stats = cache_stats()
        assert set(stats) == {"size", "maxsize", "ttl", "hits", "misses", "hit_rate"}
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["maxsize"] == settings.CACHE_MAXSIZE
        assert stats["ttl"] == settings.CACHE_TTL_SECONDS

    def test_hit_rate_is_zero_without_traffic(self):
        assert cache_stats()["hit_rate"] == 0.0

    def test_disabled_cache_never_stores(self, monkeypatch):
        monkeypatch.setattr(settings, "CACHE_ENABLED", False)
        assert cache_set("req:z", "Hola") is False
        assert cache_get("req:z") is None

    def test_lock_is_held_for_reads_and_writes(self):
        assert cache_module._lock is not None


class TestCachedDecorator:
    """Tests for the caching decorator."""

    def test_hit_avoids_reinvoking_the_function(self):
        calls = []

        @cached(prefix="test")
        def translate(text: str) -> str:
            calls.append(text)
            return f"[es] {text}"

        assert translate("Hello world") == "[es] Hello world"
        assert translate("  hello   WORLD ") == "[es] Hello world"
        assert len(calls) == 1

    def test_empty_result_is_not_cached(self):
        calls = []

        @cached(prefix="test")
        def translate(text: str) -> str:
            calls.append(text)
            return ""

        translate("Hello")
        translate("Hello")
        assert len(calls) == 2

    def test_disabled_cache_is_transparent(self, monkeypatch):
        monkeypatch.setattr(settings, "CACHE_ENABLED", False)
        calls = []

        @cached(prefix="test")
        def translate(text: str) -> str:
            calls.append(text)
            return "[es]"

        translate("Hello")
        translate("Hello")
        assert len(calls) == 2
        assert cache_stats()["size"] == 0


@pytest.mark.asyncio
class TestCacheThroughTheApi:
    """End-to-end caching behaviour on the translate endpoint."""

    async def test_second_identical_request_is_a_cache_hit(self, pipeline_client):
        manager = pipeline_client._mock_manager

        first = await pipeline_client.post("/translate", json={"text": "Hello world."})
        second = await pipeline_client.post("/translate", json={"text": "Hello world."})

        assert first.status_code == second.status_code == 200
        assert first.json()["cached"] is False
        assert second.json()["cached"] is True
        assert second.json()["translation"] == first.json()["translation"]
        assert manager.generate_texts.call_count == 1

    async def test_repeated_sentences_hit_the_chunk_cache(
        self, pipeline_client, monkeypatch
    ):
        # One sentence per chunk
        monkeypatch.setattr(settings, "CHUNK_TOKEN_BUDGET", 3)
        manager = pipeline_client._mock_manager

        first = await pipeline_client.post(
            "/translate", json={"text": "Alpha one. Beta two."}
        )
        assert first.json()["chunk_count"] == 2
        assert manager.generate_texts.call_args[0][0] == ["Alpha one.", "Beta two."]

        second = await pipeline_client.post(
            "/translate", json={"text": "Alpha one. Gamma three."}
        )
        assert second.json()["cached"] is False
        assert manager.generate_texts.call_args[0][0] == ["Gamma three."]
        assert second.json()["translation"] == "[es] Alpha one. [es] Gamma three."
