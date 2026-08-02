"""
Batch Translation Tests
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings

pytestmark = pytest.mark.asyncio


class TestBatchEndpoint:
    """Tests for POST /translate/batch."""

    async def test_order_is_preserved(self, pipeline_client):
        texts = [f"Sentence number {i}." for i in range(5)]
        response = await pipeline_client.post("/translate/batch", json={"texts": texts})

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 5
        assert body["succeeded"] == 5
        assert body["failed"] == 0
        assert [item["index"] for item in body["results"]] == [0, 1, 2, 3, 4]
        assert [item["translation"] for item in body["results"]] == [
            f"[es] {text}" for text in texts
        ]

    async def test_single_generate_call_for_the_whole_batch(self, pipeline_client):
        manager = pipeline_client._mock_manager
        texts = [f"Sentence number {i}." for i in range(5)]

        await pipeline_client.post("/translate/batch", json={"texts": texts})

        assert manager.generate_texts.call_count == 1
        assert len(manager.generate_texts.call_args[0][0]) == 5

    async def test_one_invalid_item_does_not_fail_the_batch(self, pipeline_client):
        texts = ["Hello world.", "   ", "1234 !!!", "Goodbye world.", "Fine too."]
        response = await pipeline_client.post("/translate/batch", json={"texts": texts})

        assert response.status_code == 200
        body = response.json()
        results = body["results"]

        assert body["succeeded"] == 3
        assert body["failed"] == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "empty or whitespace only" in results[1]["error"]
        assert results[2]["success"] is False
        assert "alphabetic" in results[2]["error"]
        assert results[3]["translation"] == "[es] Goodbye world."
        assert [item["index"] for item in results] == [0, 1, 2, 3, 4]

    async def test_cached_items_skip_inference(self, pipeline_client):
        manager = pipeline_client._mock_manager

        await pipeline_client.post("/translate", json={"text": "Hello world."})
        assert manager.generate_texts.call_count == 1

        response = await pipeline_client.post(
            "/translate/batch",
            json={"texts": ["Hello world.", "Something new."]},
        )

        body = response.json()
        assert body["results"][0]["cached"] is True
        assert body["results"][1]["cached"] is False
        assert manager.generate_texts.call_args[0][0] == ["Something new."]

    async def test_long_items_are_chunked_into_the_same_batch(
        self, pipeline_client, monkeypatch
    ):
        monkeypatch.setattr(settings, "CHUNK_TOKEN_BUDGET", 3)
        manager = pipeline_client._mock_manager

        response = await pipeline_client.post(
            "/translate/batch",
            json={"texts": ["Alpha one. Beta two.", "Gamma three."]},
        )

        body = response.json()
        assert body["results"][0]["chunk_count"] == 2
        assert body["results"][1]["chunk_count"] == 1
        assert manager.generate_texts.call_count == 1
        assert manager.generate_texts.call_args[0][0] == [
            "Alpha one.",
            "Beta two.",
            "Gamma three.",
        ]

    async def test_paragraph_breaks_survive_in_batch_items(self, pipeline_client):
        response = await pipeline_client.post(
            "/translate/batch",
            json={"texts": ["First paragraph.\n\nSecond paragraph."]},
        )

        translation = response.json()["results"][0]["translation"]
        assert translation.count("\n\n") == 1

    async def test_over_size_batch_is_rejected(self, pipeline_client):
        texts = [f"Item {i}." for i in range(settings.MAX_BATCH_SIZE + 1)]
        response = await pipeline_client.post("/translate/batch", json={"texts": texts})

        assert response.status_code == 422
        assert "at most" in response.json()["detail"]

    async def test_over_total_chars_batch_is_rejected(
        self, pipeline_client, monkeypatch
    ):
        monkeypatch.setattr(settings, "MAX_BATCH_CHARS", 20)
        response = await pipeline_client.post(
            "/translate/batch",
            json={"texts": ["A long enough sentence.", "And another one."]},
        )

        assert response.status_code == 422
        assert "characters in total" in response.json()["detail"]

    async def test_empty_batch_is_rejected(self, pipeline_client):
        response = await pipeline_client.post("/translate/batch", json={"texts": []})

        assert response.status_code == 422
        assert "at least one text" in response.json()["detail"]

    async def test_unexpected_error_becomes_500(self, pipeline_client):
        with patch(
            "app.api.routes.TranslationService.translate_batch",
            side_effect=RuntimeError("boom"),
        ):
            response = await pipeline_client.post(
                "/translate/batch", json={"texts": ["Hello world."]}
            )

        assert response.status_code == 500
        assert response.json()["detail"] == "Translation error"

    async def test_database_failure_does_not_fail_the_batch(self, pipeline_client):
        from app.core.database import get_db
        from app.main import app

        async def broken_db():
            session = MagicMock()
            session.add = MagicMock()
            session.commit = AsyncMock(side_effect=RuntimeError("db is down"))
            yield session

        app.dependency_overrides[get_db] = broken_db
        try:
            response = await pipeline_client.post(
                "/translate/batch", json={"texts": ["Hello world."]}
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.json()["succeeded"] == 1

    async def test_returns_503_when_the_model_is_not_loaded(self, pipeline_client):
        pipeline_client._mock_manager.is_loaded = False
        try:
            response = await pipeline_client.post(
                "/translate/batch", json={"texts": ["Hello world."]}
            )
        finally:
            pipeline_client._mock_manager.is_loaded = True

        assert response.status_code == 503
