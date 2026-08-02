"""
Input Validation Tests
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings
from app.utils.validation import (
    contains_alphabetic,
    language_warning,
    looks_like_spanish,
    spanish_score,
    validate_input_text,
)


class TestValidateInputText:
    """Unit tests for the shared validation rules."""

    def test_valid_text_is_stripped(self):
        assert validate_input_text("  Hello world  ") == "Hello world"

    def test_none_is_rejected(self):
        with pytest.raises(ValueError, match="None"):
            validate_input_text(None)

    def test_whitespace_only_is_rejected(self):
        with pytest.raises(ValueError, match="empty or whitespace only"):
            validate_input_text("   \n\t ")

    def test_over_max_is_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "MAX_INPUT_CHARS", 10)
        with pytest.raises(ValueError, match="at most 10 characters"):
            validate_input_text("a" * 11)

    def test_under_min_is_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "MIN_INPUT_CHARS", 5)
        with pytest.raises(ValueError, match="at least 5 characters"):
            validate_input_text("abc")

    def test_non_alphabetic_is_rejected(self):
        with pytest.raises(ValueError, match="alphabetic"):
            validate_input_text("12345 !!! ###")

    def test_alphabetic_helper(self):
        assert contains_alphabetic("a1") is True
        assert contains_alphabetic("123 ...") is False


class TestSpanishHeuristic:
    """Tests for the source-language sanity check."""

    def test_english_is_not_flagged(self):
        assert looks_like_spanish("Hello, how are you today?") is False
        assert (
            looks_like_spanish("The quick brown fox jumps over the lazy dog") is False
        )

    def test_spanish_characters_are_flagged(self):
        assert looks_like_spanish("Hola, ¿cómo estás?") is True

    def test_spanish_stopwords_are_flagged(self):
        assert looks_like_spanish("El libro que esta en la mesa es muy bueno") is True

    def test_score_is_bounded(self):
        assert 0.0 <= spanish_score("Hola ñ ¿qué tal?") <= 1.0
        assert spanish_score("12345") == 0.0

    def test_warning_message_mentions_the_direction(self):
        message = language_warning("Hola, ¿cómo estás?")
        assert message is not None
        assert "English to Spanish" in message

    def test_no_warning_for_english(self):
        assert language_warning("Hello there, my friend") is None

    def test_check_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "LANG_CHECK_ENABLED", False)
        assert looks_like_spanish("Hola, ¿cómo estás?") is False


@pytest.mark.asyncio
class TestValidationThroughTheApi:
    """The endpoint returns FastAPI's 422 with readable messages."""

    async def test_valid_text_passes_unchanged(self, pipeline_client):
        response = await pipeline_client.post(
            "/translate", json={"text": "Hello world."}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["translation"] == "[es] Hello world."
        assert body["warning"] is None
        assert body["chunk_count"] == 1

    async def test_whitespace_only_returns_422(self, pipeline_client):
        response = await pipeline_client.post("/translate", json={"text": "    "})

        assert response.status_code == 422
        body = response.json()
        assert "empty or whitespace only" in body["detail"]
        assert body["error"] == body["detail"]
        assert body["errors"][0]["field"] == "text"

    async def test_over_max_length_returns_422(self, pipeline_client):
        response = await pipeline_client.post(
            "/translate", json={"text": "Hello. " * 2000}
        )

        assert response.status_code == 422
        assert isinstance(response.json()["detail"], str)
        assert response.json()["detail"]

    async def test_configured_char_ceiling_returns_422(
        self, pipeline_client, monkeypatch
    ):
        monkeypatch.setattr(settings, "MAX_INPUT_CHARS", 20)
        response = await pipeline_client.post(
            "/translate", json={"text": "This sentence is definitely longer than that."}
        )

        assert response.status_code == 422
        assert "at most 20 characters" in response.json()["detail"]

    async def test_non_alphabetic_returns_422(self, pipeline_client):
        response = await pipeline_client.post(
            "/translate", json={"text": "1234 !!! ??"}
        )

        assert response.status_code == 422
        assert "alphabetic" in response.json()["detail"]

    async def test_model_not_loaded_returns_503(self, pipeline_client):
        pipeline_client._mock_manager.is_loaded = False
        try:
            response = await pipeline_client.post(
                "/translate", json={"text": "Hello world."}
            )
        finally:
            pipeline_client._mock_manager.is_loaded = True

        assert response.status_code == 503
        assert "Model is loading" in response.json()["detail"]

    async def test_database_failure_does_not_fail_the_request(self, pipeline_client):
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
                "/translate", json={"text": "Hello world."}
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.json()["translation"] == "[es] Hello world."

    async def test_service_value_error_becomes_422(self, pipeline_client):
        with patch(
            "app.api.routes.TranslationService.translate_document",
            side_effect=ValueError("Text must contain letters."),
        ):
            response = await pipeline_client.post(
                "/translate", json={"text": "Hello world."}
            )

        assert response.status_code == 422
        assert response.json()["detail"] == "Text must contain letters."

    async def test_unexpected_service_error_becomes_500(self, pipeline_client):
        with patch(
            "app.api.routes.TranslationService.translate_document",
            side_effect=RuntimeError("boom"),
        ):
            response = await pipeline_client.post(
                "/translate", json={"text": "Hello world."}
            )

        assert response.status_code == 500
        assert response.json()["detail"] == "Translation error"

    async def test_spanish_input_warns_instead_of_failing(self, pipeline_client):
        response = await pipeline_client.post(
            "/translate", json={"text": "Hola, ¿cómo estás? Muy bien, gracias."}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["translation"]
        assert body["warning"] is not None
        assert "Spanish" in body["warning"]
