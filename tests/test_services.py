"""
Translation Service Tests
"""

from unittest.mock import MagicMock, patch

import pytest


class TestTranslationServiceValidation:
    """Tests for input validation."""

    def test_validate_input_success(self):
        """Test validation with valid input."""
        from app.services.translation import TranslationService

        result = TranslationService.validate_input("Hello world")
        assert result == "Hello world"

    def test_validate_input_strips_whitespace(self):
        """Test that validation strips whitespace."""
        from app.services.translation import TranslationService

        result = TranslationService.validate_input("  Hello world  ")
        assert result == "Hello world"

    def test_validate_input_empty_string(self):
        """Test validation with empty string."""
        from app.services.translation import TranslationService

        with pytest.raises(ValueError) as excinfo:
            TranslationService.validate_input("")
        assert "empty" in str(excinfo.value).lower()

    def test_validate_input_whitespace_only(self):
        """Test validation with whitespace only."""
        from app.services.translation import TranslationService

        with pytest.raises(ValueError) as excinfo:
            TranslationService.validate_input("   ")
        assert "empty" in str(excinfo.value).lower()

    def test_validate_input_none(self):
        """Test validation with None."""
        from app.services.translation import TranslationService

        with pytest.raises(ValueError) as excinfo:
            TranslationService.validate_input(None)
        assert "None" in str(excinfo.value)


class TestTranslationServiceTranslate:
    """Tests for translation functionality."""

    def test_translate_success(self, mock_model_manager):
        """Test successful translation."""
        with patch("app.services.translation.model_manager", mock_model_manager):
            from app.services.translation import TranslationService

            mock_model_manager.translate.return_value = "Hola mundo"
            result = TranslationService.translate("Hello world")

            assert result == "Hola mundo"
            mock_model_manager.translate.assert_called_once_with("Hello world")

    def test_translate_strips_input(self, mock_model_manager):
        """Test that translation strips input whitespace."""
        with patch("app.services.translation.model_manager", mock_model_manager):
            from app.services.translation import TranslationService

            mock_model_manager.translate.return_value = "Hola mundo"
            TranslationService.translate("  Hello world  ")

            mock_model_manager.translate.assert_called_once_with("Hello world")

    def test_translate_model_not_loaded(self):
        """Test translation when model is not loaded."""
        mock_manager = MagicMock()
        mock_manager.translate.side_effect = RuntimeError("Model not loaded")

        with patch("app.services.translation.model_manager", mock_manager):
            from app.services.translation import TranslationService

            with pytest.raises(RuntimeError):
                TranslationService.translate("Hello world")

    def test_translate_empty_input(self, mock_model_manager):
        """Test translation with empty input."""
        with patch("app.services.translation.model_manager", mock_model_manager):
            from app.services.translation import TranslationService

            with pytest.raises(ValueError):
                TranslationService.translate("")

    def test_translate_long_text(self, mock_model_manager):
        """Test translation with long text."""
        with patch("app.services.translation.model_manager", mock_model_manager):
            from app.services.translation import TranslationService

            long_text = "Hello world. " * 100
            mock_model_manager.translate.return_value = "Hola mundo. " * 100

            result = TranslationService.translate(long_text)

            assert result == "Hola mundo. " * 100
            mock_model_manager.translate.assert_called_once()
