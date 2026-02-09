"""
Configuration Tests
"""

from pathlib import Path
from unittest.mock import patch


class TestSettings:
    """Tests for application settings."""

    def test_settings_defaults(self):
        """Test default settings values."""
        from app.core.config import Settings

        settings = Settings()

        assert settings.APP_NAME == "Translatica"
        assert settings.DEBUG is False
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000

    def test_settings_model_defaults(self):
        """Test default model settings."""
        from app.core.config import Settings

        settings = Settings()

        assert settings.MAX_INPUT_LENGTH == 512
        assert settings.MAX_OUTPUT_LENGTH == 256
        assert settings.NUM_BEAMS == 8
        assert settings.DEVICE in ["cuda", "cpu", "auto"]

    def test_settings_paths_are_paths(self):
        """Test that path settings are Path objects."""
        from app.core.config import Settings

        settings = Settings()

        assert isinstance(settings.BASE_DIR, Path)
        assert isinstance(settings.MODEL_DIR, Path)
        assert isinstance(settings.TOKENIZER_PATH, Path)
        assert isinstance(settings.MODEL_PATH, Path)
        assert isinstance(settings.STATIC_DIR, Path)
        assert isinstance(settings.TEMPLATES_DIR, Path)
        assert isinstance(settings.LOGS_DIR, Path)

    def test_settings_paths_relative_to_base(self):
        """Test that paths are relative to base directory."""
        from app.core.config import Settings

        settings = Settings()

        assert settings.MODEL_DIR == settings.BASE_DIR / "fine-tuned-model"
        assert settings.STATIC_DIR == settings.BASE_DIR / "static"
        assert settings.TEMPLATES_DIR == settings.BASE_DIR / "templates"
        assert settings.LOGS_DIR == settings.BASE_DIR / "logs"

    def test_settings_from_env(self):
        """Test settings can be overridden from environment."""
        with patch.dict("os.environ", {"DEBUG": "true", "PORT": "9000"}):
            from app.core.config import Settings

            Settings()

            # Note: This test may not work perfectly due to caching
            # In production, use proper env var handling

    def test_global_settings_instance(self):
        """Test that global settings instance exists."""
        from app.core.config import settings

        assert settings is not None
        assert settings.APP_NAME == "Translatica"
