"""
Translation Service
"""

from app.core.model import model_manager
from app.utils.logger import get_logger

logger = get_logger("translation_service")


class TranslationService:
    """Service layer for translation operations."""

    @staticmethod
    def validate_input(text: str) -> str:
        """
        Validate and clean input text.

        Args:
            text: Input text to validate

        Returns:
            Cleaned text

        Raises:
            ValueError: If text is empty or invalid
        """
        if text is None:
            raise ValueError("Text cannot be None")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty")

        return cleaned

    @staticmethod
    def translate(text: str) -> str:
        """
        Translate English text to Spanish.

        Args:
            text: English text to translate

        Returns:
            Spanish translation

        Raises:
            ValueError: If input is invalid
            RuntimeError: If model is not loaded
        """
        # Validate input
        cleaned_text = TranslationService.validate_input(text)

        logger.info(f"Translating text of length {len(cleaned_text)}")

        # Perform translation using model manager
        translation = model_manager.translate(cleaned_text)

        logger.info(f"Translation completed, output length {len(translation)}")

        return translation
