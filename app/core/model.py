"""
ML Model Manager for Translation
"""

from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger("model")


class ModelManager:
    """Manages the translation model and tokenizer."""

    def __init__(self):
        """Initialize the model manager."""
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._device: Optional[torch.device] = None
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    @property
    def device(self) -> torch.device:
        """Get the device for inference."""
        if self._device is None:
            if settings.DEVICE == "auto":
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self._device = torch.device(settings.DEVICE)
        return self._device

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer

    @property
    def model(self) -> AutoModelForSeq2SeqLM:
        """Get the model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def load(self) -> None:
        """Load the tokenizer and model."""
        if self._is_loaded:
            logger.info("Model already loaded.")
            return

        logger.info(f"Loading tokenizer from {settings.TOKENIZER_PATH}")
        self._tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_PATH)

        logger.info(f"Loading model from {settings.MODEL_PATH}")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_PATH)

        logger.info(f"Moving model to device: {self.device}")
        self._model.to(self.device)
        self._model.eval()

        self._is_loaded = True
        logger.info("Model loaded successfully!")

    def translate(self, text: str) -> str:
        """
        Translate text from English to Spanish.

        Args:
            text: English text to translate

        Returns:
            Spanish translation
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.MAX_INPUT_LENGTH,
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            translated_tokens = self.model.generate(
                inputs["input_ids"],
                max_length=settings.MAX_OUTPUT_LENGTH,
                num_beams=settings.NUM_BEAMS,
                early_stopping=True,
            )

        # Decode and return
        translated_text = self.tokenizer.decode(
            translated_tokens[0], skip_special_tokens=True
        )
        return translated_text

    def cleanup(self) -> None:
        """Cleanup model resources."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._is_loaded = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model resources cleaned up.")


# Global model manager instance
model_manager = ModelManager()
