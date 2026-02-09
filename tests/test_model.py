"""
Model Manager Tests
"""

from unittest.mock import patch

import pytest
import torch


class TestModelManager:
    """Tests for ModelManager class."""

    def test_model_manager_initialization(self):
        """Test model manager initializes correctly."""
        from app.core.model import ModelManager

        manager = ModelManager()

        assert manager._tokenizer is None
        assert manager._model is None
        assert manager._is_loaded is False

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        from app.core.model import ModelManager

        manager = ModelManager()

        assert manager.is_loaded is False
        manager._is_loaded = True
        assert manager.is_loaded is True

    def test_device_property_auto(self):
        """Test device property with auto detection."""
        from app.core.model import ModelManager

        manager = ModelManager()

        with patch("app.core.model.settings") as mock_settings:
            mock_settings.DEVICE = "auto"
            manager._device = None  # Reset cached device

            device = manager.device

            assert isinstance(device, torch.device)
            # Should be either cuda or cpu
            assert device.type in ["cuda", "cpu"]

    def test_device_property_cpu(self):
        """Test device property with explicit CPU."""
        from app.core.model import ModelManager

        with patch("app.core.model.settings") as mock_settings:
            mock_settings.DEVICE = "cpu"

            manager = ModelManager()
            device = manager.device

            assert device.type == "cpu"

    def test_tokenizer_property_not_loaded(self):
        """Test tokenizer property raises when not loaded."""
        from app.core.model import ModelManager

        manager = ModelManager()

        with pytest.raises(RuntimeError) as excinfo:
            _ = manager.tokenizer
        assert "not loaded" in str(excinfo.value).lower()

    def test_model_property_not_loaded(self):
        """Test model property raises when not loaded."""
        from app.core.model import ModelManager

        manager = ModelManager()

        with pytest.raises(RuntimeError) as excinfo:
            _ = manager.model
        assert "not loaded" in str(excinfo.value).lower()

    def test_load_model(self, mock_tokenizer, mock_model):
        """Test loading model and tokenizer."""
        from app.core.model import ModelManager

        with (
            patch("app.core.model.AutoTokenizer") as MockTokenizer,
            patch("app.core.model.AutoModelForSeq2SeqLM") as MockModel,
            patch("app.core.model.settings") as mock_settings,
        ):

            MockTokenizer.from_pretrained.return_value = mock_tokenizer
            MockModel.from_pretrained.return_value = mock_model
            mock_settings.TOKENIZER_PATH = "/path/to/tokenizer"
            mock_settings.MODEL_PATH = "/path/to/model"
            mock_settings.DEVICE = "cpu"

            manager = ModelManager()
            manager.load()

            assert manager.is_loaded is True
            MockTokenizer.from_pretrained.assert_called_once()
            MockModel.from_pretrained.assert_called_once()

    def test_load_model_already_loaded(self, mock_tokenizer, mock_model):
        """Test loading model when already loaded."""
        from app.core.model import ModelManager

        with (
            patch("app.core.model.AutoTokenizer") as MockTokenizer,
            patch("app.core.model.AutoModelForSeq2SeqLM") as MockModel,
            patch("app.core.model.settings") as mock_settings,
        ):

            MockTokenizer.from_pretrained.return_value = mock_tokenizer
            MockModel.from_pretrained.return_value = mock_model
            mock_settings.TOKENIZER_PATH = "/path/to/tokenizer"
            mock_settings.MODEL_PATH = "/path/to/model"
            mock_settings.DEVICE = "cpu"

            manager = ModelManager()
            manager.load()
            manager.load()  # Second call should not reload

            assert MockTokenizer.from_pretrained.call_count == 1

    def test_translate_not_loaded(self):
        """Test translate raises when model not loaded."""
        from app.core.model import ModelManager

        manager = ModelManager()

        with pytest.raises(RuntimeError) as excinfo:
            manager.translate("Hello world")
        assert "not loaded" in str(excinfo.value).lower()

    def test_cleanup(self, mock_tokenizer, mock_model):
        """Test cleanup releases resources."""
        from app.core.model import ModelManager

        with (
            patch("app.core.model.AutoTokenizer") as MockTokenizer,
            patch("app.core.model.AutoModelForSeq2SeqLM") as MockModel,
            patch("app.core.model.settings") as mock_settings,
        ):

            MockTokenizer.from_pretrained.return_value = mock_tokenizer
            MockModel.from_pretrained.return_value = mock_model
            mock_settings.TOKENIZER_PATH = "/path/to/tokenizer"
            mock_settings.MODEL_PATH = "/path/to/model"
            mock_settings.DEVICE = "cpu"

            manager = ModelManager()
            manager.load()

            assert manager.is_loaded is True

            manager.cleanup()

            assert manager.is_loaded is False
            assert manager._model is None
            assert manager._tokenizer is None


class TestGlobalModelManager:
    """Tests for global model manager instance."""

    def test_global_instance_exists(self):
        """Test that global model manager instance exists."""
        from app.core.model import model_manager

        assert model_manager is not None

    def test_global_instance_is_model_manager(self):
        """Test that global instance is ModelManager type."""
        from app.core.model import ModelManager, model_manager

        assert isinstance(model_manager, ModelManager)
