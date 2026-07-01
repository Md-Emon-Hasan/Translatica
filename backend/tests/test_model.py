"""
Unit Tests for ModelManager
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.core.model import ModelManager


@pytest.fixture
def mock_transformers():
    with (
        patch("app.core.model.AutoTokenizer") as mock_tokenizer,
        patch("app.core.model.AutoModelForSeq2SeqLM") as mock_model,
        patch("app.core.model.PeftModel") as mock_peft,
        patch("app.core.model.torch") as mock_torch,
    ):

        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        # PeftModel.from_pretrained(...).merge_and_unload() returns the final
        # model used for inference.
        merged_model = MagicMock()
        mock_peft.from_pretrained.return_value.merge_and_unload.return_value = (
            merged_model
        )

        mock_torch.device = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        yield {
            "tokenizer": mock_tokenizer,
            "model": mock_model,
            "peft": mock_peft,
            "merged": merged_model,
            "torch": mock_torch,
        }


def test_initialization():
    manager = ModelManager()
    assert not manager.is_loaded
    assert manager._tokenizer is None
    assert manager._model is None


def test_device_cpu(mock_transformers):
    manager = ModelManager()
    settings.DEVICE = "auto"
    mock_transformers["torch"].cuda.is_available.return_value = False

    device = manager.device
    assert device is not None
    # Verify torch.device called with cpu logic handled by mock or integration


def test_device_cuda(mock_transformers):
    manager = ModelManager()
    settings.DEVICE = "auto"
    mock_transformers["torch"].cuda.is_available.return_value = True

    _ = manager.device
    mock_transformers["torch"].device.assert_called_with("cuda")


def test_device_explicit(mock_transformers):
    manager = ModelManager()
    settings.DEVICE = "cpu"

    _ = manager.device
    mock_transformers["torch"].device.assert_called_with("cpu")


def test_load_already_loaded(mock_transformers):
    manager = ModelManager()
    manager._is_loaded = True

    manager.load()

    mock_transformers["tokenizer"].from_pretrained.assert_not_called()


def test_load_success(mock_transformers):
    manager = ModelManager()

    manager.load()

    assert manager.is_loaded
    mock_transformers["tokenizer"].from_pretrained.assert_called_once()
    # Base model loaded, then LoRA adapter attached and merged.
    mock_transformers["model"].from_pretrained.assert_called_once()
    mock_transformers["peft"].from_pretrained.assert_called_once()
    assert manager._model is mock_transformers["merged"]
    manager._model.to.assert_called_once()
    manager._model.eval.assert_called_once()


def test_translate_not_loaded():
    manager = ModelManager()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        manager.translate("hello")


def test_translate_success(mock_transformers):
    manager = ModelManager()
    manager.load()

    # Mock tokenization output
    mock_input = MagicMock()
    mock_input.to.return_value = {"input_ids": "fake_ids"}
    manager._tokenizer.return_value = mock_input

    # Mock generation
    manager._model.generate.return_value = ["fake_output"]

    # Mock decoding
    manager._tokenizer.decode.return_value = "Hola"

    result = manager.translate("Hello")
    assert result == "Hola"


def test_translate_applies_t5_prefix(mock_transformers):
    manager = ModelManager()
    manager.load()

    mock_input = MagicMock()
    mock_input.to.return_value = {"input_ids": "fake_ids"}
    manager._tokenizer.return_value = mock_input
    manager._model.generate.return_value = ["fake_output"]
    manager._tokenizer.decode.return_value = "Hola"

    manager.translate("Hello")

    # The text passed to the tokenizer must start with the T5 task prefix.
    called_text = manager._tokenizer.call_args[0][0]
    assert called_text == settings.TRANSLATION_PREFIX + "Hello"


def test_cleanup(mock_transformers):
    manager = ModelManager()
    manager.load()

    manager.cleanup()

    assert not manager.is_loaded
    assert manager._model is None
    assert manager._tokenizer is None
    mock_transformers["torch"].cuda.empty_cache.assert_not_called()


def test_cleanup_cuda(mock_transformers):
    manager = ModelManager()
    manager.load()

    # Mock cuda available
    mock_transformers["torch"].cuda.is_available.return_value = True

    manager.cleanup()

    mock_transformers["torch"].cuda.empty_cache.assert_called_once()


def test_get_tokenizer_error():
    manager = ModelManager()
    with pytest.raises(RuntimeError):
        _ = manager.tokenizer


def test_get_model_error():
    manager = ModelManager()
    with pytest.raises(RuntimeError):
        _ = manager.model
