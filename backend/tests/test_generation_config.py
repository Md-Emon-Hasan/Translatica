"""
Generation Configuration (Beam Search) Tests
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.core.model import ModelManager, build_generation_kwargs


@pytest.fixture
def loaded_manager():
    """A ModelManager with mocked tokenizer / model, marked as loaded."""
    with (
        patch("app.core.model.AutoTokenizer"),
        patch("app.core.model.AutoModelForSeq2SeqLM"),
        patch("app.core.model.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = False
        manager = ModelManager()
        manager.load()

        tokenized = MagicMock()
        tokenized.to.return_value = {"input_ids": "ids", "attention_mask": "mask"}
        manager._tokenizer.return_value = tokenized
        manager._tokenizer.batch_decode.return_value = ["Hola"]
        manager._model.generate.return_value = [[1, 2, 3]]

        yield manager


class TestBuildGenerationKwargs:
    """Tests for configuration-driven generation parameters."""

    def test_beam_search_defaults(self):
        kwargs = build_generation_kwargs()

        assert kwargs["num_beams"] == settings.GEN_NUM_BEAMS == 4
        assert kwargs["max_new_tokens"] == settings.GEN_MAX_NEW_TOKENS == 256
        assert kwargs["length_penalty"] == settings.GEN_LENGTH_PENALTY == 1.0
        assert kwargs["early_stopping"] is settings.GEN_EARLY_STOPPING is True
        assert kwargs["no_repeat_ngram_size"] == settings.GEN_NO_REPEAT_NGRAM_SIZE == 3

    def test_output_is_deterministic(self):
        assert build_generation_kwargs()["do_sample"] is False

    def test_num_beams_one_falls_back_to_greedy(self, monkeypatch):
        monkeypatch.setattr(settings, "GEN_NUM_BEAMS", 1)
        kwargs = build_generation_kwargs()

        assert kwargs["num_beams"] == 1
        assert kwargs["max_length"] == settings.MAX_OUTPUT_LENGTH
        assert kwargs["do_sample"] is False
        assert "no_repeat_ngram_size" not in kwargs
        assert "length_penalty" not in kwargs
        assert "max_new_tokens" not in kwargs

    def test_config_overrides_are_honoured(self, monkeypatch):
        monkeypatch.setattr(settings, "GEN_NUM_BEAMS", 6)
        monkeypatch.setattr(settings, "GEN_MAX_NEW_TOKENS", 64)
        monkeypatch.setattr(settings, "GEN_LENGTH_PENALTY", 2.5)
        monkeypatch.setattr(settings, "GEN_NO_REPEAT_NGRAM_SIZE", 5)
        monkeypatch.setattr(settings, "GEN_EARLY_STOPPING", False)

        kwargs = build_generation_kwargs()

        assert kwargs["num_beams"] == 6
        assert kwargs["max_new_tokens"] == 64
        assert kwargs["length_penalty"] == 2.5
        assert kwargs["no_repeat_ngram_size"] == 5
        assert kwargs["early_stopping"] is False

    def test_explicit_overrides_win(self):
        assert build_generation_kwargs(num_beams=2)["num_beams"] == 2


class TestGenerateTexts:
    """Tests for the batched generation entry point."""

    def test_generate_called_with_configured_beam_params(self, loaded_manager):
        loaded_manager.generate_texts(["Hello world"])

        _, kwargs = loaded_manager._model.generate.call_args
        assert kwargs["num_beams"] == settings.GEN_NUM_BEAMS
        assert kwargs["max_new_tokens"] == settings.GEN_MAX_NEW_TOKENS
        assert kwargs["length_penalty"] == settings.GEN_LENGTH_PENALTY
        assert kwargs["no_repeat_ngram_size"] == settings.GEN_NO_REPEAT_NGRAM_SIZE
        assert kwargs["do_sample"] is False
        assert kwargs["input_ids"] == "ids"
        assert kwargs["attention_mask"] == "mask"

    def test_greedy_path_when_num_beams_is_one(self, loaded_manager, monkeypatch):
        monkeypatch.setattr(settings, "GEN_NUM_BEAMS", 1)
        loaded_manager.generate_texts(["Hello world"])

        _, kwargs = loaded_manager._model.generate.call_args
        assert kwargs["num_beams"] == 1
        assert kwargs["max_length"] == settings.MAX_OUTPUT_LENGTH
        assert "no_repeat_ngram_size" not in kwargs

    def test_returns_batch_decoded_output(self, loaded_manager):
        assert loaded_manager.generate_texts(["Hello world"]) == ["Hola"]

    def test_empty_input_skips_the_model(self, loaded_manager):
        assert loaded_manager.generate_texts([]) == []
        loaded_manager._model.generate.assert_not_called()

    def test_padding_is_enabled_for_batching(self, loaded_manager):
        loaded_manager.generate_texts(["one", "two"])

        _, tokenizer_kwargs = loaded_manager._tokenizer.call_args
        assert tokenizer_kwargs["padding"] is True
        assert tokenizer_kwargs["max_length"] == settings.MAX_INPUT_TOKENS

    def test_requires_loaded_model(self):
        with pytest.raises(RuntimeError, match="Model not loaded"):
            ModelManager().generate_texts(["Hello"])

    def test_count_tokens_uses_tokenizer(self, loaded_manager):
        loaded_manager._tokenizer.return_value = {"input_ids": [1, 2, 3, 4]}
        assert loaded_manager.count_tokens("a b c d") == 4
