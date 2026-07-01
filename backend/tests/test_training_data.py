"""
Tests for training.data module
"""

from unittest.mock import MagicMock, patch


class TestLoadTranslationDataset:
    """Tests for load_translation_dataset function."""

    @patch("training.data.load_dataset")
    def test_load_translation_dataset_default(self, mock_load_dataset):
        """Test loading dataset with default parameters."""
        from training.data import load_translation_dataset

        mock_load_dataset.return_value = {"train": []}
        result = load_translation_dataset()

        mock_load_dataset.assert_called_once_with("Helsinki-NLP/opus_books", "en-es")
        assert result == {"train": []}

    @patch("training.data.load_dataset")
    def test_load_translation_dataset_custom(self, mock_load_dataset):
        """Test loading dataset with custom parameters."""
        from training.data import load_translation_dataset

        mock_load_dataset.return_value = {"train": []}
        load_translation_dataset("custom_dataset", "fr-de")

        mock_load_dataset.assert_called_once_with("custom_dataset", "fr-de")


class TestLoadTokenizer:
    """Tests for load_tokenizer function."""

    @patch("training.data.AutoTokenizer.from_pretrained")
    def test_load_tokenizer(self, mock_tokenizer):
        """Test loading tokenizer."""
        from training.data import load_tokenizer

        mock_tokenizer.return_value = MagicMock()
        load_tokenizer("test-checkpoint")

        mock_tokenizer.assert_called_once_with("test-checkpoint")


class TestPreprocessTranslationExamples:
    """Tests for preprocess_translation_examples function."""

    def test_preprocess_translation_examples(self):
        """Test preprocessing translation examples."""
        from training.data import preprocess_translation_examples

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        examples = {
            "translation": [
                {"en": "Hello", "es": "Hola"},
                {"en": "World", "es": "Mundo"},
            ]
        }

        result = preprocess_translation_examples(examples, mock_tokenizer)

        assert "labels" in result

        # The source inputs must be prefixed with the T5 task prefix.
        source_inputs = mock_tokenizer.call_args_list[0].args[0]
        assert all(
            s.startswith("translate English to Spanish: ") for s in source_inputs
        )

        # Labels must be tokenized via the text_target keyword.
        assert "text_target" in mock_tokenizer.call_args_list[1].kwargs


class TestPrepareDataset:
    """Tests for prepare_dataset function."""

    @patch("training.data.load_translation_dataset")
    @patch("training.data.load_tokenizer")
    def test_prepare_dataset(self, mock_load_tokenizer, mock_load_dataset):
        """Test preparing full dataset (split first, then map)."""
        from training.data import prepare_dataset

        # Final tokenized DatasetDict returned by .map()
        mock_tokenized = {"train": [], "test": []}

        # train_test_split() returns an object whose .map() gives the tokenized data
        mock_split = MagicMock()
        mock_split.map.return_value = mock_tokenized

        # dataset["train"] -> object with train_test_split()
        mock_train_subset = MagicMock()
        mock_train_subset.train_test_split.return_value = mock_split

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=mock_train_subset)

        # Setup mocks
        mock_load_dataset.return_value = mock_dataset
        mock_load_tokenizer.return_value = MagicMock()

        result, tokenizer = prepare_dataset("test-checkpoint")

        mock_train_subset.train_test_split.assert_called_once()
        mock_split.map.assert_called_once()
        assert "train" in result
        assert "test" in result
