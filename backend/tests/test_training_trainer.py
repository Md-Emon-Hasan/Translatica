"""
Tests for training.trainer module
"""

import sys
from unittest.mock import MagicMock, patch

# Mock evaluate module before importing trainer
sys.modules["evaluate"] = MagicMock()


class TestGetTrainingArguments:
    """Tests for get_training_arguments function."""

    def test_get_training_arguments_default(self):
        """Test training arguments with defaults."""
        from training.trainer import get_training_arguments

        args = get_training_arguments()

        assert args.output_dir == "./results"
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 16
        assert args.learning_rate == 2e-5

    def test_get_training_arguments_custom(self):
        """Test training arguments with custom values."""
        from training.trainer import get_training_arguments

        args = get_training_arguments(
            output_dir="./custom", num_epochs=5, batch_size=32, learning_rate=1e-4
        )

        assert args.output_dir == "./custom"
        assert args.num_train_epochs == 5
        assert args.per_device_train_batch_size == 32
        assert args.learning_rate == 1e-4


class TestGetComputeMetrics:
    """Tests for get_compute_metrics function."""

    def test_get_compute_metrics(self):
        """Test compute metrics function creation."""

        from training.trainer import get_compute_metrics

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.batch_decode.return_value = ["test text"]

        compute_metrics = get_compute_metrics(mock_tokenizer)

        # Verify it's callable
        assert callable(compute_metrics)


class TestCreateTrainer:
    """Tests for create_trainer function."""

    @patch("training.trainer.Seq2SeqTrainer")
    @patch("training.trainer.DataCollatorForSeq2Seq")
    @patch("training.trainer.get_compute_metrics")
    def test_create_trainer_default_args(
        self, mock_metrics, mock_collator, mock_trainer
    ):
        """Test creating trainer with default args."""
        from training.trainer import create_trainer

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()

        mock_trainer.return_value = MagicMock()

        create_trainer(
            mock_model, mock_tokenizer, mock_train_dataset, mock_eval_dataset
        )

        mock_trainer.assert_called_once()

    @patch("training.trainer.Seq2SeqTrainer")
    @patch("training.trainer.DataCollatorForSeq2Seq")
    @patch("training.trainer.get_compute_metrics")
    def test_create_trainer_custom_args(
        self, mock_metrics, mock_collator, mock_trainer
    ):
        """Test creating trainer with custom args."""
        from training.trainer import create_trainer, get_training_arguments

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()

        custom_args = get_training_arguments(num_epochs=10)
        mock_trainer.return_value = MagicMock()

        create_trainer(
            mock_model,
            mock_tokenizer,
            mock_train_dataset,
            mock_eval_dataset,
            training_args=custom_args,
        )

        mock_trainer.assert_called_once()
