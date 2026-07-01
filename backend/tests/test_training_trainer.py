"""
Tests for training.trainer module
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

# Mock heavy optional deps before importing trainer
sys.modules["evaluate"] = MagicMock()
sys.modules["nltk"] = MagicMock()


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

    @patch("training.trainer.nltk")
    @patch("training.trainer.evaluate")
    def test_compute_metrics_returns_all_metrics(self, mock_evaluate, mock_nltk):
        """Test the metrics function returns BLEU, chrF and METEOR."""
        from training.trainer import get_compute_metrics

        bleu_metric = MagicMock()
        bleu_metric.compute.return_value = {"bleu": 0.5}
        chrf_metric = MagicMock()
        chrf_metric.compute.return_value = {"score": 42.0}
        meteor_metric = MagicMock()
        meteor_metric.compute.return_value = {"meteor": 0.6}
        mock_evaluate.load.side_effect = [bleu_metric, chrf_metric, meteor_metric]

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__len__.return_value = 100
        mock_tokenizer.batch_decode.side_effect = [["hola"], ["hola"]]

        compute_metrics = get_compute_metrics(mock_tokenizer)

        # preds arrive as a tuple from predict_with_generate; labels use -100.
        preds = (np.array([[1, 2, 3]]),)
        labels = np.array([[1, 2, -100]])

        result = compute_metrics((preds, labels))

        assert result == {"bleu": 0.5, "chrf": 42.0, "meteor": 0.6}


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
