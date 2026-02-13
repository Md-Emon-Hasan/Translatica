"""
Tests for training.train module (main script)
"""

import sys
from unittest.mock import MagicMock, patch

# Mock evaluate module
sys.modules["evaluate"] = MagicMock()


class TestParseArgs:
    """Tests for parse_args function."""

    @patch("sys.argv", ["train.py"])
    def test_parse_args_defaults(self):
        """Test parsing args with defaults."""
        from training.train import parse_args

        args = parse_args()

        assert args.model_checkpoint == "Helsinki-NLP/opus-mt-en-es"
        assert args.output_dir == "./fine-tuned-model"
        assert args.num_epochs == 3
        assert args.batch_size == 16
        assert args.learning_rate == 2e-5

    @patch(
        "sys.argv",
        [
            "train.py",
            "--model-checkpoint",
            "custom-model",
            "--output-dir",
            "./custom-output",
            "--num-epochs",
            "5",
            "--batch-size",
            "32",
            "--learning-rate",
            "1e-4",
        ],
    )
    def test_parse_args_custom(self):
        """Test parsing custom args."""
        from training.train import parse_args

        args = parse_args()

        assert args.model_checkpoint == "custom-model"
        assert args.output_dir == "./custom-output"
        assert args.num_epochs == 5
        assert args.batch_size == 32
        assert args.learning_rate == 1e-4


class TestMainFunction:
    """Tests for main training function."""

    @patch("training.train.Path")
    @patch("training.train.create_trainer")
    @patch("training.train.get_training_arguments")
    @patch("training.train.print_trainable_parameters")
    @patch("training.train.create_peft_model")
    @patch("training.train.prepare_dataset")
    @patch("training.train.get_training_logger")
    @patch("sys.argv", ["train.py", "--num-epochs", "1"])
    def test_main_full_pipeline(
        self,
        mock_logger,
        mock_prepare,
        mock_create_peft,
        mock_print_params,
        mock_training_args,
        mock_create_trainer,
        mock_path,
    ):
        """Test full training pipeline."""
        from training.train import main

        # Setup mocks
        mock_logger.return_value = MagicMock()

        # Mock dataset
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=100)
        mock_test_ds.__len__ = MagicMock(return_value=20)
        mock_prepare.return_value = (
            {"train": mock_train_ds, "test": mock_test_ds},
            MagicMock(),  # tokenizer
        )

        # Mock PEFT model
        mock_peft_model = MagicMock()
        mock_create_peft.return_value = mock_peft_model

        # Mock training args
        mock_training_args.return_value = MagicMock()

        # Mock trainer
        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer

        # Mock Path
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        # Run main
        main()

        # Verify pipeline was executed
        mock_prepare.assert_called_once()
        mock_create_peft.assert_called_once()
        mock_create_trainer.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_peft_model.save_pretrained.assert_called_once()
