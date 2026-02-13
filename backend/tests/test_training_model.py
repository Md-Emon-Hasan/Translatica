"""
Tests for training.model module
"""

from unittest.mock import MagicMock, patch


class TestLoadBaseModel:
    """Tests for load_base_model function."""

    @patch("training.model.AutoModelForSeq2SeqLM.from_pretrained")
    def test_load_base_model(self, mock_model):
        """Test loading base model."""
        from training.model import load_base_model

        mock_model.return_value = MagicMock()
        load_base_model("test-checkpoint")

        mock_model.assert_called_once_with("test-checkpoint")


class TestGetLoraConfig:
    """Tests for get_lora_config function."""

    def test_get_lora_config_default(self):
        """Test LoRA config with defaults."""
        from training.model import get_lora_config

        config = get_lora_config()

        assert config.r == 8
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        # target_modules can be list or set, check contents
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_get_lora_config_custom(self):
        """Test LoRA config with custom values."""
        from training.model import get_lora_config

        config = get_lora_config(
            r=16, lora_alpha=64, lora_dropout=0.2, target_modules=["k_proj"]
        )

        assert config.r == 16
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.2
        assert "k_proj" in config.target_modules


class TestCreatePeftModel:
    """Tests for create_peft_model function."""

    @patch("training.model.get_peft_model")
    @patch("training.model.load_base_model")
    def test_create_peft_model_default_config(self, mock_load, mock_peft):
        """Test creating PEFT model with default config."""
        from training.model import create_peft_model

        mock_load.return_value = MagicMock()
        mock_peft.return_value = MagicMock()

        create_peft_model("test-checkpoint")

        mock_load.assert_called_once_with("test-checkpoint")
        mock_peft.assert_called_once()

    @patch("training.model.get_peft_model")
    @patch("training.model.load_base_model")
    def test_create_peft_model_custom_config(self, mock_load, mock_peft):
        """Test creating PEFT model with custom config."""
        from training.model import create_peft_model, get_lora_config

        mock_load.return_value = MagicMock()
        mock_peft.return_value = MagicMock()

        custom_config = get_lora_config(r=16)
        create_peft_model("test-checkpoint", lora_config=custom_config)

        mock_peft.assert_called_once()


class TestPrintTrainableParameters:
    """Tests for print_trainable_parameters function."""

    def test_print_trainable_parameters(self, capsys):
        """Test printing trainable parameters."""
        from training.model import print_trainable_parameters

        # Create mock model with parameters
        mock_model = MagicMock()
        mock_param1 = MagicMock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True

        mock_param2 = MagicMock()
        mock_param2.numel.return_value = 500
        mock_param2.requires_grad = False

        mock_model.named_parameters.return_value = [
            ("layer1", mock_param1),
            ("layer2", mock_param2),
        ]

        print_trainable_parameters(mock_model)

        captured = capsys.readouterr()
        assert "Trainable params" in captured.out
        assert "1,000" in captured.out
