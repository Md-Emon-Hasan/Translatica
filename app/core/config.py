"""
Application Configuration using Pydantic Settings
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App settings
    APP_NAME: str = "Translatica"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_DIR: Path = BASE_DIR / "fine-tuned-model"
    TOKENIZER_PATH: Path = MODEL_DIR / "fine-tuned-tokenizer"
    MODEL_PATH: Path = MODEL_DIR / "fine-tuned-model"
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Model settings
    MAX_INPUT_LENGTH: int = 512
    MAX_OUTPUT_LENGTH: int = 256
    NUM_BEAMS: int = 8
    DEVICE: Literal["cuda", "cpu", "auto"] = "auto"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
