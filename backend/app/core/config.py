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
    DATA_DIR: Path = BASE_DIR / "data"
    DATABASE_PATH: Path = DATA_DIR / "translations.db"
    DATABASE_URL: str = f"sqlite+aiosqlite:///{DATABASE_PATH}"

    # Model settings
    MAX_INPUT_LENGTH: int = 512
    MAX_OUTPUT_LENGTH: int = 256
    NUM_BEAMS: int = 8
    DEVICE: Literal["cuda", "cpu", "auto"] = "auto"

    # Generation settings
    GEN_NUM_BEAMS: int = 4
    GEN_MAX_NEW_TOKENS: int = 256
    GEN_LENGTH_PENALTY: float = 1.0
    GEN_EARLY_STOPPING: bool = True
    GEN_NO_REPEAT_NGRAM_SIZE: int = 3

    # Chunking settings
    MAX_INPUT_TOKENS: int = 384
    CHUNK_TOKEN_BUDGET: int = 256
    CHARS_PER_TOKEN_ESTIMATE: int = 4

    # Validation settings
    MIN_INPUT_CHARS: int = 1
    MAX_INPUT_CHARS: int = 10000
    LANG_CHECK_ENABLED: bool = True

    # Cache settings
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAXSIZE: int = 500

    # Rate limit settings
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_TRANSLATE: str = "20/minute"
    RATE_LIMIT_BATCH: str = "5/minute"
    RATE_LIMIT_HISTORY: str = "60/minute"
    RATE_LIMIT_STATS: str = "60/minute"

    # Batch settings
    MAX_BATCH_SIZE: int = 20
    MAX_BATCH_CHARS: int = 50000

    # History settings
    HISTORY_DEFAULT_LIMIT: int = 20
    HISTORY_MAX_LIMIT: int = 100
    HISTORY_TEXT_PREVIEW_CHARS: int = 200

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
