"""
Pytest Configuration and Fixtures
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    tokenizer.decode.return_value = "Hola mundo"
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock translation model."""
    model = MagicMock()
    model.generate.return_value = [[1, 2, 3]]
    model.eval.return_value = None
    model.to.return_value = model
    return model


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager for service tests."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.translate.return_value = "Hola mundo"
    mock.load.return_value = None
    mock.cleanup.return_value = None
    return mock


@pytest.fixture
async def client():
    """Create a test client with mocked model."""
    # Create mock model manager
    mock_manager = MagicMock()
    mock_manager.is_loaded = True
    mock_manager.translate.return_value = "Hola mundo"
    mock_manager.load.return_value = None
    mock_manager.cleanup.return_value = None

    # Patch the model manager in all modules before importing app
    with (
        patch("app.core.model.model_manager", mock_manager),
        patch("app.main.model_manager", mock_manager),
        patch("app.api.routes.TranslationService") as MockService,
    ):

        # Configure the mock service
        MockService.translate.return_value = "Hola mundo"

        from httpx import ASGITransport, AsyncClient

        from app.main import app

        # Also patch the health check route's import
        with patch.object(app, "_mock_manager", mock_manager, create=True):
            # Override get_db dependency
            async def override_get_db():
                mock_session = MagicMock()
                mock_session.add = MagicMock()
                mock_session.commit = AsyncMock()
                mock_session.refresh = AsyncMock()
                yield mock_session

            app.dependency_overrides[get_db] = override_get_db

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                client._mock_manager = mock_manager
                client._mock_service = MockService
                yield client

            # Clean up overrides
            app.dependency_overrides.clear()


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return {
        "english": "Hello, how are you?",
        "spanish": "Hola, ¿cómo estás?",
        "long_text": "This is a longer text that should be translated. " * 10,
        "empty": "",
        "whitespace": "   ",
    }
