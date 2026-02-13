"""
Tests for Main Application and Lifespan
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from app.main import lifespan


@pytest.mark.asyncio
async def test_lifespan():
    # Mock the model manager
    with patch("app.main.model_manager") as mock_manager:
        # Create a mock app
        mock_app = MagicMock(spec=FastAPI)

        # Run lifespan
        async with lifespan(mock_app):
            # Verify startup
            mock_manager.load.assert_called_once()

        # Verify shutdown
        mock_manager.cleanup.assert_called_once()
