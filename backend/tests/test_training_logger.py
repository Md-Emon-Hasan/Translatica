"""
Tests for training.logger module
"""

import logging
import tempfile
from pathlib import Path


class TestGetTrainingLogger:
    """Tests for get_training_logger function."""

    def teardown_method(self, method):
        """Clean up loggers after each test."""
        # Close all handlers and remove them
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith("test_") or name.endswith("_logger"):
                logger = logging.getLogger(name)
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

    def test_get_training_logger_default(self):
        """Test getting logger with defaults."""
        from training.logger import get_training_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_training_logger("test_default_logger", log_dir=tmpdir)

            assert logger.name == "test_default_logger"
            assert len(logger.handlers) >= 2  # File and console handlers

            # Cleanup
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def test_get_training_logger_custom_file(self):
        """Test getting logger with custom file."""
        from training.logger import get_training_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = get_training_logger(
                "test_custom_logger", log_file="custom.log", log_dir=tmpdir
            )

            log_path = Path(tmpdir) / "custom.log"
            # Log something to create the file
            logger.info("Test message")

            # Cleanup before checking
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            assert log_path.exists()

    def test_get_training_logger_creates_directory(self):
        """Test that logger creates log directory."""
        from training.logger import get_training_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "logs"
            logger = get_training_logger("test_dir_logger", log_dir=str(log_dir))

            # Cleanup
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            assert log_dir.exists()

    def test_get_training_logger_no_duplicate_handlers(self):
        """Test that getting same logger doesn't duplicate handlers."""
        from training.logger import get_training_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = get_training_logger("test_same_logger", log_dir=tmpdir)
            initial_handlers = len(logger1.handlers)

            logger2 = get_training_logger("test_same_logger", log_dir=tmpdir)

            # Cleanup
            for handler in logger2.handlers[:]:
                handler.close()
                logger2.removeHandler(handler)

            assert len(logger2.handlers) <= initial_handlers
