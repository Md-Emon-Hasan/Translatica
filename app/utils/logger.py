"""
Logging Utility
"""

import logging
import os
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically module name)
        log_file: Log file name (default: app.log)

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / log_file

    # Create logger
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Stream handler (console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
