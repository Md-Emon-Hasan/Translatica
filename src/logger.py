# src/logger.py
import logging
import os

def get_logger(name: str, log_file: str = "training.log"):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)