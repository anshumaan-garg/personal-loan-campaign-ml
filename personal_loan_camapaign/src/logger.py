import os
import logging
from logging.handlers import RotatingFileHandler
from src.config import BASE_DIR

def setup_app_logger(module_name: str) -> logging.Logger:
    """Configures a standardized rolling file and console logger instance."""
    # Ensure a local logs directory exists at the root level
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)  # Capture everything from DEBUG up
    
    # Prevent duplicate log messages if logger is re-initialized
    if logger.hasHandlers():
        return logger

    # Clean, informative string format mapping timestamps, levels, and source lines
    log_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (Standard Output streaming)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Keep console logs clean
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # 2. Rolling File Handler (Persistent disk logging up to 5MB per file)
    log_file_path = os.path.join(log_dir, "campaign_pipeline.log")
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # Log comprehensive details to disk
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger