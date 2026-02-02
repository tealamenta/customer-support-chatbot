import pytest
from src.config.logging_config import setup_logging, LOG_DIR


def test_setup_logging():
    logger = setup_logging("test")
    assert logger.name == "test"
    assert logger.level == 20  # INFO


def test_log_dir_exists():
    assert LOG_DIR.exists()
