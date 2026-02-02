import pytest
from src.config.settings import Config, CONFIG


def test_config_defaults():
    config = Config()
    assert config.base_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.max_new_tokens == 150
    assert config.temperature == 0.7


def test_config_adapter_path():
    config = Config()
    assert "customer-support-model" in str(config.adapter_path)


def test_config_singleton():
    assert CONFIG.base_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert CONFIG.repetition_penalty == 1.2
