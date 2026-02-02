import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token = "</s>"
    tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    tokenizer.decode.return_value = "<|assistant|>This is a test response."
    return tokenizer


@pytest.fixture
def sample_test_data():
    return [
        {"instruction": "I want to cancel my order", "response": "I can help you cancel your order.", "intent": "cancel_order"},
        {"instruction": "Where is my package?", "response": "Let me check your package status.", "intent": "track_order"},
        {"instruction": "How do I get a refund?", "response": "I'll help you with the refund process.", "intent": "get_refund"},
    ]
