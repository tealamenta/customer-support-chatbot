import pytest
from unittest.mock import MagicMock, patch
from src.model.inference import CustomerSupportBot


def test_customer_support_bot_init():
    bot = CustomerSupportBot()
    assert bot.model is None
    assert bot.tokenizer is None


def test_customer_support_bot_custom_path():
    bot = CustomerSupportBot(adapter_path="/custom/path")
    assert bot.adapter_path == "/custom/path"


def test_chat_prompt_format():
    bot = CustomerSupportBot()
    
    # Mock tokenizer with proper .to() method
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    
    bot.tokenizer = MagicMock()
    bot.tokenizer.return_value = mock_inputs
    bot.tokenizer.eos_token_id = 2
    
    bot.model = MagicMock()
    bot.model.device = "cpu"
    bot.model.generate.return_value = [[1, 2, 3]]
    
    bot.tokenizer.decode.return_value = "<|assistant|>Test response"
    bot.device = "cpu"
    
    response = bot.chat("Hello")
    
    call_args = bot.tokenizer.call_args[0][0]
    assert "<|system|>" in call_args
    assert "<|user|>" in call_args
    assert "<|assistant|>" in call_args


def test_chat_response_extraction():
    bot = CustomerSupportBot()
    
    # Mock tokenizer with proper .to() method
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    
    bot.tokenizer = MagicMock()
    bot.tokenizer.return_value = mock_inputs
    bot.tokenizer.eos_token_id = 2
    
    bot.model = MagicMock()
    bot.model.device = "cpu"
    bot.model.generate.return_value = [[1, 2, 3]]
    
    bot.tokenizer.decode.return_value = "Some prefix<|assistant|>Clean response here"
    bot.device = "cpu"
    
    response = bot.chat("Test")
    assert response == "Clean response here"


def test_chat_removes_trailing_brackets():
    bot = CustomerSupportBot()
    
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    
    bot.tokenizer = MagicMock()
    bot.tokenizer.return_value = mock_inputs
    bot.tokenizer.eos_token_id = 2
    
    bot.model = MagicMock()
    bot.model.device = "cpu"
    bot.model.generate.return_value = [[1, 2, 3]]
    
    bot.tokenizer.decode.return_value = "<|assistant|>Response here<|user|>extra"
    bot.device = "cpu"
    
    response = bot.chat("Test")
    assert response == "Response here"
    assert "<" not in response
