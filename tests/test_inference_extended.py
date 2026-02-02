import pytest
from unittest.mock import MagicMock, patch
from src.model.inference import CustomerSupportBot, load_model


class TestCustomerSupportBotLoad:
    @patch("src.model.inference.PeftModel")
    @patch("src.model.inference.AutoTokenizer")
    @patch("src.model.inference.AutoModelForCausalLM")
    def test_load_model_success(self, mock_auto_model, mock_tokenizer, mock_peft):
        mock_base = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_base
        mock_peft.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        bot = CustomerSupportBot(adapter_path="test/path")
        result = bot.load()
        
        assert result == bot
        assert bot.model is not None
        assert bot.tokenizer is not None


class TestLoadModelFunction:
    @patch("src.model.inference.CustomerSupportBot")
    def test_load_model_helper(self, mock_bot_class):
        mock_instance = MagicMock()
        mock_instance.load.return_value = mock_instance
        mock_bot_class.return_value = mock_instance
        
        result = load_model("test/path")
        
        mock_bot_class.assert_called_once_with("test/path")
        mock_instance.load.assert_called_once()
        assert result == mock_instance


class TestChatGeneration:
    def test_chat_with_no_assistant_tag(self):
        bot = CustomerSupportBot()
        
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        
        bot.tokenizer = MagicMock()
        bot.tokenizer.return_value = mock_inputs
        bot.tokenizer.eos_token_id = 2
        
        bot.model = MagicMock()
        bot.model.device = "cpu"
        bot.model.generate.return_value = [[1, 2, 3]]
        
        # Response without <|assistant|> tag
        bot.tokenizer.decode.return_value = "Direct response without tags"
        bot.device = "cpu"
        
        response = bot.chat("Test")
        assert response == "Direct response without tags"
