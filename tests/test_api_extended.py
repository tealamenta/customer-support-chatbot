import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


class TestChatEndpoint:
    @patch("src.api.app.bot")
    def test_chat_success(self, mock_bot):
        mock_bot.chat.return_value = "I can help you with that."
        
        from src.api.app import app
        client = TestClient(app)
        
        response = client.post("/chat", json={"question": "Cancel my order"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "Cancel my order"
        assert data["response"] == "I can help you with that."


class TestHealthEndpoint:
    @patch("src.api.app.bot")
    def test_health_model_loaded(self, mock_bot):
        mock_bot.return_value = MagicMock()
        
        from src.api.app import app
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "status" in response.json()
