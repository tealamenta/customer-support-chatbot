import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def test_import_app():
    from src.api.app import app
    assert app.title == "Customer Support Chatbot API"


def test_root_endpoint():
    with patch("src.api.app.bot", None):
        from src.api.app import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_health_endpoint():
    with patch("src.api.app.bot", None):
        from src.api.app import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
