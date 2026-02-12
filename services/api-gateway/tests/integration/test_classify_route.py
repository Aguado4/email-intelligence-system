"""
Integration test for the /api/v1/classify route.

Concepts introduced:
  - dependency_overrides pattern for FastAPI
  - httpx.ASGITransport for testing full request/response cycle
  - Testing response structure and status codes
"""

import pytest
import httpx
from httpx import ASGITransport
from unittest.mock import AsyncMock

from app.clients.classifier_client import ClassifierClient


@pytest.fixture
async def api_client(mock_classifier_client):
    """
    httpx.AsyncClient wired to the api-gateway FastAPI app.

    Injects the mock classifier client via the routes module
    (the same pattern used in production via lifespan).
    """
    from app.main import app
    from app.routes import classify

    original_client = classify.classifier_client
    classify.classifier_client = mock_classifier_client

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    classify.classifier_client = original_client


class TestClassifyRoute:

    async def test_classify_email_success(self, api_client):
        """POST /api/v1/classify should return classification results."""
        payload = {
            "email_id": "test_001",
            "subject": "You won a prize!",
            "body": "Click here to claim your prize money now.",
            "sender": "scam@fake.com",
        }
        response = await api_client.post("/api/v1/classify", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert data["email_id"] == "test_001"
        assert "classification" in data
        assert data["classification"]["category"] == "spam"
        assert data["classification"]["confidence"] == 0.95
        assert "processing_time_ms" in data

    async def test_classify_invalid_email_returns_422(self, api_client):
        """Missing required fields should return 422 Unprocessable Entity."""
        payload = {"email_id": "test_bad"}  # missing subject, body, sender
        response = await api_client.post("/api/v1/classify", json=payload)

        assert response.status_code == 422

    async def test_classify_short_body_returns_422(self, api_client):
        """Body that's too short (< 10 chars) should fail validation."""
        payload = {
            "email_id": "test_short",
            "subject": "Test",
            "body": "Short",  # min_length=10 in shared schema
            "sender": "test@test.com",
        }
        response = await api_client.post("/api/v1/classify", json=payload)

        assert response.status_code == 422

    async def test_classify_service_error_returns_500(self, api_client, mock_classifier_client):
        """If the classifier service raises, the gateway should return 500."""
        mock_classifier_client.classify_email.side_effect = Exception("Service down")

        payload = {
            "email_id": "test_err",
            "subject": "Test error",
            "body": "This should trigger a 500 error from the gateway.",
            "sender": "test@test.com",
        }
        response = await api_client.post("/api/v1/classify", json=payload)

        assert response.status_code == 500
