"""
Integration test for the /classify FastAPI endpoint.

Concepts introduced:
  - httpx.ASGITransport for testing FastAPI apps without a real server
  - Patching at the workflow level to avoid LLM calls
  - Testing HTTP status codes and JSON response structure
"""

import pytest
import httpx
from httpx import ASGITransport
from unittest.mock import patch, AsyncMock


@pytest.fixture
async def classifier_app_client():
    """
    httpx.AsyncClient connected to the classifier FastAPI app.

    We patch classify_email (the workflow entry point) so no LLM is called.
    """
    mock_result = {
        "email_id": "test_001",
        "category": "spam",
        "confidence": 0.95,
        "reasoning": "Scam indicators detected",
        "keywords": ["prize", "click", "urgent"],
        "processing_stage": "classified",
    }

    with patch("app.main.classify_email", new_callable=AsyncMock, return_value=mock_result):
        from app.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


class TestClassifyEndpoint:

    async def test_classify_returns_200(self, classifier_app_client):
        """POST /classify with valid data should return 200."""
        payload = {
            "email_id": "test_001",
            "subject": "You won a prize!",
            "body": "Click here to claim your prize money now.",
            "sender": "scam@fake.com",
        }
        response = await classifier_app_client.post("/classify", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "spam"
        assert data["confidence"] == 0.95

    async def test_classify_missing_fields_returns_422(self, classifier_app_client):
        """POST /classify with missing required fields should return 422."""
        payload = {"email_id": "test_002"}  # missing subject, body, sender
        response = await classifier_app_client.post("/classify", json=payload)

        assert response.status_code == 422

    async def test_health_endpoint(self, classifier_app_client):
        """GET /health should return healthy status."""
        response = await classifier_app_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_root_endpoint(self, classifier_app_client):
        """GET / should return service info."""
        response = await classifier_app_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "classifier"
