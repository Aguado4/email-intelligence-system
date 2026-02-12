"""
Tests for app/clients/classifier_client.py — HTTP client for classifier service.

Concepts introduced:
  - AsyncMock for mocking async HTTP calls
  - Patching httpx.AsyncClient
  - Testing error handling paths
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.clients.classifier_client import ClassifierClient


class TestClassifierClient:

    @pytest.fixture
    def client(self):
        """Create a ClassifierClient with a known base_url."""
        return ClassifierClient(base_url="http://classifier:8001")

    # ---------- classify_email ----------

    async def test_classify_email_success(self, client):
        """Successful classification should return parsed JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "category": "spam",
            "confidence": 0.95,
            "reasoning": "Scam",
            "keywords": ["prize"],
        }
        mock_response.raise_for_status = MagicMock()

        client.client = AsyncMock()
        client.client.post.return_value = mock_response

        result = await client.classify_email(
            email_id="test_001",
            subject="You won!",
            body="Click here",
            sender="scam@fake.com",
        )

        assert result["category"] == "spam"
        assert result["confidence"] == 0.95

        # Verify the correct URL was called
        client.client.post.assert_called_once_with(
            "http://classifier:8001/classify",
            json={
                "email_id": "test_001",
                "subject": "You won!",
                "body": "Click here",
                "sender": "scam@fake.com",
            },
        )

    async def test_classify_email_http_error(self, client):
        """HTTP errors should propagate as httpx.HTTPError."""
        client.client = AsyncMock()
        client.client.post.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.classify_email(
                email_id="test_err",
                subject="Test",
                body="Body",
                sender="test@test.com",
            )

    # ---------- health_check ----------

    async def test_health_check_success(self, client):
        """Healthy response should return the JSON payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = MagicMock()

        client.client = AsyncMock()
        client.client.get.return_value = mock_response

        result = await client.health_check()
        assert result == {"status": "healthy"}

    async def test_health_check_failure(self, client):
        """Failed health check should return unhealthy status."""
        client.client = AsyncMock()
        client.client.get.side_effect = httpx.ConnectError("Connection refused")

        result = await client.health_check()
        assert result["status"] == "unhealthy"
        assert "Connection refused" in result["error"]

    # ---------- close ----------

    async def test_close_calls_aclose(self, client):
        """close() should call aclose() on the underlying httpx client."""
        client.client = AsyncMock()

        await client.close()
        client.client.aclose.assert_called_once()

    # ---------- URL construction ----------

    def test_base_url_trailing_slash_stripped(self):
        """Trailing slash on base_url should be removed."""
        c = ClassifierClient(base_url="http://classifier:8001/")
        assert c.base_url == "http://classifier:8001"
