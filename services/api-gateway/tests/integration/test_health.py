"""
Integration tests for health and root endpoints.

Concepts introduced:
  - Testing multiple endpoints on the same app
  - Verifying JSON response shapes
"""

import pytest
import httpx
from httpx import ASGITransport


@pytest.fixture
async def api_client(mock_classifier_client):
    """httpx.AsyncClient wired to the api-gateway app."""
    from app.main import app
    from app.routes import classify

    original_client = classify.classifier_client
    classify.classifier_client = mock_classifier_client

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    classify.classifier_client = original_client


class TestRootEndpoint:

    async def test_root_returns_service_info(self, api_client):
        """GET / should return service metadata."""
        response = await api_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "api-gateway"
        assert "version" in data
        assert data["status"] == "running"

    async def test_root_includes_docs_link(self, api_client):
        """Root response should point to /docs."""
        response = await api_client.get("/")
        data = response.json()
        assert data["docs"] == "/docs"


class TestHealthEndpoint:

    async def test_simple_health_check(self, api_client):
        """GET /health should return healthy."""
        response = await api_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    async def test_api_v1_health_check(self, api_client):
        """GET /api/v1/health should include both gateway and classifier status."""
        response = await api_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["api_gateway"] == "healthy"
        assert data["classifier_service"] == "healthy"
        assert "timestamp" in data
