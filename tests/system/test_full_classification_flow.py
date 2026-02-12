"""
System test: full email classification flow through all services.

Requirements:
  - Docker Compose must be running: docker compose up -d
  - All services must be healthy

Concepts introduced:
  - @pytest.mark.system marker
  - Real HTTP calls to live services
  - End-to-end response validation
"""

import pytest


pytestmark = pytest.mark.system


class TestFullClassificationFlow:

    async def test_gateway_is_running(self, gateway_client):
        """Verify the API Gateway is reachable."""
        response = await gateway_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    async def test_classify_spam_email(self, gateway_client, spam_email):
        """
        End-to-end: send a spam email through the gateway
        and verify it's classified as spam.
        """
        response = await gateway_client.post("/api/v1/classify", json=spam_email)

        assert response.status_code == 200

        data = response.json()
        assert data["email_id"] == spam_email["email_id"]
        assert "classification" in data

        classification = data["classification"]
        assert classification["category"] == "spam"
        assert classification["confidence"] >= 0.7
        assert len(classification["reasoning"]) > 0
        assert isinstance(classification["keywords"], list)
        assert data["processing_time_ms"] > 0

    async def test_classify_important_email(self, gateway_client, important_email):
        """
        End-to-end: an important business email should be classified
        as 'important' with reasonable confidence.
        """
        response = await gateway_client.post("/api/v1/classify", json=important_email)

        assert response.status_code == 200

        data = response.json()
        classification = data["classification"]
        assert classification["category"] == "important"
        assert classification["confidence"] >= 0.5

    async def test_classify_returns_all_fields(self, gateway_client, spam_email):
        """Verify the response has the complete EmailProcessingResponse shape."""
        response = await gateway_client.post("/api/v1/classify", json=spam_email)
        data = response.json()

        # Top-level fields
        assert "email_id" in data
        assert "classification" in data
        assert "processing_time_ms" in data
        assert "timestamp" in data
        assert "service_version" in data

        # Classification fields
        classification = data["classification"]
        assert "category" in classification
        assert "confidence" in classification
        assert "reasoning" in classification
        assert "keywords" in classification

    async def test_health_check_shows_all_services(self, gateway_client):
        """GET /api/v1/health should report on both gateway and classifier."""
        response = await gateway_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["api_gateway"] == "healthy"
        assert data["classifier_service"] == "healthy"

    async def test_vector_db_service_healthy(self, vector_db_client):
        """Verify the Vector DB Service is running and has examples loaded."""
        response = await vector_db_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["total_examples"] > 0

    async def test_invalid_input_rejected(self, gateway_client):
        """Sending invalid input should return 422, not 500."""
        bad_payload = {"email_id": "bad"}
        response = await gateway_client.post("/api/v1/classify", json=bad_payload)

        assert response.status_code == 422
