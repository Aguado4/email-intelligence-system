"""
Integration test for POST /store endpoint.

Concepts introduced:
  - Testing 201 Created responses
  - Verifying side effects on mock objects
  - Testing error propagation
"""

import pytest


class TestStoreEndpoint:

    async def test_store_example_returns_201(self, vdb_client):
        """POST /store with valid data should return 201 Created."""
        payload = {
            "example": {
                "email_id": "new_spam_001",
                "subject": "Free money!",
                "body": "Click here to get free money right now! Limited offer!",
                "sender": "scam@free-money.com",
                "category": "spam",
                "confidence": 1.0,
                "metadata": {"source": "test"},
            }
        }
        response = await vdb_client.post("/store", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"

    async def test_store_calls_vector_store(self, vdb_client, mock_vector_store):
        """The endpoint should call vector_store.store_example."""
        payload = {
            "example": {
                "email_id": "new_important_001",
                "subject": "Board meeting",
                "body": "The quarterly board meeting has been moved to next week.",
                "sender": "ceo@company.com",
                "category": "important",
                "confidence": 0.95,
            }
        }
        await vdb_client.post("/store", json=payload)

        mock_vector_store.store_example.assert_called_once()

    async def test_store_invalid_category_returns_422(self, vdb_client):
        """Invalid category should fail Pydantic validation -> 422."""
        payload = {
            "example": {
                "email_id": "bad_001",
                "subject": "Test",
                "body": "A body that is long enough to pass the validation.",
                "sender": "test@test.com",
                "category": "phishing",  # not valid
                "confidence": 0.5,
            }
        }
        response = await vdb_client.post("/store", json=payload)

        assert response.status_code == 422

    async def test_store_missing_example_returns_422(self, vdb_client):
        """Missing the 'example' wrapper should fail validation."""
        payload = {
            "email_id": "no_wrapper",
            "subject": "Test",
            "body": "Body content here for the email.",
            "sender": "test@test.com",
            "category": "spam",
            "confidence": 0.5,
        }
        response = await vdb_client.post("/store", json=payload)

        assert response.status_code == 422

    async def test_store_error_returns_500(self, vdb_client, mock_vector_store):
        """If the vector store raises, the endpoint should return 500."""
        mock_vector_store.store_example.side_effect = RuntimeError("DB error")

        payload = {
            "example": {
                "email_id": "err_001",
                "subject": "Test error",
                "body": "This should trigger a 500 error from the store endpoint.",
                "sender": "test@test.com",
                "category": "neutral",
                "confidence": 0.5,
            }
        }
        response = await vdb_client.post("/store", json=payload)

        assert response.status_code == 500

    async def test_stats_endpoint(self, vdb_client):
        """GET /stats should return database statistics."""
        response = await vdb_client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_examples"] == 15
        assert "examples_by_category" in data

    async def test_health_endpoint(self, vdb_client):
        """GET /health should return healthy when store is connected."""
        response = await vdb_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
