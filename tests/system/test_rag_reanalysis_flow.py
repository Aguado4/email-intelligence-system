"""
System test: RAG reanalysis flow.

Tests the path where initial classification confidence is low
and the system fetches similar examples from the vector DB
for a second classification attempt.

Requirements:
  - Docker Compose must be running: docker compose up -d
  - Vector DB must have ground truth examples loaded
"""

import pytest


pytestmark = pytest.mark.system


class TestRAGReanalysisFlow:

    async def test_vector_db_has_ground_truth(self, vector_db_client):
        """The vector DB should have pre-loaded ground truth examples."""
        response = await vector_db_client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_examples"] >= 15  # 5 spam + 5 important + 5 neutral

    async def test_vector_db_search_returns_results(self, vector_db_client):
        """Searching for a spam-like email should find similar examples."""
        payload = {
            "subject": "You won a prize!",
            "body": "Click here to claim your free prize money today!",
            "k": 3,
        }
        response = await vector_db_client.post("/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] > 0

        # At least one result should be spam (ground truth has spam examples)
        categories = [r["category"] for r in data["results"]]
        assert "spam" in categories

    async def test_vector_db_search_with_filter(self, vector_db_client):
        """Category filter should narrow results."""
        payload = {
            "subject": "Meeting rescheduled",
            "body": "The board meeting has been moved to next Friday at 3pm.",
            "k": 3,
            "category_filter": "important",
        }
        response = await vector_db_client.post("/search", json=payload)

        assert response.status_code == 200
        data = response.json()

        # All results should be 'important' due to filter
        for result in data["results"]:
            assert result["category"] == "important"

    async def test_ambiguous_email_gets_classified(self, gateway_client, ambiguous_email):
        """
        An ambiguous email should still get a valid classification,
        possibly after RAG reanalysis.
        """
        response = await gateway_client.post("/api/v1/classify", json=ambiguous_email)

        assert response.status_code == 200

        data = response.json()
        classification = data["classification"]

        # Must be a valid category regardless of confidence
        assert classification["category"] in ("spam", "important", "neutral")
        assert 0.0 <= classification["confidence"] <= 1.0
        assert len(classification["reasoning"]) > 0

    async def test_store_and_retrieve_example(self, vector_db_client):
        """Store a new example and verify it can be found via search."""
        # Store
        store_payload = {
            "example": {
                "email_id": "system_test_new_001",
                "subject": "System test example",
                "body": "This is a unique system test email for verifying store and search.",
                "sender": "system-test@test.com",
                "category": "neutral",
                "confidence": 0.9,
                "metadata": {"source": "system_test"},
            }
        }
        store_response = await vector_db_client.post("/store", json=store_payload)
        assert store_response.status_code == 201

        # Search for it
        search_payload = {
            "subject": "System test example",
            "body": "This is a unique system test email for verifying store and search.",
            "k": 1,
        }
        search_response = await vector_db_client.post("/search", json=search_payload)
        assert search_response.status_code == 200

        data = search_response.json()
        assert data["count"] >= 1

        # Clean up
        delete_response = await vector_db_client.delete("/examples/system_test_new_001")
        assert delete_response.status_code == 200
