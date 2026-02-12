"""
Integration test for POST /search endpoint.

Concepts introduced:
  - Patching a global variable (vector_store) in the app module
  - Testing search with and without category filters
  - Mocking ChromaDB query results
"""

import pytest


class TestSearchEndpoint:

    async def test_search_returns_results(self, vdb_client):
        """POST /search should return matching examples."""
        payload = {
            "subject": "You won!",
            "body": "Click here to claim your prize money now.",
        }
        response = await vdb_client.post("/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["category"] == "spam"

    async def test_search_with_category_filter(self, vdb_client, mock_vector_store):
        """category_filter should be passed to the store."""
        payload = {
            "subject": "Meeting tomorrow",
            "body": "Please confirm your attendance at the board meeting.",
            "category_filter": "important",
        }
        response = await vdb_client.post("/search", json=payload)

        assert response.status_code == 200
        mock_vector_store.search_similar.assert_called_once()
        call_kwargs = mock_vector_store.search_similar.call_args
        assert call_kwargs.kwargs.get("category_filter") == "important"

    async def test_search_with_custom_k(self, vdb_client, mock_vector_store):
        """k parameter should control number of results requested."""
        payload = {
            "subject": "Test",
            "body": "A body that is long enough for the search request.",
            "k": 5,
        }
        response = await vdb_client.post("/search", json=payload)

        assert response.status_code == 200

    async def test_search_missing_body_returns_422(self, vdb_client):
        """Missing required 'body' field should return 422."""
        payload = {"subject": "Only subject"}
        response = await vdb_client.post("/search", json=payload)

        assert response.status_code == 422

    async def test_search_empty_results(self, vdb_client, mock_vector_store):
        """When no similar examples found, return empty results."""
        mock_vector_store.search_similar.return_value = []

        payload = {
            "subject": "Unique email",
            "body": "This email has no similar matches in the database.",
        }
        response = await vdb_client.post("/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["results"] == []
