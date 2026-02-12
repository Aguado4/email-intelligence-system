"""
Integration test conftest for vector-db-service.

Mocks the heavy dependencies (chromadb, sentence-transformers) so that
integration tests can run without these installed locally.
The tests verify FastAPI routing and request/response handling,
not the actual vector store or embedding logic.
"""

import sys
from unittest.mock import MagicMock

import pytest
import httpx
from httpx import ASGITransport


# ============= Mock heavy dependencies before app.main is imported =============

def _ensure_mock_modules():
    """
    Insert mock modules for chromadb and sentence_transformers into sys.modules
    so that `app.storage.*` can be imported without the real packages.
    """
    for mod_name in [
        "chromadb",
        "chromadb.config",
        "sentence_transformers",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_ensure_mock_modules()

from app.models.schemas import SimilarExample


# ============= Fixtures =============

@pytest.fixture
def mock_vector_store():
    """A mock VectorStore with sensible defaults."""
    store = MagicMock()

    # Return real Pydantic SimilarExample instances so FastAPI serialization works
    store.search_similar.return_value = [
        SimilarExample(
            email_id="spam_001",
            subject="You won a prize!",
            body="Click here to claim now.",
            sender="scam@fake.com",
            category="spam",
            confidence=1.0,
            similarity_score=0.92,
            metadata={"source": "test"},
        )
    ]
    store.store_example.return_value = "new_example_001"
    store.get_stats.return_value = {
        "total_examples": 15,
        "examples_by_category": {"spam": 5, "important": 5, "neutral": 5},
        "collection_name": "email_examples",
    }
    store.delete_example.return_value = True
    return store


@pytest.fixture
async def vdb_client(mock_vector_store):
    """
    httpx.AsyncClient connected to the vector-db-service FastAPI app.

    Replaces the global `vector_store` in app.main so no real
    ChromaDB or embedding model is needed.
    """
    import app.main as main_module

    original_store = main_module.vector_store
    main_module.vector_store = mock_vector_store

    transport = ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    main_module.vector_store = original_store
