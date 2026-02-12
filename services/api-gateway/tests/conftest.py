"""
Shared fixtures for api-gateway tests.

Provides mock classifier clients and async HTTP test clients.
"""

import sys
from pathlib import Path

import pytest
from unittest.mock import AsyncMock
import httpx
from httpx import ASGITransport

# Make shared models importable outside Docker
# In Docker the path is /app/shared; locally we resolve from the repo root.
_repo_root = Path(__file__).resolve().parents[3]  # email-intelligence-system/
_shared_path = _repo_root / "shared"
if str(_shared_path) not in sys.path:
    sys.path.insert(0, str(_shared_path))

from app.clients.classifier_client import ClassifierClient


# ============= Mock Classifier Client =============

@pytest.fixture
def mock_classifier_response():
    """Standard successful classification response from the classifier service."""
    return {
        "category": "spam",
        "confidence": 0.95,
        "reasoning": "Classic lottery scam with urgency markers",
        "keywords": ["urgent", "won", "prize", "click"],
    }


@pytest.fixture
def mock_classifier_client(mock_classifier_response):
    """
    A mock ClassifierClient where classify_email and health_check are AsyncMocks.

    This avoids real HTTP calls to the classifier service.
    """
    client = AsyncMock(spec=ClassifierClient)
    client.classify_email.return_value = mock_classifier_response
    client.health_check.return_value = {"status": "healthy"}
    client.close.return_value = None
    return client


# ============= Async HTTP Test Client =============

@pytest.fixture
async def async_client(mock_classifier_client):
    """
    httpx.AsyncClient wired to the FastAPI app via ASGITransport.

    This lets us test real HTTP requests without starting a server.
    The classifier client is injected via the routes module.
    """
    from app.main import app
    from app.routes import classify

    # Inject the mock client into the routes module
    original_client = classify.classifier_client
    classify.classifier_client = mock_classifier_client

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Restore original (cleanup)
    classify.classifier_client = original_client
