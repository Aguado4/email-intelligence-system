"""
Shared fixtures for vector-db-service tests.

Provides mock ChromaDB clients and sample EmailExample objects.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.models.schemas import EmailExample


# ============= Sample Data =============

@pytest.fixture
def sample_spam_example():
    """A labeled spam email example."""
    return EmailExample(
        email_id="spam_test_001",
        subject="URGENT: Verify your account",
        body="Click here immediately to verify your account or it will be deleted.",
        sender="noreply@phishing.com",
        category="spam",
        confidence=1.0,
        metadata={"source": "test", "verified": True},
    )


@pytest.fixture
def sample_important_example():
    """A labeled important email example."""
    return EmailExample(
        email_id="important_test_001",
        subject="Board meeting rescheduled",
        body="The Q4 board review has been moved to tomorrow at 2 PM. Please confirm.",
        sender="ceo@company.com",
        category="important",
        confidence=1.0,
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_neutral_example():
    """A labeled neutral email example."""
    return EmailExample(
        email_id="neutral_test_001",
        subject="Weekly newsletter",
        body="Here is this week's roundup of team achievements and upcoming events.",
        sender="newsletter@company.com",
        category="neutral",
        confidence=1.0,
        metadata={"source": "test"},
    )


@pytest.fixture
def all_sample_examples(sample_spam_example, sample_important_example, sample_neutral_example):
    """All three sample examples in a list."""
    return [sample_spam_example, sample_important_example, sample_neutral_example]


# ============= Mock ChromaDB =============

@pytest.fixture
def mock_chroma_collection():
    """A mock ChromaDB collection with basic method stubs."""
    collection = MagicMock()
    collection.count.return_value = 0
    collection.add.return_value = None
    collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    return collection


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """A mock ChromaDB HttpClient that returns the mock collection."""
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_chroma_collection
    return client
