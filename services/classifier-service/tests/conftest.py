"""
Shared fixtures for classifier-service tests.

Fixtures defined here are automatically available to ALL tests
in this directory and subdirectories — no imports needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from clients.vector_db_client import SimilarExample


# ============= Sample States =============

@pytest.fixture
def sample_email_state():
    """A minimal EmailClassificationState for testing."""
    return {
        "email_id": "test_001",
        "subject": "URGENT: You won $1,000,000!!!",
        "body": "Click here NOW to claim your prize! Limited time offer!",
        "sender": "scam@fake-lottery.com",
        "category": "",
        "confidence": 0.0,
        "reasoning": "",
        "keywords": [],
        "retry_count": 0,
        "processing_stage": "initialized",
    }


@pytest.fixture
def classified_low_confidence_state(sample_email_state):
    """State after a low-confidence classification (triggers reanalysis)."""
    return {
        **sample_email_state,
        "category": "neutral",
        "confidence": 0.55,
        "reasoning": "Unclear intent",
        "keywords": ["urgent"],
        "processing_stage": "classified",
    }


@pytest.fixture
def classified_high_confidence_state(sample_email_state):
    """State after a high-confidence classification (no reanalysis)."""
    return {
        **sample_email_state,
        "category": "spam",
        "confidence": 0.95,
        "reasoning": "Classic lottery scam indicators",
        "keywords": ["urgent", "won", "prize", "click"],
        "processing_stage": "classified",
    }


# ============= Mock LLM =============

@pytest.fixture
def mock_llm():
    """
    A mock LangChain LLM that returns a configurable JSON response.

    Usage in tests:
        result = mock_llm  # already an AsyncMock
        mock_llm.ainvoke.return_value.content = '{"category": "spam", ...}'
    """
    llm = AsyncMock()
    llm.ainvoke.return_value = MagicMock(
        content='{"category": "spam", "confidence": 0.95, '
                '"reasoning": "Obvious spam", "keywords": ["scam"]}'
    )
    return llm


# ============= Mock Vector DB =============

@pytest.fixture
def sample_similar_examples():
    """Sample SimilarExample objects for RAG tests."""
    return [
        SimilarExample(
            email_id="spam_001",
            subject="You won a prize!",
            body="Click here to claim your prize money now.",
            sender="winner@lottery.com",
            category="spam",
            confidence=1.0,
            similarity_score=0.92,
        ),
        SimilarExample(
            email_id="spam_002",
            subject="Congratulations! Free gift",
            body="Claim your free gift by clicking below.",
            sender="gifts@promo.com",
            category="spam",
            confidence=1.0,
            similarity_score=0.85,
        ),
    ]


@pytest.fixture
def mock_vector_db_client(sample_similar_examples):
    """Mock VectorDBClient that returns sample similar examples."""
    client = AsyncMock()
    client.search_similar.return_value = sample_similar_examples
    client.health_check.return_value = True
    return client
