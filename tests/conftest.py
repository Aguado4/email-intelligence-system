"""
Shared fixtures for system/E2E tests.

These tests require Docker Compose to be running:
    docker compose up -d

Provides real HTTP clients that hit the actual services.
"""

import pytest
import httpx


# ============= Service URLs =============

API_GATEWAY_URL = "http://localhost:8000"
CLASSIFIER_URL = "http://localhost:8001"
VECTOR_DB_SERVICE_URL = "http://localhost:8003"


# ============= HTTP Clients =============

@pytest.fixture
async def gateway_client():
    """Real HTTP client for the API Gateway (port 8000)."""
    async with httpx.AsyncClient(
        base_url=API_GATEWAY_URL, timeout=60.0
    ) as client:
        yield client


@pytest.fixture
async def vector_db_client():
    """Real HTTP client for the Vector DB Service (port 8003)."""
    async with httpx.AsyncClient(
        base_url=VECTOR_DB_SERVICE_URL, timeout=30.0
    ) as client:
        yield client


# ============= Sample Emails =============

@pytest.fixture
def spam_email():
    """An obviously spam email for testing."""
    return {
        "email_id": "system_test_spam_001",
        "subject": "URGENT: You won $1,000,000!!!",
        "body": "Click here NOW to claim your prize! This is a limited time offer. Act fast!",
        "sender": "winner@totally-legit-lottery.com",
    }


@pytest.fixture
def important_email():
    """An important business email for testing."""
    return {
        "email_id": "system_test_important_001",
        "subject": "Board meeting rescheduled to tomorrow 2pm",
        "body": "Due to scheduling conflicts, the Q4 board review has been moved to tomorrow at 2 PM. Please confirm.",
        "sender": "ceo@company.com",
    }


@pytest.fixture
def ambiguous_email():
    """An ambiguous email that may trigger RAG reanalysis."""
    return {
        "email_id": "system_test_ambiguous_001",
        "subject": "Hi there",
        "body": "Just checking in to see how things are going with the project. Let me know if you need anything.",
        "sender": "colleague@company.com",
    }
