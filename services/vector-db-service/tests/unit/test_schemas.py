"""
Tests for app/models/schemas.py — Pydantic model validation.

Concepts introduced:
  - pytest.raises with Pydantic ValidationError
  - Testing model constraints (min_length, ge/le, Literal)
  - Verifying default values
"""

import pytest
from pydantic import ValidationError
from datetime import datetime

from app.models.schemas import (
    EmailExample,
    SearchRequest,
    SimilarExample,
    StoreExampleRequest,
)


class TestEmailExample:

    def test_valid_example(self, sample_spam_example):
        """A well-formed EmailExample should be accepted."""
        assert sample_spam_example.email_id == "spam_test_001"
        assert sample_spam_example.category == "spam"
        assert sample_spam_example.confidence == 1.0

    def test_invalid_category_rejected(self):
        """Category must be one of: spam, important, neutral."""
        with pytest.raises(ValidationError):
            EmailExample(
                email_id="bad_001",
                subject="Test",
                body="A body that is long enough to pass validation.",
                sender="test@example.com",
                category="phishing",  # not a valid Literal
                confidence=0.8,
            )

    def test_confidence_below_zero_rejected(self):
        """Confidence must be >= 0.0."""
        with pytest.raises(ValidationError):
            EmailExample(
                email_id="bad_002",
                subject="Test",
                body="A body that is long enough to pass validation.",
                sender="test@example.com",
                category="spam",
                confidence=-0.1,
            )

    def test_confidence_above_one_rejected(self):
        """Confidence must be <= 1.0."""
        with pytest.raises(ValidationError):
            EmailExample(
                email_id="bad_003",
                subject="Test",
                body="A body that is long enough to pass validation.",
                sender="test@example.com",
                category="spam",
                confidence=1.5,
            )

    def test_body_too_short_rejected(self):
        """Body must have min_length=10."""
        with pytest.raises(ValidationError):
            EmailExample(
                email_id="bad_004",
                subject="Test",
                body="short",  # < 10 chars
                sender="test@example.com",
                category="spam",
                confidence=0.8,
            )

    def test_subject_empty_rejected(self):
        """Subject must have min_length=1."""
        with pytest.raises(ValidationError):
            EmailExample(
                email_id="bad_005",
                subject="",  # empty
                body="A body that is long enough to pass validation.",
                sender="test@example.com",
                category="neutral",
                confidence=0.5,
            )

    def test_default_metadata_is_empty_dict(self):
        """metadata should default to an empty dict."""
        ex = EmailExample(
            email_id="defaults_001",
            subject="Test defaults",
            body="A body that is long enough to pass validation.",
            sender="test@example.com",
            category="neutral",
            confidence=0.5,
        )
        assert ex.metadata == {}

    def test_created_at_auto_set(self):
        """created_at should be auto-populated with a datetime."""
        ex = EmailExample(
            email_id="time_001",
            subject="Test time",
            body="A body that is long enough to pass validation.",
            sender="test@example.com",
            category="neutral",
            confidence=0.5,
        )
        assert isinstance(ex.created_at, datetime)


class TestSearchRequest:

    def test_valid_search(self):
        """A well-formed SearchRequest should be accepted."""
        req = SearchRequest(
            subject="Test",
            body="A body that is long enough for search.",
        )
        assert req.k == 3  # default
        assert req.category_filter is None

    def test_k_must_be_at_least_1(self):
        """k < 1 should fail validation."""
        with pytest.raises(ValidationError):
            SearchRequest(
                subject="Test",
                body="A body that is long enough for search.",
                k=0,
            )

    def test_k_max_is_10(self):
        """k > 10 should fail validation."""
        with pytest.raises(ValidationError):
            SearchRequest(
                subject="Test",
                body="A body that is long enough for search.",
                k=11,
            )

    def test_category_filter_valid_values(self):
        """category_filter accepts valid Literal values."""
        for cat in ("spam", "important", "neutral"):
            req = SearchRequest(
                subject="Test",
                body="A body that is long enough for search.",
                category_filter=cat,
            )
            assert req.category_filter == cat

    def test_category_filter_invalid_rejected(self):
        """Invalid category_filter should fail validation."""
        with pytest.raises(ValidationError):
            SearchRequest(
                subject="Test",
                body="A body that is long enough for search.",
                category_filter="phishing",
            )
