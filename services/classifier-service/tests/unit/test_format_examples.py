"""
Tests for format_examples_for_prompt() — RAG prompt formatting.

Concepts introduced:
  - Using fixtures defined in conftest.py (sample_similar_examples)
  - String assertions (in, startswith)
  - Testing empty inputs
"""

import pytest
from workflows.email_classifier import format_examples_for_prompt
from clients.vector_db_client import SimilarExample


class TestFormatExamplesForPrompt:

    def test_empty_list_returns_empty_string(self):
        """No examples → empty string (no few-shot section)."""
        assert format_examples_for_prompt([]) == ""

    def test_formats_single_example(self):
        """One example should produce a numbered block."""
        examples = [
            SimilarExample(
                email_id="ex_001",
                subject="Test Subject",
                body="Test body content here for the email example.",
                sender="test@example.com",
                category="spam",
                confidence=0.9,
                similarity_score=0.88,
            )
        ]
        result = format_examples_for_prompt(examples)

        assert "Example 1:" in result
        assert "Subject: Test Subject" in result
        assert "From: test@example.com" in result
        assert "Classification: spam" in result
        assert "0.88" in result  # similarity score

    def test_formats_multiple_examples(self, sample_similar_examples):
        """Multiple examples should be numbered sequentially."""
        result = format_examples_for_prompt(sample_similar_examples)

        assert "Example 1:" in result
        assert "Example 2:" in result
        assert result.startswith("Here are similar emails")

    def test_header_present(self, sample_similar_examples):
        """Output should start with an explanatory header."""
        result = format_examples_for_prompt(sample_similar_examples)
        assert result.startswith("Here are similar emails and their correct classifications:")

    def test_body_truncated_at_200_chars(self):
        """Long bodies should be truncated in the output."""
        long_body = "A" * 500
        examples = [
            SimilarExample(
                email_id="ex_long",
                subject="Long Email",
                body=long_body,
                sender="long@example.com",
                category="neutral",
                confidence=0.7,
                similarity_score=0.75,
            )
        ]
        result = format_examples_for_prompt(examples)

        # The function slices body[:200], so we shouldn't see all 500 A's
        assert "A" * 201 not in result
        assert "..." in result

    def test_confidence_and_similarity_shown(self, sample_similar_examples):
        """Both confidence and similarity scores should appear."""
        result = format_examples_for_prompt(sample_similar_examples)

        # First example: confidence=1.0, similarity=0.92
        assert "confidence: 1.0" in result
        assert "0.92" in result
