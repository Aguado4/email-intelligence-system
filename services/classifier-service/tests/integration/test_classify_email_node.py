"""
Integration test for classify_email_node — the main LangGraph node.

Concepts introduced:
  - mocker.patch (pytest-mock) to replace the LLM at runtime
  - Testing async node functions with real state dicts
  - Verifying state transitions
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestClassifyEmailNode:

    @pytest.fixture
    def good_llm_response(self):
        """A well-formed LLM response."""
        return MagicMock(
            content='{"category": "spam", "confidence": 0.95, '
                    '"reasoning": "Lottery scam", "keywords": ["prize", "click"]}'
        )

    @pytest.fixture
    def malformed_llm_response(self):
        """An LLM response that is not valid JSON."""
        return MagicMock(content="I think this is spam because it mentions prizes.")

    async def test_successful_classification(
        self, sample_email_state, good_llm_response
    ):
        """The node should parse the LLM JSON and update the state."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = good_llm_response

        with patch("workflows.email_classifier.get_llm", return_value=mock_llm):
            from workflows.email_classifier import classify_email_node

            result = await classify_email_node(sample_email_state)

        assert result["category"] == "spam"
        assert result["confidence"] == 0.95
        assert result["processing_stage"] == "classified"
        assert "prize" in result["keywords"]

    async def test_json_parse_error_defaults_to_neutral(
        self, sample_email_state, malformed_llm_response
    ):
        """When the LLM returns non-JSON, the node should fallback gracefully."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = malformed_llm_response

        with patch("workflows.email_classifier.get_llm", return_value=mock_llm):
            from workflows.email_classifier import classify_email_node

            result = await classify_email_node(sample_email_state)

        assert result["category"] == "neutral"
        assert result["confidence"] == 0.3
        assert result["processing_stage"] == "error_json_parse"

    async def test_llm_exception_handled(self, sample_email_state):
        """If the LLM raises an exception, the node should not crash."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("API quota exceeded")

        with patch("workflows.email_classifier.get_llm", return_value=mock_llm):
            from workflows.email_classifier import classify_email_node

            result = await classify_email_node(sample_email_state)

        assert result["category"] == "neutral"
        assert result["confidence"] == 0.3
        assert result["processing_stage"] == "error"

    async def test_invalid_category_from_llm(self, sample_email_state):
        """If the LLM returns an invalid category, treat it as an error."""
        bad_response = MagicMock(
            content='{"category": "phishing", "confidence": 0.9, '
                    '"reasoning": "Looks like phishing", "keywords": ["link"]}'
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = bad_response

        with patch("workflows.email_classifier.get_llm", return_value=mock_llm):
            from workflows.email_classifier import classify_email_node

            result = await classify_email_node(sample_email_state)

        # Invalid category should fall through to the error handler
        assert result["category"] == "neutral"
        assert result["processing_stage"] == "error"

    async def test_missing_fields_from_llm(self, sample_email_state):
        """If the LLM omits required fields, treat it as an error."""
        partial_response = MagicMock(
            content='{"category": "spam"}'  # missing confidence, reasoning, keywords
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = partial_response

        with patch("workflows.email_classifier.get_llm", return_value=mock_llm):
            from workflows.email_classifier import classify_email_node

            result = await classify_email_node(sample_email_state)

        assert result["processing_stage"] == "error"
