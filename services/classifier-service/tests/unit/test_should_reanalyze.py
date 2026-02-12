"""
Tests for should_reanalyze() — the conditional routing function.

Concepts introduced:
  - Boundary testing (exactly at threshold)
  - Fixture usage from conftest.py (sample_email_state)
  - Testing decision logic with multiple outcomes
"""

import pytest
from unittest.mock import patch
from workflows.email_classifier import should_reanalyze


class TestShouldReanalyze:
    """Group related tests inside a class for clarity."""

    # ---------- Below threshold → reanalyze ----------

    def test_low_confidence_triggers_reanalysis(self, sample_email_state):
        """Confidence below 0.75 should return 'reanalyze'."""
        state = {**sample_email_state, "confidence": 0.50, "retry_count": 0}
        assert should_reanalyze(state) == "reanalyze"

    def test_just_below_threshold(self, sample_email_state):
        """Confidence at 0.74 (just below 0.75) should trigger reanalysis."""
        state = {**sample_email_state, "confidence": 0.74, "retry_count": 0}
        assert should_reanalyze(state) == "reanalyze"

    # ---------- At / above threshold → end ----------

    def test_exactly_at_threshold(self, sample_email_state):
        """Confidence exactly at 0.75 should NOT trigger reanalysis."""
        state = {**sample_email_state, "confidence": 0.75, "retry_count": 0}
        assert should_reanalyze(state) == "end"

    def test_high_confidence_ends(self, sample_email_state):
        """Confidence well above threshold should end immediately."""
        state = {**sample_email_state, "confidence": 0.95, "retry_count": 0}
        assert should_reanalyze(state) == "end"

    # ---------- Max retries → max_retries ----------

    def test_max_retries_reached(self, sample_email_state):
        """Even low confidence should end when retry_count >= 1."""
        state = {**sample_email_state, "confidence": 0.30, "retry_count": 1}
        assert should_reanalyze(state) == "max_retries"

    def test_multiple_retries_still_stops(self, sample_email_state):
        """retry_count > 1 should also return max_retries."""
        state = {**sample_email_state, "confidence": 0.30, "retry_count": 5}
        assert should_reanalyze(state) == "max_retries"

    # ---------- Custom threshold via settings ----------

    def test_custom_threshold(self, sample_email_state):
        """Verify the function respects settings.confidence_threshold."""
        state = {**sample_email_state, "confidence": 0.60, "retry_count": 0}

        with patch("workflows.email_classifier.settings") as mock_settings:
            mock_settings.confidence_threshold = 0.50
            # 0.60 >= 0.50 → end
            assert should_reanalyze(state) == "end"

    def test_custom_threshold_low(self, sample_email_state):
        """With a higher threshold, more emails trigger reanalysis."""
        state = {**sample_email_state, "confidence": 0.80, "retry_count": 0}

        with patch("workflows.email_classifier.settings") as mock_settings:
            mock_settings.confidence_threshold = 0.90
            # 0.80 < 0.90 → reanalyze
            assert should_reanalyze(state) == "reanalyze"
