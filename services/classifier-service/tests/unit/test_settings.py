"""
Tests for config/settings.py — configuration management.

Concepts introduced:
  - pytest.raises  — asserting that exceptions are raised
  - monkeypatch / environment variable manipulation
  - Testing Pydantic Settings validation
"""

import pytest
from config.settings import Settings


class TestSettings:

    def test_default_values(self):
        """Settings should have sensible defaults even with no .env file."""
        s = Settings(
            _env_file=None,         # don't read .env
            gemini_api_key="test",  # provide a key so get_api_key works
        )
        assert s.llm_provider == "gemini"
        assert s.temperature == 0.0
        assert s.confidence_threshold == 0.75
        assert s.max_tokens == 1000

    def test_get_api_key_returns_correct_key(self):
        """get_api_key() should return the key matching the provider."""
        s = Settings(
            _env_file=None,
            llm_provider="gemini",
            gemini_api_key="my-gemini-key",
        )
        assert s.get_api_key() == "my-gemini-key"

    def test_get_api_key_openai(self):
        """get_api_key() should work for OpenAI provider."""
        s = Settings(
            _env_file=None,
            llm_provider="openai",
            openai_api_key="my-openai-key",
        )
        assert s.get_api_key() == "my-openai-key"

    def test_get_api_key_raises_when_missing(self):
        """get_api_key() should raise ValueError when key is not set."""
        s = Settings(
            _env_file=None,
            llm_provider="gemini",
            gemini_api_key=None,
        )
        with pytest.raises(ValueError, match="API key for provider 'gemini' not found"):
            s.get_api_key()

    def test_invalid_provider_rejected(self):
        """An invalid llm_provider should fail Pydantic validation."""
        with pytest.raises(Exception):  # ValidationError
            Settings(
                _env_file=None,
                llm_provider="invalid_provider",
            )

    def test_confidence_threshold_boundaries(self):
        """Confidence threshold should accept floats between 0 and 1."""
        s = Settings(_env_file=None, confidence_threshold=0.5)
        assert s.confidence_threshold == 0.5

        s2 = Settings(_env_file=None, confidence_threshold=0.0)
        assert s2.confidence_threshold == 0.0

    def test_settings_from_env(self, monkeypatch):
        """Settings should pick up values from environment variables."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")
        monkeypatch.setenv("TEMPERATURE", "0.5")

        s = Settings(_env_file=None)
        assert s.llm_provider == "openai"
        assert s.openai_api_key == "env-key-123"
        assert s.temperature == 0.5
