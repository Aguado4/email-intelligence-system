"""
Tests for clean_json_response() — the simplest function in the codebase.

Concepts introduced:
  - Basic test functions (test_ prefix)
  - @pytest.mark.parametrize  — run the same test with many inputs
  - assert statements
"""

import pytest
from workflows.email_classifier import clean_json_response


# ---------- Simple cases ----------

def test_already_clean_json():
    """JSON without any wrapper should be returned as-is."""
    raw = '{"category": "spam", "confidence": 0.9}'
    assert clean_json_response(raw) == raw


def test_strips_whitespace():
    """Leading/trailing whitespace should be removed."""
    raw = '   {"key": "value"}   '
    assert clean_json_response(raw) == '{"key": "value"}'


# ---------- Markdown code-block wrappers ----------

@pytest.mark.parametrize(
    "raw, expected",
    [
        # ```json ... ```
        (
            '```json\n{"category": "spam"}\n```',
            '{"category": "spam"}',
        ),
        # ``` ... ``` (no language tag)
        (
            '```\n{"category": "neutral"}\n```',
            '{"category": "neutral"}',
        ),
        # Extra whitespace around the block
        (
            '  ```json\n{"a": 1}\n```  ',
            '{"a": 1}',
        ),
    ],
    ids=["json-fence", "plain-fence", "whitespace-around-fence"],
)
def test_removes_markdown_fences(raw, expected):
    """Markdown code fences wrapping JSON should be stripped."""
    assert clean_json_response(raw) == expected


# ---------- Edge cases ----------

def test_empty_string():
    """An empty string should remain empty."""
    assert clean_json_response("") == ""


def test_only_fences():
    """If the fences contain nothing useful, return what's left."""
    result = clean_json_response("```json\n```")
    assert result == ""


def test_nested_backticks_not_stripped():
    """Backticks inside the JSON value should be left alone."""
    raw = '```json\n{"code": "use `print()`"}\n```'
    assert "use `print()`" in clean_json_response(raw)
