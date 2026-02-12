"""
Tests for app/storage/embeddings.py — embedding generation.

Concepts introduced:
  - @pytest.mark.slow  — marking tests that take time (model loading)
  - Testing numeric outputs (dimensions, types)
  - Semantic similarity assertions
"""

import pytest
import numpy as np


@pytest.mark.slow
class TestGenerateEmbedding:
    """These tests load a real model — mark them as slow."""

    def test_returns_list_of_floats(self):
        """Embedding should be a list of floats."""
        from app.storage.embeddings import generate_embedding

        result = generate_embedding("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embedding_dimension_is_384(self):
        """all-MiniLM-L6-v2 produces 384-dimensional vectors."""
        from app.storage.embeddings import generate_embedding

        result = generate_embedding("Test text")
        assert len(result) == 384

    def test_different_texts_produce_different_embeddings(self):
        """Semantically different texts should have different vectors."""
        from app.storage.embeddings import generate_embedding

        emb1 = generate_embedding("I love programming")
        emb2 = generate_embedding("The weather is sunny today")

        # They should not be identical
        assert emb1 != emb2

    def test_similar_texts_have_higher_cosine_similarity(self):
        """Semantically similar texts should be closer in vector space."""
        from app.storage.embeddings import generate_embedding

        emb_spam1 = np.array(generate_embedding("You won a free prize! Click here now!"))
        emb_spam2 = np.array(generate_embedding("Congratulations! Claim your free gift!"))
        emb_work = np.array(generate_embedding("Quarterly board meeting rescheduled to Friday"))

        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_spam_spam = cosine_sim(emb_spam1, emb_spam2)
        sim_spam_work = cosine_sim(emb_spam1, emb_work)

        # Two spam-like texts should be more similar to each other
        assert sim_spam_spam > sim_spam_work

    def test_generate_email_embedding(self):
        """generate_email_embedding combines subject + body."""
        from app.storage.embeddings import generate_email_embedding

        result = generate_email_embedding(
            subject="Test Subject",
            body="This is a test email body with enough content.",
        )
        assert isinstance(result, list)
        assert len(result) == 384
