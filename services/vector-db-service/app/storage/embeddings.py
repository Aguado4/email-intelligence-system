"""
Embeddings generation using sentence-transformers.

This module handles converting text (subject + body) into
vector embeddings for similarity search.
"""

from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Get cached embedding model.
    
    Model: all-MiniLM-L6-v2
    - 384 dimensions
    - Fast inference
    - Good for semantic similarity
    - ~80MB model size
    
    Args:
        model_name: Name of sentence-transformers model
        
    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """
    Generate embedding vector for text.
    
    Args:
        text: Text to embed (usually subject + body combined)
        model_name: Embedding model to use
        
    Returns:
        List of floats representing the embedding vector
    """
    model = get_embedding_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_email_embedding(subject: str, body: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """
    Generate embedding for an email by combining subject and body.
    
    Format: "Subject: {subject}\nBody: {body}"
    This gives more weight to subject while including body context.
    
    Args:
        subject: Email subject line
        body: Email body content
        model_name: Embedding model to use
        
    Returns:
        Embedding vector
    """
    # Combine subject and body with clear delimiters
    combined_text = f"Subject: {subject}\nBody: {body[:500]}"  # Limit body to 500 chars for performance
    
    return generate_embedding(combined_text, model_name)