"""
Pydantic models for Vector DB Service.

Defines the structure for storing and retrieving email examples.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class EmailExample(BaseModel):
    """
    Email example for storage in vector database.
    
    These examples are used for few-shot learning when
    classification confidence is low.
    """
    email_id: str = Field(..., description="Unique identifier")
    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=10, max_length=10000)
    sender: str = Field(..., description="Sender email or domain")
    category: Literal["spam", "important", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[dict] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "example_spam_001",
                "subject": "URGENT: Verify your account",
                "body": "Click here immediately to verify...",
                "sender": "noreply@phishing.com",
                "category": "spam",
                "confidence": 1.0,
                "metadata": {"source": "labeled_dataset", "verified": True}
            }
        }


class StoreExampleRequest(BaseModel):
    """Request to store an email example"""
    example: EmailExample


class SearchRequest(BaseModel):
    """Request to search for similar examples"""
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=10)
    k: int = Field(default=3, ge=1, le=10, description="Number of results to return")
    category_filter: Optional[Literal["spam", "important", "neutral"]] = None


class SimilarExample(BaseModel):
    """A similar example with distance score"""
    email_id: str
    subject: str
    body: str
    sender: str
    category: str
    confidence: float
    similarity_score: float = Field(..., description="Cosine similarity (0-1, higher is more similar)")
    metadata: dict


class SearchResponse(BaseModel):
    """Response containing similar examples"""
    query: str = Field(..., description="Original query")
    results: list[SimilarExample]
    count: int


class StatsResponse(BaseModel):
    """Database statistics"""
    total_examples: int
    examples_by_category: dict[str, int]
    collection_name: str