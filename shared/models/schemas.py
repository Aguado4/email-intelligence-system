"""
Shared Pydantic models for type safety across microservices.

These models define the contract between services and ensure
consistent data validation throughout the system.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Literal, Optional
from datetime import datetime


# ============= Email Classification Models =============

class EmailInput(BaseModel):
    """
    Input model for email classification requests.
    
    Attributes:
        email_id: Unique identifier for tracking
        subject: Email subject line
        body: Email body content
        sender: Sender email address
    """
    email_id: str = Field(..., description="Unique email identifier")
    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=10, max_length=10000)
    sender: EmailStr = Field(..., description="Sender email address")

    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "email_001",
                "subject": "Urgent: Account verification required",
                "body": "Click here to verify your account immediately...",
                "sender": "noreply@suspicious.com"
            }
        }


class ClassificationResult(BaseModel):
    """
    Classification output with category and confidence.
    
    Categories:
        - spam: Unsolicited/malicious emails
        - important: Critical business emails
        - neutral: Regular correspondence
    """
    category: Literal["spam", "important", "neutral"] = Field(
        ..., 
        description="Email category"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        ..., 
        description="Explanation for classification"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms that influenced decision"
    )


class EmailProcessingResponse(BaseModel):
    """
    Complete response from the classification pipeline.
    """
    email_id: str
    classification: ClassificationResult
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service_version: str = "1.0.0"


# ============= Evaluation Models =============

class GroundTruth(BaseModel):
    """
    Ground truth data for evaluation.
    """
    email_id: str
    expected_category: Literal["spam", "important", "neutral"]
    notes: Optional[str] = None


class EvaluationMetrics(BaseModel):
    """
    Evaluation metrics for model performance.
    
    Metrics:
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1: Harmonic mean of precision and recall
    """
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    total_samples: int
    correct_predictions: int
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy from correct predictions"""
        return self.correct_predictions / self.total_samples if self.total_samples > 0 else 0.0


class EvaluationReport(BaseModel):
    """
    Complete evaluation report with per-category breakdown.
    """
    overall_metrics: EvaluationMetrics
    per_category_metrics: dict[str, EvaluationMetrics]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str