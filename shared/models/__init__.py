"""Shared models package"""
from .schemas import (
    EmailInput,
    ClassificationResult,
    EmailProcessingResponse,
    GroundTruth,
    EvaluationMetrics,
    EvaluationReport
)

__all__ = [
    "EmailInput",
    "ClassificationResult", 
    "EmailProcessingResponse",
    "GroundTruth",
    "EvaluationMetrics",
    "EvaluationReport"
]