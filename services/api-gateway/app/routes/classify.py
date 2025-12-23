"""
Classification API routes.

These endpoints are exposed to external clients and handle
email classification requests.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated
import logging
import time

# Import shared models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "shared"))
from models.schemas import EmailInput, EmailProcessingResponse, ClassificationResult

from app.clients.classifier_client import ClassifierClient

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["classification"])

# We'll inject this dependency in main.py
classifier_client: ClassifierClient = None


def get_classifier_client() -> ClassifierClient:
    """
    Dependency injection for classifier client.
    
    This allows us to easily mock the client in tests.
    """
    if classifier_client is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier service client not initialized"
        )
    return classifier_client


@router.post("/classify", response_model=EmailProcessingResponse)
async def classify_email(
    email: EmailInput,
    client: Annotated[ClassifierClient, Depends(get_classifier_client)]
):
    """
    Classify an email into spam, important, or neutral.
    
    This endpoint:
    1. Validates the input email
    2. Sends it to the classifier service
    3. Returns structured classification results
    
    Args:
        email: Email data to classify
        client: Injected classifier client
        
    Returns:
        Classification results with confidence and reasoning
        
    Raises:
        HTTPException: If classification fails
    """
    start_time = time.time()
    
    logger.info(f"Received classification request for email {email.email_id}")
    
    try:
        # Call classifier service
        result = await client.classify_email(
            email_id=email.email_id,
            subject=email.subject,
            body=email.body,
            sender=email.sender
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build response
        classification = ClassificationResult(
            category=result["category"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            keywords=result["keywords"]
        )
        
        response = EmailProcessingResponse(
            email_id=email.email_id,
            classification=classification,
            processing_time_ms=processing_time_ms
        )
        
        logger.info(
            f"Classification complete for {email.email_id}: "
            f"{classification.category} ({classification.confidence:.2f})"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Classification failed for {email.email_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.get("/health")
async def health_check(
    client: Annotated[ClassifierClient, Depends(get_classifier_client)]
):
    """
    Health check endpoint.
    
    Checks:
    1. API Gateway is running
    2. Classifier service is reachable
    
    Returns:
        Health status of both services
    """
    classifier_health = await client.health_check()
    
    return {
        "api_gateway": "healthy",
        "classifier_service": classifier_health.get("status", "unknown"),
        "timestamp": time.time()
    }