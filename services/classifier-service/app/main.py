"""
Classifier Service - FastAPI wrapper for LangGraph workflow
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from workflows.email_classifier import classify_email

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Email Classifier Service",
    description="LangGraph-powered email classification microservice",
    version="1.0.0"
)


# Request/Response models
class ClassifyRequest(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str


class ClassifyResponse(BaseModel):
    email_id: str
    category: str
    confidence: float
    reasoning: str
    keywords: list[str]
    processing_stage: str


@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(request: ClassifyRequest):
    """
    Classify an email using LangGraph workflow.
    
    This is the internal API endpoint called by the API Gateway.
    """
    logger.info(f"Received classification request for email {request.email_id}")
    
    try:
        result = await classify_email(
            email_id=request.email_id,
            subject=request.subject,
            body=request.body,
            sender=request.sender
        )
        
        logger.info(f"Classification complete for {request.email_id}")
        return ClassifyResponse(**result)
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "classifier"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "classifier",
        "version": "1.0.0",
        "status": "running"
    }