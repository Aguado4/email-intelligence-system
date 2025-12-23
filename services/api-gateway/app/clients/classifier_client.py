"""
Client for communicating with the Classifier Service.

This abstracts the internal microservice communication,
making it easy to call the classifier from the API Gateway.
"""

import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ClassifierClient:
    """
    HTTP client for the Classifier Service.
    
    In a microservices architecture, services communicate via HTTP.
    This client handles all communication with the classifier.
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the classifier client.
        
        Args:
            base_url: Base URL of classifier service (e.g., http://classifier:8001)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        logger.info(f"Initialized ClassifierClient with base_url: {base_url}")
    
    async def classify_email(
        self,
        email_id: str,
        subject: str,
        body: str,
        sender: str
    ) -> dict:
        """
        Send email to classifier service for classification.
        
        Args:
            email_id: Unique email identifier
            subject: Email subject
            body: Email body
            sender: Sender email address
            
        Returns:
            Classification result dictionary
            
        Raises:
            httpx.HTTPError: If request fails
        """
        url = f"{self.base_url}/classify"
        
        payload = {
            "email_id": email_id,
            "subject": subject,
            "body": body,
            "sender": sender
        }
        
        logger.info(f"Sending classification request for email {email_id}")
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Classification successful for {email_id}: {result['category']}")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Classification request failed: {e}")
            raise
    
    async def health_check(self) -> dict:
        """
        Check if classifier service is healthy.
        
        Returns:
            Health status dictionary
        """
        url = f"{self.base_url}/health"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()