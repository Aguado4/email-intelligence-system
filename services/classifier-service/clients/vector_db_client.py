"""
HTTP client for Vector DB Service.

Provides methods to search for similar email examples
for RAG-enhanced classification.
"""

import httpx
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimilarExample:
    """A similar email example from the vector database."""
    email_id: str
    subject: str
    body: str
    sender: str
    category: str
    confidence: float
    similarity_score: float


class VectorDBClient:
    """
    Client for the Vector DB Service.

    Used to fetch similar email examples for few-shot learning
    when classification confidence is low.
    """

    def __init__(self, base_url: str):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the vector-db-service (from settings)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = 10.0

    async def search_similar(
        self,
        subject: str,
        body: str,
        k: int = 3,
        category_filter: Optional[str] = None
    ) -> list[SimilarExample]:
        """
        Search for similar email examples.

        Args:
            subject: Email subject to search for
            body: Email body to search for
            k: Number of results to return
            category_filter: Optional category to filter by

        Returns:
            List of similar examples sorted by similarity
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "subject": subject,
                    "body": body,
                    "k": k
                }

                if category_filter:
                    payload["category_filter"] = category_filter

                response = await client.post(
                    f"{self.base_url}/search",
                    json=payload
                )
                response.raise_for_status()

                data = response.json()

                examples = [
                    SimilarExample(
                        email_id=result["email_id"],
                        subject=result["subject"],
                        body=result["body"],
                        sender=result["sender"],
                        category=result["category"],
                        confidence=result["confidence"],
                        similarity_score=result["similarity_score"]
                    )
                    for result in data.get("results", [])
                ]

                logger.info(f"Found {len(examples)} similar examples")
                return examples

        except httpx.HTTPStatusError as e:
            logger.error(f"Vector DB HTTP error: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Vector DB request error: {e}")
            return []
        except Exception as e:
            logger.error(f"Vector DB client error: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if the vector-db-service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
