"""
ChromaDB vector store interface.

Handles all interactions with ChromaDB for storing and
retrieving email examples using semantic similarity.
"""

import chromadb
from chromadb.config import Settings
import logging
from typing import Optional
import uuid

from app.storage.embeddings import generate_email_embedding
from app.models.schemas import EmailExample, SimilarExample

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database interface using ChromaDB.
    
    Provides methods to:
    - Store email examples with embeddings
    - Search for similar emails
    - Manage the vector database
    """
    
    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        collection_name: str = "email_examples"
    ):
        """
        Initialize ChromaDB client.
        
        Args:
            chroma_host: ChromaDB server host
            chroma_port: ChromaDB server port
            collection_name: Name of the collection to use
        """
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.collection_name = collection_name
        
        # Connect to ChromaDB
        logger.info(f"Connecting to ChromaDB at {chroma_host}:{chroma_port}")
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Email examples for few-shot classification"}
        )
        
        logger.info(f"Connected to collection: {collection_name}")
    
    def store_example(self, example: EmailExample) -> str:
        """
        Store an email example in the vector database.
        
        Args:
            example: EmailExample to store
            
        Returns:
            ID of stored example
        """
        # Generate embedding
        embedding = generate_email_embedding(example.subject, example.body)
        
        # Prepare metadata (ChromaDB requires flat dict)
        metadata = {
            "subject": example.subject,
            "sender": example.sender,
            "category": example.category,
            "confidence": example.confidence,
            "created_at": example.created_at.isoformat(),
            **example.metadata  # Merge additional metadata
        }
        
        # Store in ChromaDB
        self.collection.add(
            ids=[example.email_id],
            embeddings=[embedding],
            documents=[example.body],  # Store body as document
            metadatas=[metadata]
        )
        
        logger.info(f"Stored example {example.email_id} in vector DB")
        return example.email_id
    
    def search_similar(
        self,
        subject: str,
        body: str,
        k: int = 3,
        category_filter: Optional[str] = None
    ) -> list[SimilarExample]:
        """
        Search for similar email examples.
        
        Args:
            subject: Query email subject
            body: Query email body
            k: Number of results to return
            category_filter: Optional category to filter by
            
        Returns:
            List of similar examples with similarity scores
        """
        # Generate query embedding
        query_embedding = generate_email_embedding(subject, body)
        
        # Build where filter
        where_filter = {"category": category_filter} if category_filter else None
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter
        )
        
        # Parse results
        similar_examples = []
        
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity (ChromaDB uses L2 distance)
                # For normalized vectors, similarity = 1 - (distance^2 / 2)
                # Simplified: higher distance = lower similarity
                similarity_score = max(0, 1 - (distance / 2))
                
                similar_example = SimilarExample(
                    email_id=results['ids'][0][i],
                    subject=metadata['subject'],
                    body=results['documents'][0][i],
                    sender=metadata['sender'],
                    category=metadata['category'],
                    confidence=metadata['confidence'],
                    similarity_score=round(similarity_score, 3),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ['subject', 'sender', 'category', 'confidence']}
                )
                similar_examples.append(similar_example)
        
        logger.info(f"Found {len(similar_examples)} similar examples for query")
        return similar_examples
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database stats
        """
        count = self.collection.count()
        
        # Get all items to count by category (for small datasets)
        if count > 0 and count < 1000:  # Only for small datasets
            all_items = self.collection.get()
            categories = {}
            for metadata in all_items['metadatas']:
                category = metadata.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
        else:
            categories = {"note": "Too many items to categorize"}
        
        return {
            "total_examples": count,
            "examples_by_category": categories,
            "collection_name": self.collection_name
        }
    
    def delete_example(self, email_id: str) -> bool:
        """
        Delete an example from the database.
        
        Args:
            email_id: ID of example to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.collection.delete(ids=[email_id])
            logger.info(f"Deleted example {email_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {email_id}: {e}")
            return False