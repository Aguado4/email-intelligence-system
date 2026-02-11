"""
Vector DB Service - FastAPI Application

Provides HTTP endpoints for storing and retrieving email examples
using semantic similarity search with ChromaDB.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
from pydantic_settings import BaseSettings

from app.storage.vector_store import VectorStore
from app.models.schemas import (
    EmailExample,
    StoreExampleRequest,
    SearchRequest,
    SearchResponse,
    SimilarExample,
    StatsResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
class Settings(BaseSettings):
    """Vector DB Service settings"""
    service_name: str = "vector-db-service"
    service_version: str = "1.0.0"
    port: int = 8003
    chroma_host: str = "vector-db"
    chroma_port: int = 8000
    collection_name: str = "email_examples"
    embedding_model: str = "all-MiniLM-L6-v2"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


settings = Settings()

# Global vector store instance
vector_store: VectorStore = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Startup:
    - Initialize ChromaDB connection
    - Load ground truth examples
    
    Shutdown:
    - Close connections
    """
    global vector_store
    
    # Startup
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    
    try:
        # Initialize vector store
        vector_store = VectorStore(
            chroma_host=settings.chroma_host,
            chroma_port=settings.chroma_port,
            collection_name=settings.collection_name
        )
        
        # Load ground truth examples if collection is empty
        stats = vector_store.get_stats()
        if stats["total_examples"] == 0:
            logger.info("Collection is empty, loading ground truth examples...")
            await load_ground_truth_examples(vector_store)
        
        logger.info(f"Vector DB Service initialized with {stats['total_examples']} examples")
        
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Vector DB Service")


# Create FastAPI app
app = FastAPI(
    title="Email Vector DB Service",
    description="Semantic search for email examples using ChromaDB",
    version=settings.service_version,
    lifespan=lifespan
)


# ============= Helper Functions =============

async def load_ground_truth_examples(store: VectorStore):
    """
    Load ground truth examples from JSON file.
    
    This runs on startup if the collection is empty.
    """
    import json
    from pathlib import Path
    from datetime import datetime
    
    ground_truth_file = Path(__file__).parent.parent / "data" / "ground_truth.json"
    
    if not ground_truth_file.exists():
        logger.warning(f"Ground truth file not found: {ground_truth_file}")
        return
    
    try:
        with open(ground_truth_file, 'r') as f:
            examples_data = json.load(f)
        
        logger.info(f"Loading {len(examples_data)} ground truth examples...")
        
        for example_dict in examples_data:
            # Convert to EmailExample
            example = EmailExample(**example_dict)
            store.store_example(example)
        
        logger.info(f"Successfully loaded {len(examples_data)} examples")
        
    except Exception as e:
        logger.error(f"Failed to load ground truth: {e}")


# ============= API Endpoints =============

@app.get("/")
async def root():
    """Service information"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "description": "Semantic search for email classification examples"
    }


@app.get("/health")
async def health():
    """Health check"""
    try:
        stats = vector_store.get_stats()
        return {
            "status": "healthy",
            "vector_db_connected": True,
            "total_examples": stats["total_examples"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector DB unhealthy: {str(e)}")


@app.post("/store", status_code=201)
async def store_example(request: StoreExampleRequest):
    """
    Store a new email example in the vector database.
    
    This endpoint is used to add new labeled examples that can be
    used for few-shot learning.
    """
    try:
        example_id = vector_store.store_example(request.example)
        
        return {
            "status": "success",
            "email_id": example_id,
            "message": f"Example stored successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to store example: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_similar_examples(request: SearchRequest):
    """
    Search for similar email examples using semantic similarity.
    
    This is the main endpoint used by the classifier service for RAG.
    
    Args:
        request: Search parameters (subject, body, k, optional category filter)
        
    Returns:
        List of similar examples ranked by similarity
    """
    try:
        # Search vector database
        similar_examples = vector_store.search_similar(
            subject=request.subject,
            body=request.body,
            k=request.k,
            category_filter=request.category_filter
        )
        
        query_text = f"Subject: {request.subject}\nBody: {request.body[:100]}..."
        
        return SearchResponse(
            query=query_text,
            results=similar_examples,
            count=len(similar_examples)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get vector database statistics.
    
    Returns counts of examples by category and total count.
    """
    try:
        stats = vector_store.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/examples/{email_id}")
async def delete_example(email_id: str):
    """Delete an example from the database"""
    try:
        success = vector_store.delete_example(email_id)
        
        if success:
            return {"status": "success", "message": f"Deleted example {email_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"Example {email_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete example: {e}")
        raise HTTPException(status_code=500, detail=str(e))