"""
API Gateway - Main FastAPI Application

This is the entry point for external requests.
It routes requests to appropriate microservices.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pydantic_settings import BaseSettings

from app.routes import classify
from app.clients.classifier_client import ClassifierClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
class Settings(BaseSettings):
    """API Gateway settings"""
    service_name: str = "api-gateway"
    service_version: str = "1.0.0"
    port: int = 8000
    classifier_service_url: str = "http://classifier:8001"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


settings = Settings()


# Lifespan context manager (startup/shutdown events)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Startup:
    - Initialize classifier client
    - Connect to other services
    
    Shutdown:
    - Close connections
    - Cleanup resources
    """
    # Startup
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    
    # Initialize classifier client
    client = ClassifierClient(base_url=settings.classifier_service_url)
    classify.classifier_client = client  # Inject into routes module
    
    logger.info("API Gateway initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway")
    await client.close()
    logger.info("API Gateway shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Email Intelligence API",
    description="Microservices API for intelligent email classification",
    version=settings.service_version,
    lifespan=lifespan
)

# CORS middleware (allows requests from web browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(classify.router)


# Root endpoint
@app.get("/")
async def root():
    """API root - returns service information"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "healthy"}