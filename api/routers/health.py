
from fastapi import APIRouter
from typing import Dict

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "knowledgehub-api",
        "timestamp": "2025-08-17T00:00:00Z"
    }

@router.get("/api")
async def api_info() -> Dict[str, str]:
    """API information endpoint"""
    return {
        "name": "KnowledgeHub API",
        "version": "1.0.0",
        "status": "operational"
    }
