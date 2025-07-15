"""FastAPI dependencies"""

from typing import Generator
from sqlalchemy.orm import Session

from .models import get_db
from .services.cache import redis_client
from .services.vector_store import vector_store
from .services.message_queue import message_queue


def get_source_service():
    """Get source service instance"""
    from .services.source_service import SourceService
    return SourceService()


def get_job_service():
    """Get job service instance"""
    from .services.job_service import JobService
    return JobService()


def get_search_service():
    """Get search service instance"""
    from .services.search_service import SearchService
    return SearchService()


def get_memory_service():
    """Get memory service instance"""
    from .services.memory_service import MemoryService
    return MemoryService()


def get_current_user(request):
    """Get current authenticated user from request state"""
    if hasattr(request.state, 'api_key') and request.state.authenticated:
        return {
            "id": request.state.api_key["id"],
            "name": request.state.api_key["name"], 
            "permissions": request.state.api_key["permissions"],
            "type": "api_key"
        }
    # Fallback for development/unauthenticated requests
    return {"id": "system", "name": "system", "permissions": ["read"], "type": "system"}


# Re-export commonly used dependencies
__all__ = [
    "get_db",
    "redis_client",
    "vector_store",
    "message_queue",
    "get_source_service",
    "get_job_service",
    "get_search_service",
    "get_memory_service",
    "get_current_user"
]