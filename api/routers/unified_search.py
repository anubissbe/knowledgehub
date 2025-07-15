"""Unified search router combining document and memory search"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from uuid import UUID
import logging
import asyncio

from ..dependencies import get_search_service, get_db
from ..schemas.search import SearchQuery, SearchType
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from ..memory_system.api.schemas import MemorySearchRequest
from ..memory_system.services.memory_search_service import memory_search_service

logger = logging.getLogger(__name__)

router = APIRouter()


class UnifiedSearchRequest(BaseModel):
    """Extended search request for unified search"""
    # Base search fields
    query: str = Field(..., min_length=1)
    search_type: SearchType = SearchType.HYBRID
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)
    filters: Optional[Dict[str, Any]] = None
    
    # Memory-specific fields
    include_memories: bool = True
    memory_user_id: Optional[str] = None
    memory_session_id: Optional[UUID] = None
    memory_min_importance: float = 0.5


class UnifiedSearchResult:
    """Unified search result combining documents and memories"""
    def __init__(self):
        self.documents = []
        self.memories = []
        self.total_documents = 0
        self.total_memories = 0
        self.search_time_ms = 0


@router.post("/unified", response_model=Dict[str, Any])
async def unified_search(
    query: UnifiedSearchRequest,
    db: Session = Depends(get_db),
    search_service=Depends(get_search_service)
):
    """
    Execute a unified search across documents and memories.
    
    This endpoint searches both:
    - Knowledge base documents (documentation, code, etc.)
    - User memories from conversations
    
    Results are returned in separate sections for clarity.
    """
    import time
    start_time = time.time()
    
    try:
        # Execute document and memory searches in parallel
        tasks = []
        
        # Document search task
        doc_search_query = SearchQuery(
            query=query.query,
            search_type=query.search_type,
            limit=query.limit,
            offset=query.offset,
            filters=query.filters
        )
        tasks.append(search_service.search(db, doc_search_query))
        
        # Memory search task (if enabled)
        if query.include_memories and query.memory_user_id:
            memory_request = MemorySearchRequest(
                query=query.query,
                user_id=query.memory_user_id,
                min_importance=query.memory_min_importance,
                limit=query.limit,
                offset=0,  # Memory search has its own pagination
                use_vector_search=query.search_type != SearchType.KEYWORD
            )
            tasks.append(memory_search_service.search_memories(db, memory_request))
        else:
            tasks.append(asyncio.create_task(_empty_memory_result()))
        
        # Execute searches in parallel
        doc_results, memory_results = await asyncio.gather(*tasks)
        
        # Calculate total search time
        total_time_ms = int((time.time() - start_time) * 1000)
        
        # Format unified response
        response = {
            "query": query.query,
            "search_type": query.search_type.value,
            "documents": {
                "results": doc_results.get("results", []),
                "total": doc_results.get("total", 0),
                "search_time_ms": doc_results.get("search_time_ms", 0)
            },
            "memories": {
                "results": [
                    {
                        "id": str(mem.id),
                        "content": mem.content,
                        "summary": mem.summary,
                        "memory_type": mem.memory_type,
                        "importance": mem.importance,
                        "relevance_score": mem.relevance_score,
                        "session_id": str(mem.session_id),
                        "created_at": mem.created_at.isoformat()
                    }
                    for mem in memory_results.results
                ] if hasattr(memory_results, 'results') else [],
                "total": memory_results.total if hasattr(memory_results, 'total') else 0,
                "search_time_ms": memory_results.search_time_ms if hasattr(memory_results, 'search_time_ms') else 0
            },
            "total_results": doc_results.get("total", 0) + (memory_results.total if hasattr(memory_results, 'total') else 0),
            "total_search_time_ms": total_time_ms,
            "filters": {
                "document_filters": query.filters,
                "memory_filters": {
                    "user_id": query.memory_user_id,
                    "session_id": str(query.memory_session_id) if query.memory_session_id else None,
                    "min_importance": query.memory_min_importance
                }
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Unified search error: {e}")
        raise HTTPException(status_code=500, detail="Unified search failed")


@router.get("/suggest")
async def search_suggestions(
    q: str = Query(..., min_length=2, description="Query prefix for suggestions"),
    include_memories: bool = Query(True, description="Include memory suggestions"),
    user_id: Optional[str] = Query(None, description="User ID for memory suggestions"),
    limit: int = Query(5, gt=0, le=20, description="Maximum suggestions per type"),
    db: Session = Depends(get_db)
):
    """
    Get search suggestions based on query prefix.
    
    Returns suggestions from:
    - Document titles and content
    - Memory content (if user_id provided)
    - Popular search terms
    """
    suggestions = {
        "query": q,
        "documents": [],
        "memories": [],
        "popular": []
    }
    
    try:
        # Get document suggestions (simple title/content prefix match)
        from ..models.document import Document, DocumentChunk
        
        # Search document titles
        doc_suggestions = db.query(Document.title).filter(
            Document.title.ilike(f"{q}%")
        ).limit(limit).all()
        
        suggestions["documents"] = [title[0] for title in doc_suggestions]
        
        # Get memory suggestions if requested
        if include_memories and user_id:
            from ..memory_system.models import Memory, MemorySession
            
            memory_suggestions = db.query(Memory.content).join(MemorySession).filter(
                MemorySession.user_id == user_id,
                Memory.content.ilike(f"{q}%")
            ).limit(limit).all()
            
            suggestions["memories"] = [
                content[0][:100] + "..." if len(content[0]) > 100 else content[0]
                for content in memory_suggestions
            ]
        
        # Add some popular/common suggestions (could be from search history)
        # For now, just return empty list
        suggestions["popular"] = []
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        return suggestions


async def _empty_memory_result():
    """Return empty memory search result"""
    class EmptyResult:
        results = []
        total = 0
        search_time_ms = 0
    return EmptyResult()