"""Search router"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from sqlalchemy.orm import Session
import logging

from ..dependencies import get_search_service, get_db
from ..schemas.search import SearchQuery, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def search(
    query: SearchQuery,
    db: Session = Depends(get_db)
):
    """
    Execute a search query against the knowledge base.
    
    Supports three search types:
    - hybrid: Combines semantic and keyword search (default)
    - vector: Pure semantic/similarity search
    - keyword: Traditional keyword-based search
    """
    try:
        # Simple fallback search implementation
        from ..models.memory import Memory
        from sqlalchemy import or_, func
        
        # Basic text search in memories
        search_term = f"%{query.query}%"
        memories = db.query(Memory).filter(
            or_(
                Memory.content.ilike(search_term),
                func.array_to_string(Memory.tags, ',').ilike(search_term)
            )
        ).limit(query.limit).all()
        
        # Convert to search result format
        results = []
        for memory in memories:
            results.append({
                "id": str(memory.id),
                "content": memory.content,
                "score": 1.0,  # Default score
                "source": "memory",
                "metadata": {
                    "memory_type": memory.memory_type,
                    "tags": memory.tags,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None
                }
            })
        
        return {
            "results": results,
            "total": len(results),
            "query": query.query,
            "search_type": "simple_text"
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        logger.error(f"Search traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")