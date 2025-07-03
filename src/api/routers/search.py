"""Search router"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from sqlalchemy.orm import Session
import logging

from ..dependencies import get_search_service, get_db
from ..schemas.search import SearchQuery, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=SearchResponse)
async def search(
    query: SearchQuery,
    db: Session = Depends(get_db),
    search_service=Depends(get_search_service)
):
    """
    Execute a search query against the knowledge base.
    
    Supports three search types:
    - hybrid: Combines semantic and keyword search (default)
    - vector: Pure semantic/similarity search
    - keyword: Traditional keyword-based search
    """
    try:
        results = await search_service.search(db, query)
        return results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")