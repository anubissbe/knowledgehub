"""
Public Search Router - Provides unauthenticated search access
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import logging

from ..dependencies import get_db, get_search_service
from ..schemas.search import SearchQuery, SearchType
from ..models.document import Document, DocumentChunk
from ..models.memory import MemoryItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/public", tags=["public-search"])


@router.get("/search")
async def public_search(
    q: str = Query(..., min_length=1, description="Search query"),
    search_type: SearchType = Query(SearchType.HYBRID, description="Type of search"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    include_memories: bool = Query(False, description="Include memory items"),
    db: Session = Depends(get_db),
    search_service=Depends(get_search_service)
) -> Dict[str, Any]:
    """
    Public search endpoint - no authentication required
    
    Searches across:
    - Knowledge base documents
    - Memory items (if requested)
    
    Returns simplified results for public consumption
    """
    try:
        # Create search query
        search_query = SearchQuery(
            query=q,
            search_type=search_type,
            limit=limit,
            offset=offset
        )
        
        # Execute document search
        doc_results = await search_service.search(db, search_query)
        
        logger.info(f"Search results structure: {doc_results}")
        logger.info(f"Number of results: {len(doc_results.get('results', []))}")
        if doc_results.get('results'):
            logger.info(f"First result keys: {list(doc_results['results'][0].keys())}")
        
        # Prepare response
        response = {
            "query": q,
            "search_type": search_type.value,
            "total_results": doc_results.get("total", 0),
            "documents": []
        }
        
        # Simplify document results for public
        for doc in doc_results.get("results", []):
            response["documents"].append({
                "id": doc.get("id"),
                "title": doc.get("title", "Untitled"),
                "content": doc.get("content", "")[:500] + "..." if len(doc.get("content", "")) > 500 else doc.get("content", ""),
                "score": doc.get("score", 0),
                "source": doc.get("source_name", "Unknown"),
                "url": doc.get("url"),
                "created_at": doc.get("created_at")
            })
        
        # Include memory search if requested
        if include_memories:
            memory_results = []
            
            # Simple keyword search in memories
            memories = db.query(MemoryItem).filter(
                MemoryItem.content.ilike(f"%{q}%")
            ).limit(5).all()
            
            for memory in memories:
                memory_results.append({
                    "id": str(memory.id),
                    "content": memory.content[:300] + "..." if len(memory.content) > 300 else memory.content,
                    "tags": memory.tags,
                    "created_at": memory.created_at.isoformat()
                })
            
            response["memories"] = memory_results
            response["total_results"] += len(memory_results)
        
        return response
        
    except Exception as e:
        logger.error(f"Public search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/search/suggest")
async def public_search_suggestions(
    q: str = Query(..., min_length=2, description="Query prefix"),
    limit: int = Query(5, ge=1, le=20, description="Maximum suggestions"),
    db: Session = Depends(get_db)
) -> Dict[str, List[str]]:
    """
    Get search suggestions for autocomplete
    
    Returns suggestions based on:
    - Document titles
    - Popular search terms
    - Memory content (limited)
    """
    try:
        suggestions = {
            "query": q,
            "suggestions": []
        }
        
        # Get document title suggestions
        doc_titles = db.query(Document.title).filter(
            Document.title.ilike(f"{q}%")
        ).limit(limit).all()
        
        suggestions["suggestions"] = [title[0] for title in doc_titles if title[0]]
        
        # Add some common search suggestions
        common_terms = [
            "API documentation",
            "Getting started",
            "Configuration",
            "Troubleshooting",
            "Best practices"
        ]
        
        for term in common_terms:
            if term.lower().startswith(q.lower()) and term not in suggestions["suggestions"]:
                suggestions["suggestions"].append(term)
                if len(suggestions["suggestions"]) >= limit:
                    break
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        return {"query": q, "suggestions": []}


@router.get("/topics")
async def get_topics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get available topics/categories for browsing
    
    Returns a list of topics with document counts
    """
    try:
        # Get unique source types
        sources = db.query(
            Document.source_type,
            db.func.count(Document.id).label("count")
        ).group_by(Document.source_type).all()
        
        topics = []
        for source_type, count in sources:
            if source_type:
                topics.append({
                    "name": source_type.replace("_", " ").title(),
                    "key": source_type,
                    "document_count": count
                })
        
        # Add memory topic if any exist
        memory_count = db.query(MemoryItem).count()
        if memory_count > 0:
            topics.append({
                "name": "Memories",
                "key": "memories",
                "document_count": memory_count
            })
        
        return {
            "total_topics": len(topics),
            "topics": sorted(topics, key=lambda x: x["document_count"], reverse=True)
        }
        
    except Exception as e:
        logger.error(f"Get topics error: {e}")
        return {"total_topics": 0, "topics": []}


@router.get("/stats")
async def get_search_stats(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get public search statistics
    
    Returns general stats about the knowledge base
    """
    try:
        doc_count = db.query(Document).count()
        chunk_count = db.query(DocumentChunk).count()
        memory_count = db.query(MemoryItem).count()
        
        # Get document age range
        oldest_doc = db.query(Document.created_at).order_by(Document.created_at).first()
        newest_doc = db.query(Document.created_at).order_by(Document.created_at.desc()).first()
        
        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "total_memories": memory_count,
            "avg_chunks_per_doc": chunk_count / doc_count if doc_count > 0 else 0,
            "oldest_document": oldest_doc[0].isoformat() if oldest_doc else None,
            "newest_document": newest_doc[0].isoformat() if newest_doc else None,
            "searchable_items": doc_count + memory_count
        }
        
    except Exception as e:
        logger.error(f"Get stats error: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_memories": 0,
            "searchable_items": 0
        }