"""Vector search API router for memory system"""

import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ....dependencies import get_db
from ...services.embedding_service import memory_embedding_service
from ..schemas import MemoryResponse
from .memory import memory_to_response

logger = logging.getLogger(__name__)
router = APIRouter()


class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search"""
    query: str = Field(..., description="Query text to search for")
    limit: int = Field(10, gt=0, le=50, description="Maximum number of results")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score")
    session_id: Optional[UUID] = Field(None, description="Filter by session ID")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    memory_types: Optional[List[str]] = Field(None, description="Filter by memory types")


class VectorSearchResult(BaseModel):
    """Single vector search result"""
    memory: MemoryResponse
    similarity: float = Field(..., description="Similarity score (0-1)")


class VectorSearchResponse(BaseModel):
    """Response model for vector search"""
    results: List[VectorSearchResult]
    query: str
    total: int


@router.post("/search", response_model=VectorSearchResponse)
async def vector_search_memories(
    search_request: VectorSearchRequest,
    db: Session = Depends(get_db)
):
    """Search memories using vector similarity"""
    try:
        # Generate embedding for the query
        embeddings_client = memory_embedding_service.embeddings_client
        query_embedding = await embeddings_client.generate_embedding(
            search_request.query,
            normalize=True
        )
        
        if not query_embedding:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate query embedding"
            )
        
        # Perform vector similarity search
        similar_memories = await memory_embedding_service.find_similar_memories(
            db=db,
            query_embedding=query_embedding,
            limit=search_request.limit,
            min_similarity=search_request.min_similarity,
            session_id=search_request.session_id,
            user_id=search_request.user_id
        )
        
        # Filter by memory types if specified
        if search_request.memory_types:
            similar_memories = [
                (memory, score) for memory, score in similar_memories
                if memory.memory_type in search_request.memory_types
            ]
        
        # Convert to response format
        results = [
            VectorSearchResult(
                memory=memory_to_response(memory),
                similarity=score
            )
            for memory, score in similar_memories
        ]
        
        return VectorSearchResponse(
            results=results,
            query=search_request.query,
            total=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail="Vector search failed")


@router.post("/similar/{memory_id}", response_model=VectorSearchResponse)
async def find_similar_memories(
    memory_id: UUID,
    limit: int = 10,
    min_similarity: float = 0.5,
    db: Session = Depends(get_db)
):
    """Find memories similar to a specific memory"""
    from ...models import Memory
    
    # Get the target memory
    memory = db.query(Memory).filter_by(id=memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    if not memory.embedding:
        raise HTTPException(
            status_code=400, 
            detail="Memory does not have an embedding"
        )
    
    try:
        # Find similar memories using the memory's embedding
        similar_memories = await memory_embedding_service.find_similar_memories(
            db=db,
            query_embedding=memory.embedding,
            limit=limit + 1,  # Get one extra to exclude self
            min_similarity=min_similarity,
            session_id=None,  # Search across all sessions
            user_id=None
        )
        
        # Filter out the query memory itself
        similar_memories = [
            (m, score) for m, score in similar_memories
            if m.id != memory_id
        ][:limit]
        
        # Convert to response format
        results = [
            VectorSearchResult(
                memory=memory_to_response(m),
                similarity=score
            )
            for m, score in similar_memories
        ]
        
        return VectorSearchResponse(
            results=results,
            query=f"Similar to memory: {memory.content[:50]}...",
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"Similar memory search failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to find similar memories"
        )


@router.post("/reindex/{session_id}")
async def reindex_session_embeddings(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Regenerate embeddings for all memories in a session"""
    from ...models import Memory
    
    # Get all memories in the session
    memories = db.query(Memory).filter_by(session_id=session_id).all()
    
    if not memories:
        raise HTTPException(
            status_code=404, 
            detail="No memories found for session"
        )
    
    try:
        # Generate embeddings in batch
        embeddings = await memory_embedding_service.generate_batch_embeddings(memories)
        
        # Update memories with new embeddings
        success_count = 0
        for memory, embedding in zip(memories, embeddings):
            if embedding:
                memory.set_embedding(embedding)
                success_count += 1
        
        db.commit()
        
        return {
            "message": f"Reindexed {success_count} of {len(memories)} memories",
            "session_id": session_id,
            "total_memories": len(memories),
            "embeddings_generated": success_count
        }
        
    except Exception as e:
        logger.error(f"Failed to reindex session {session_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail="Failed to reindex session embeddings"
        )