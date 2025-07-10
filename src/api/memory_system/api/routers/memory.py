"""Memory CRUD API router"""

import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ....models import get_db
from ...models import Memory, MemoryType
from ..schemas import (
    MemoryCreate, MemoryUpdate, MemoryResponse,
    MemoryBatchCreate, MemoryBatchResponse,
    MemorySearchRequest, MemorySearchResponse
)
from ...services.embedding_service import memory_embedding_service

logger = logging.getLogger(__name__)
router = APIRouter()


def memory_to_response(memory: Memory) -> MemoryResponse:
    """Convert SQLAlchemy model to response schema"""
    return MemoryResponse(
        id=memory.id,
        session_id=memory.session_id,
        content=memory.content,
        summary=memory.summary,
        memory_type=memory.memory_type,
        importance=memory.importance,
        confidence=memory.confidence,
        entities=memory.entities or [],
        related_memories=memory.related_memories or [],
        metadata=memory.memory_metadata or {},
        access_count=memory.access_count,
        last_accessed=memory.last_accessed,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        has_embedding=memory.embedding is not None,
        age_days=memory.age_days,
        relevance_score=memory.relevance_score,
        is_recent=memory.is_recent,
        is_high_importance=memory.is_high_importance
    )


@router.post("/", response_model=MemoryResponse)
async def create_memory(
    memory_data: MemoryCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new memory"""
    try:
        # Validate session exists
        from ...models import MemorySession
        session = db.query(MemorySession).filter_by(id=memory_data.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create memory
        memory = Memory(
            session_id=memory_data.session_id,
            content=memory_data.content,
            summary=memory_data.summary,
            memory_type=memory_data.memory_type.value,  # Use string value directly
            importance=memory_data.importance,
            confidence=memory_data.confidence,
            entities=memory_data.entities or [],
            memory_metadata=memory_data.metadata or {}
        )
        
        # Add related memories
        if memory_data.related_memories:
            for related_id in memory_data.related_memories:
                memory.add_related_memory(related_id)
        
        db.add(memory)
        db.commit()
        db.refresh(memory)
        
        # Schedule embedding generation in background
        background_tasks.add_task(
            memory_embedding_service.update_memory_embedding,
            db,
            memory.id
        )
        
        logger.info(f"Created memory {memory.id} for session {session.id}")
        return memory_to_response(memory)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create memory: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create memory")


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific memory"""
    memory = db.query(Memory).filter_by(id=memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Update access tracking
    memory.update_access()
    db.commit()
    
    return memory_to_response(memory)


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: UUID,
    update_data: MemoryUpdate,
    db: Session = Depends(get_db)
):
    """Update a memory"""
    memory = db.query(Memory).filter_by(id=memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    try:
        # Update fields
        if update_data.summary is not None:
            memory.summary = update_data.summary
        
        if update_data.importance is not None:
            memory.importance = update_data.importance
        
        if update_data.entities is not None:
            memory.entities = update_data.entities
        
        if update_data.metadata is not None:
            # Update metadata by creating a new dict that merges existing and new data
            if memory.memory_metadata:
                updated_metadata = dict(memory.memory_metadata)
                updated_metadata.update(update_data.metadata)
                memory.memory_metadata = updated_metadata
            else:
                memory.memory_metadata = update_data.metadata
        
        db.commit()
        db.refresh(memory)
        
        logger.info(f"Updated memory {memory_id}")
        return memory_to_response(memory)
        
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update memory")


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory_put(
    memory_id: UUID,
    update_data: MemoryUpdate,
    db: Session = Depends(get_db)
):
    """Update a memory (PUT method - same as PATCH for backward compatibility)"""
    # Delegate to the PATCH handler for consistency
    return await update_memory(memory_id, update_data, db)


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: UUID,
    db: Session = Depends(get_db)
) -> dict:
    """Delete a memory"""
    memory = db.query(Memory).filter_by(id=memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    try:
        db.delete(memory)
        db.commit()
        
        logger.info(f"Deleted memory {memory_id}")
        return {"message": "Memory deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete memory")


@router.get("/session/{session_id}", response_model=List[MemoryResponse])
async def get_session_memories(
    session_id: UUID,
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    min_importance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance"),
    limit: int = Query(50, gt=0, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db)
):
    """Get all memories for a session"""
    query = db.query(Memory).filter_by(session_id=session_id)
    
    if memory_type:
        try:
            query = query.filter(Memory.memory_type == memory_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid memory type")
    
    if min_importance > 0:
        query = query.filter(Memory.importance >= min_importance)
    
    memories = query.order_by(desc(Memory.created_at)).offset(offset).limit(limit).all()
    
    return [memory_to_response(m) for m in memories]


@router.post("/batch", response_model=MemoryBatchResponse)
async def create_memories_batch(
    batch_data: MemoryBatchCreate,
    db: Session = Depends(get_db)
) -> dict:
    """Create multiple memories in batch"""
    # Validate session exists
    from ...models import MemorySession
    session = db.query(MemorySession).filter_by(id=batch_data.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    created_memories = []
    failed_count = 0
    errors = []
    
    for memory_data in batch_data.memories:
        try:
            memory = Memory(
                session_id=batch_data.session_id,
                content=memory_data.content,
                summary=memory_data.summary,
                memory_type=memory_data.memory_type.value,
                importance=memory_data.importance,
                confidence=memory_data.confidence,
                entities=memory_data.entities or [],
                memory_metadata=memory_data.metadata or {}
            )
            db.add(memory)
            created_memories.append(memory)
            
        except Exception as e:
            failed_count += 1
            errors.append({
                "content": memory_data.content[:50] + "...",
                "error": str(e)
            })
            logger.error(f"Failed to create memory in batch: {e}")
    
    try:
        db.commit()
        # Refresh all created memories
        for memory in created_memories:
            db.refresh(memory)
        
        logger.info(f"Batch created {len(created_memories)} memories for session {session.id}")
        
        return MemoryBatchResponse(
            created=len(created_memories),
            failed=failed_count,
            memories=[memory_to_response(m) for m in created_memories],
            errors=errors if errors else None
        )
        
    except Exception as e:
        logger.error(f"Failed to commit batch: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create memories batch")


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(
    search_request: MemorySearchRequest,
    db: Session = Depends(get_db)
) -> dict:
    """Search memories with hybrid vector/keyword search
    
    This endpoint provides advanced search capabilities:
    - Vector similarity search for semantic relevance
    - Keyword search for exact matches
    - Hybrid search combining both approaches
    - Filtering by user, project, memory type, and importance
    """
    try:
        # Use the integrated memory search service
        from ...services.memory_search_service import memory_search_service
        
        result = await memory_search_service.search_memories(db, search_request)
        
        logger.info(
            f"Memory search completed: query='{search_request.query}', "
            f"results={result.total}, time={result.search_time_ms}ms"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        # Fallback to basic search if integrated search fails
        return await _fallback_memory_search(db, search_request)


async def _fallback_memory_search(
    db: Session, 
    search_request: MemorySearchRequest
) -> MemorySearchResponse:
    """Fallback to basic database search if vector search fails"""
    from ...models import MemorySession
    
    # Base query
    query = db.query(Memory).join(MemorySession)
    
    # Apply filters
    if search_request.user_id:
        query = query.filter(MemorySession.user_id == search_request.user_id)
    
    if search_request.project_id:
        query = query.filter(MemorySession.project_id == search_request.project_id)
    
    if search_request.memory_types:
        type_values = [t.value for t in search_request.memory_types]
        query = query.filter(Memory.memory_type.in_(type_values))
    
    if search_request.min_importance > 0:
        query = query.filter(Memory.importance >= search_request.min_importance)
    
    # Text search (simple ILIKE)
    if search_request.query:
        search_term = f"%{search_request.query}%"
        query = query.filter(
            or_(
                Memory.content.ilike(search_term),
                Memory.summary.ilike(search_term)
            )
        )
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering
    memories = query.order_by(
        desc(Memory.importance),
        desc(Memory.created_at)
    ).offset(search_request.offset).limit(search_request.limit).all()
    
    return MemorySearchResponse(
        results=[memory_to_response(m) for m in memories],
        total=total,
        query=search_request.query,
        limit=search_request.limit,
        offset=search_request.offset,
        search_time_ms=0  # No timing for fallback search
    )