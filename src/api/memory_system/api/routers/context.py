"""Context injection API router for Claude-Code integration"""

import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ....models import get_db
from ..context_schemas import (
    ContextRequest, ContextResponse, ContextStats, ContextUpdateRequest,
    ContextTypeEnum
)
from ...services.context_service import context_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/retrieve", response_model=ContextResponse)
async def retrieve_context(
    request: ContextRequest,
    db: Session = Depends(get_db)
):
    """
    Retrieve relevant context for Claude-Code based on user, session, and query.
    
    This endpoint provides the main interface for Claude-Code to get relevant
    memories formatted for LLM consumption.
    """
    try:
        context_response = await context_service.retrieve_context(db, request)
        
        logger.info(
            f"Retrieved context for user {request.user_id}: "
            f"{context_response.total_memories} memories, "
            f"{context_response.total_tokens} tokens, "
            f"max relevance {context_response.max_relevance:.3f}"
        )
        
        return context_response
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve context"
        )


@router.get("/quick/{user_id}", response_model=ContextResponse)
async def get_quick_context(
    user_id: str,
    query: Optional[str] = Query(None, description="Query for context"),
    session_id: Optional[UUID] = Query(None, description="Current session"),
    max_memories: int = Query(10, ge=1, le=50, description="Max memories"),
    max_tokens: int = Query(2000, ge=100, le=8000, description="Max tokens"),
    db: Session = Depends(get_db)
):
    """
    Quick context retrieval with sensible defaults for Claude-Code.
    
    This is a simplified endpoint for common use cases where you just need
    recent and similar context quickly.
    """
    try:
        request = ContextRequest(
            user_id=user_id,
            session_id=session_id,
            query=query,
            context_types=[ContextTypeEnum.recent, ContextTypeEnum.similar],
            max_memories=max_memories,
            max_tokens=max_tokens,
            min_relevance=0.3,
            time_window_hours=24  # Last 24 hours
        )
        
        return await context_service.retrieve_context(db, request)
        
    except Exception as e:
        logger.error(f"Quick context retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve quick context"
        )


@router.post("/comprehensive", response_model=ContextResponse)
async def get_comprehensive_context(
    user_id: str,
    query: str,
    session_id: Optional[UUID] = None,
    project_id: Optional[UUID] = None,
    max_tokens: int = 6000,
    db: Session = Depends(get_db)
):
    """
    Comprehensive context retrieval including all context types.
    
    This endpoint retrieves context from all available sources for complex
    queries that need deep context understanding.
    """
    try:
        request = ContextRequest(
            user_id=user_id,
            session_id=session_id,
            query=query,
            project_id=project_id,
            context_types=[
                ContextTypeEnum.recent,
                ContextTypeEnum.similar,
                ContextTypeEnum.entities,
                ContextTypeEnum.decisions,
                ContextTypeEnum.errors,
                ContextTypeEnum.patterns,
                ContextTypeEnum.preferences
            ],
            max_memories=30,
            max_tokens=max_tokens,
            min_relevance=0.2,  # Lower threshold for comprehensive search
            time_window_hours=168  # Last week
        )
        
        return await context_service.retrieve_context(db, request)
        
    except Exception as e:
        logger.error(f"Comprehensive context retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve comprehensive context"
        )


@router.post("/feedback")
async def provide_context_feedback(
    feedback_request: ContextUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Provide feedback on context effectiveness for future improvements.
    
    This allows Claude-Code to report back on how useful the provided
    context was, enabling the system to learn and improve.
    """
    try:
        # Update access counts for used memories
        from ...models import Memory
        
        for memory_id in feedback_request.memory_ids:
            memory = db.query(Memory).filter_by(id=memory_id).first()
            if memory:
                memory.access_count += 1
                memory.last_accessed = memory.updated_at
        
        db.commit()
        
        logger.info(
            f"Updated {len(feedback_request.memory_ids)} memories with "
            f"effectiveness score {feedback_request.effectiveness_score}"
        )
        
        return {
            "message": "Feedback recorded successfully",
            "memories_updated": len(feedback_request.memory_ids),
            "effectiveness_score": feedback_request.effectiveness_score
        }
        
    except Exception as e:
        logger.error(f"Context feedback failed: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to record context feedback"
        )


@router.get("/stats/{user_id}")
async def get_context_stats(
    user_id: str,
    days: int = Query(7, ge=1, le=30, description="Days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get context retrieval statistics for a user.
    
    Provides insights into context usage patterns and effectiveness.
    """
    try:
        from ...models import Memory, MemorySession
        from sqlalchemy import func
        from datetime import datetime, timedelta, timezone
        
        # Calculate cutoff date
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get statistics
        query = db.query(Memory).join(MemorySession).filter(
            MemorySession.user_id == user_id,
            Memory.created_at >= cutoff
        )
        
        total_memories = query.count()
        accessed_memories = query.filter(Memory.access_count > 0).count()
        avg_importance = query.with_entities(func.avg(Memory.importance)).scalar() or 0
        avg_confidence = query.with_entities(func.avg(Memory.confidence)).scalar() or 0
        
        # Memory type distribution
        type_counts = db.query(
            Memory.memory_type,
            func.count(Memory.id)
        ).join(MemorySession).filter(
            MemorySession.user_id == user_id,
            Memory.created_at >= cutoff
        ).group_by(Memory.memory_type).all()
        
        stats = {
            "user_id": user_id,
            "analysis_period_days": days,
            "total_memories": total_memories,
            "accessed_memories": accessed_memories,
            "access_rate": accessed_memories / total_memories if total_memories > 0 else 0,
            "average_importance": float(avg_importance),
            "average_confidence": float(avg_confidence),
            "memory_type_distribution": {
                memory_type: count for memory_type, count in type_counts
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Context stats retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve context statistics"
        )


@router.get("/health")
async def context_health_check():
    """Health check for context injection service"""
    return {
        "status": "healthy",
        "service": "context_injection",
        "features": [
            "context_retrieval",
            "relevance_scoring", 
            "llm_formatting",
            "token_optimization",
            "feedback_collection"
        ]
    }