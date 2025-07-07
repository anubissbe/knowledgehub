"""Session management API router"""

import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ....api.dependencies import get_db
from ...models import MemorySession
from ...core.session_manager import SessionManager
from ..schemas import (
    SessionCreate, SessionUpdate, SessionResponse, SessionSummary
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/start", response_model=SessionResponse)
async def start_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db)
) -> MemorySession:
    """Start a new Claude-Code session"""
    manager = SessionManager(db)
    session = await manager.create_session(session_data)
    return session


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    db: Session = Depends(get_db)
) -> MemorySession:
    """Get session details"""
    manager = SessionManager(db)
    session = await manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: UUID,
    update_data: SessionUpdate,
    db: Session = Depends(get_db)
) -> MemorySession:
    """Update session metadata"""
    manager = SessionManager(db)
    session = await manager.update_session(session_id, update_data)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.post("/{session_id}/end", response_model=SessionResponse)
async def end_session(
    session_id: UUID,
    db: Session = Depends(get_db)
) -> MemorySession:
    """End a session"""
    manager = SessionManager(db)
    session = await manager.end_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.get("/user/{user_id}", response_model=List[SessionSummary])
async def get_user_sessions(
    user_id: str,
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    active_only: bool = Query(False, description="Only active sessions"),
    limit: int = Query(10, gt=0, le=50, description="Maximum results"),
    db: Session = Depends(get_db)
) -> List[MemorySession]:
    """Get sessions for a user"""
    manager = SessionManager(db)
    sessions = await manager.get_user_sessions(
        user_id=user_id,
        project_id=project_id,
        active_only=active_only,
        limit=limit
    )
    
    # Convert to summary format
    summaries = []
    for session in sessions:
        summaries.append(SessionSummary(
            id=session.id,
            user_id=session.user_id,
            started_at=session.started_at,
            duration=session.duration,
            memory_count=session.memory_count,
            is_active=session.is_active,
            tags=session.tags or []
        ))
    
    return summaries


@router.get("/{session_id}/chain", response_model=List[SessionResponse])
async def get_session_chain(
    session_id: UUID,
    db: Session = Depends(get_db)
) -> List[MemorySession]:
    """Get all sessions in a conversation chain"""
    manager = SessionManager(db)
    chain = await manager.get_session_chain(session_id)
    
    if not chain:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chain


@router.post("/cleanup")
async def cleanup_stale_sessions(
    hours: int = Query(24, gt=0, description="Hours of inactivity"),
    db: Session = Depends(get_db)
) -> dict:
    """Clean up stale sessions"""
    manager = SessionManager(db)
    closed_count = await manager.cleanup_stale_sessions(hours)
    
    return {
        "message": f"Closed {closed_count} stale sessions",
        "closed_count": closed_count
    }


@router.get("/{session_id}/context", response_model=dict)
async def get_session_context(
    session_id: UUID,
    db: Session = Depends(get_db)
) -> dict:
    """Get session context summary"""
    manager = SessionManager(db)
    session = await manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.get_context_summary()