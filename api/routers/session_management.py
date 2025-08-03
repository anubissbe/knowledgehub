"""
Session Management API Router.

This module provides REST API endpoints for AI session management including
session creation, state management, handoffs, and recovery operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session as DBSession

from ..models.base import get_db
from ..models.session import (
    SessionCreate, SessionUpdate, SessionHandoffCreate, SessionCheckpointCreate,
    SessionResponse, SessionContextResponse, SessionAnalytics, SessionRecoveryInfo,
    SessionState, SessionType, HandoffReason
)
from ..services.session_service import session_service
from shared.logging import setup_logging

logger = setup_logging("session_router")

router = APIRouter(prefix="/api/sessions", tags=["session-management"])


@router.post("/", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    auto_restore: bool = Query(True, description="Automatically restore context from related sessions"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Create a new AI session with intelligent initialization.
    
    Features:
    - Automatic context restoration from related sessions
    - Parent session linking
    - Initial checkpoint creation
    - Performance tracking
    """
    try:
        session = await session_service.create_session(
            session_data=session_data,
            auto_restore=auto_restore
        )
        
        # Schedule background optimization
        background_tasks.add_task(_optimize_session_performance, session.id)
        
        logger.info(f"Created session {session.id} for user {session.user_id}")
        return session
        
    except ValueError as e:
        logger.error(f"Session creation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str = Path(..., description="Session ID")
):
    """
    Get session details by ID.
    
    Automatically updates activity tracking when accessed.
    """
    try:
        session = await session_service.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    update_data: SessionUpdate
):
    """
    Update session with new data.
    
    Handles state transitions and triggers appropriate workflows.
    """
    try:
        session = await session_service.update_session(
            session_id=session_id,
            update_data=update_data
        )
        
        logger.info(f"Updated session {session_id}")
        return session
        
    except ValueError as e:
        logger.error(f"Session update validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Session update failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update session")


@router.post("/{session_id}/context", response_model=SessionContextResponse)
async def add_to_context(
    session_id: str,
    memory_id: str = Query(..., description="Memory ID to add to context"),
    auto_optimize: bool = Query(True, description="Automatically optimize context window size")
):
    """
    Add memory to session context window.
    
    Features:
    - Automatic context window optimization
    - Context summary updates
    - Performance monitoring
    """
    try:
        context = await session_service.add_to_context(
            session_id=session_id,
            memory_id=memory_id,
            auto_optimize=auto_optimize
        )
        
        logger.debug(f"Added memory {memory_id} to session {session_id} context")
        return context
        
    except ValueError as e:
        logger.error(f"Context addition validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add to context: {e}")
        raise HTTPException(status_code=500, detail="Failed to add memory to context")


@router.get("/{session_id}/context", response_model=SessionContextResponse)
async def get_session_context(
    session_id: str = Path(..., description="Session ID")
):
    """Get current session context window."""
    try:
        session = await session_service.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionContextResponse(
            session_id=session_id,
            context_window=[],  # Would get from session
            context_summary=session.description,
            context_size=session.context_size,
            max_context_size=session.max_context_size,
            last_updated=session.last_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session context: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session context")


@router.post("/handoffs", response_model=Dict[str, str])
async def create_handoff(
    handoff_data: SessionHandoffCreate,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Create a session handoff for context transfer.
    
    Features:
    - Complete state snapshot creation
    - Context preservation
    - Automatic session state transitions
    - Background cleanup
    """
    try:
        handoff_id = await session_service.create_handoff(handoff_data)
        
        # Schedule background cleanup
        background_tasks.add_task(_cleanup_handoff_session, handoff_data.source_session_id)
        
        logger.info(f"Created handoff {handoff_id}")
        return {"handoff_id": handoff_id, "status": "created"}
        
    except ValueError as e:
        logger.error(f"Handoff creation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Handoff creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create handoff")


@router.post("/handoffs/{handoff_id}/restore", response_model=SessionResponse)
async def restore_from_handoff(
    handoff_id: str,
    new_session_data: SessionCreate
):
    """
    Create new session from handoff context.
    
    Restores complete session state including:
    - Context window
    - Session variables
    - Active tasks
    - Task queue
    """
    try:
        session = await session_service.restore_from_handoff(
            handoff_id=handoff_id,
            new_session_data=new_session_data
        )
        
        logger.info(f"Restored session {session.id} from handoff {handoff_id}")
        return session
        
    except ValueError as e:
        logger.error(f"Handoff restoration validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Handoff restoration failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to restore from handoff")


@router.post("/checkpoints", response_model=Dict[str, str])
async def create_checkpoint(
    checkpoint_data: SessionCheckpointCreate
):
    """
    Create a manual session checkpoint.
    
    Creates a complete state snapshot for recovery purposes.
    """
    try:
        checkpoint_id = await session_service.create_checkpoint(checkpoint_data)
        
        logger.info(f"Created checkpoint {checkpoint_id}")
        return {"checkpoint_id": checkpoint_id, "status": "created"}
        
    except ValueError as e:
        logger.error(f"Checkpoint creation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Checkpoint creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkpoint")


@router.get("/{session_id}/recovery", response_model=SessionRecoveryInfo)
async def get_recovery_info(
    session_id: str = Path(..., description="Session ID")
):
    """
    Get session recovery information.
    
    Provides:
    - Available recovery checkpoints
    - Estimated data loss
    - Recommended recovery actions
    """
    try:
        recovery_info = await session_service.get_recovery_info(session_id)
        
        return recovery_info
        
    except ValueError as e:
        logger.error(f"Recovery info validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get recovery info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recovery information")


@router.get("/analytics/{user_id}", response_model=SessionAnalytics)
async def get_session_analytics(
    user_id: str = Path(..., description="User ID"),
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    time_window_hours: int = Query(24, description="Time window in hours", ge=1, le=8760)
):
    """
    Get comprehensive session analytics.
    
    Provides:
    - Session counts and metrics
    - Performance trends
    - User satisfaction data
    - Handoff statistics
    """
    try:
        analytics = await session_service.get_session_analytics(
            user_id=user_id,
            project_id=project_id,
            time_window_hours=time_window_hours
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get session analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session analytics")


@router.get("/", response_model=List[SessionResponse])
async def list_sessions(
    user_id: str = Query(..., description="User ID"),
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    state: Optional[SessionState] = Query(None, description="Session state filter"),
    session_type: Optional[SessionType] = Query(None, description="Session type filter"),
    limit: int = Query(50, description="Maximum number of sessions", ge=1, le=200),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    db: DBSession = Depends(get_db)
):
    """
    List sessions with filtering and pagination.
    """
    try:
        from ..models.session import Session
        
        # Build query
        query = db.query(Session).filter(Session.user_id == user_id)
        
        if project_id:
            query = query.filter(Session.project_id == project_id)
        if state:
            query = query.filter(Session.state == state.value)
        if session_type:
            query = query.filter(Session.session_type == session_type.value)
        
        # Apply pagination and ordering
        sessions = query.order_by(Session.started_at.desc()).offset(offset).limit(limit).all()
        
        # Convert to response format
        session_responses = []
        for session in sessions:
            response = session_service._session_to_response(session)
            session_responses.append(response)
        
        logger.info(f"Listed {len(session_responses)} sessions for user {user_id}")
        return session_responses
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")


@router.delete("/{session_id}")
async def delete_session(
    session_id: str = Path(..., description="Session ID"),
    force: bool = Query(False, description="Force delete even if active"),
    db: DBSession = Depends(get_db)
):
    """
    Delete a session and its associated data.
    
    WARNING: This operation cannot be undone.
    """
    try:
        from ..models.session import Session
        
        session = db.query(Session).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if session is active
        if session.state == SessionState.ACTIVE.value and not force:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete active session. Use force=true to override."
            )
        
        # Delete session (cascading deletes will handle related records)
        db.delete(session)
        db.commit()
        
        logger.info(f"Deleted session {session_id}")
        return {"status": "deleted", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.post("/{session_id}/actions/pause")
async def pause_session(
    session_id: str = Path(..., description="Session ID"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Pause an active session."""
    try:
        session = await session_service.update_session(
            session_id=session_id,
            update_data=SessionUpdate(state=SessionState.PAUSED)
        )
        
        # Create pause checkpoint
        background_tasks.add_task(_create_pause_checkpoint, session_id)
        
        logger.info(f"Paused session {session_id}")
        return {"status": "paused", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Failed to pause session: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause session")


@router.post("/{session_id}/actions/resume")
async def resume_session(
    session_id: str = Path(..., description="Session ID")
):
    """Resume a paused session."""
    try:
        session = await session_service.update_session(
            session_id=session_id,
            update_data=SessionUpdate(state=SessionState.ACTIVE)
        )
        
        logger.info(f"Resumed session {session_id}")
        return {"status": "resumed", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Failed to resume session: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume session")


@router.post("/{session_id}/actions/complete")
async def complete_session(
    session_id: str = Path(..., description="Session ID"),
    completion_status: str = Query("completed", description="Completion status"),
    user_satisfaction: Optional[float] = Query(None, description="User satisfaction (0-1)", ge=0.0, le=1.0),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Mark session as completed."""
    try:
        update_data = SessionUpdate(
            state=SessionState.COMPLETED,
            completion_status=completion_status
        )
        
        if user_satisfaction is not None:
            update_data.user_satisfaction = user_satisfaction
        
        session = await session_service.update_session(
            session_id=session_id,
            update_data=update_data
        )
        
        # Create completion checkpoint
        background_tasks.add_task(_create_completion_checkpoint, session_id, completion_status)
        
        logger.info(f"Completed session {session_id} with status: {completion_status}")
        return {
            "status": "completed", 
            "session_id": session_id,
            "completion_status": completion_status
        }
        
    except Exception as e:
        logger.error(f"Failed to complete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete session")


# Background task functions

async def _optimize_session_performance(session_id: str):
    """Background task to optimize session performance."""
    try:
        # This could include:
        # - Context window optimization
        # - Memory cleanup
        # - Performance analytics
        logger.debug(f"Optimizing performance for session {session_id}")
        
    except Exception as e:
        logger.warning(f"Session optimization failed for {session_id}: {e}")


async def _cleanup_handoff_session(session_id: str):
    """Background task to clean up after handoff."""
    try:
        # Archive session data, clean up temporary resources
        logger.debug(f"Cleaning up handoff session {session_id}")
        
    except Exception as e:
        logger.warning(f"Handoff cleanup failed for {session_id}: {e}")


async def _create_pause_checkpoint(session_id: str):
    """Background task to create checkpoint when pausing."""
    try:
        checkpoint_data = SessionCheckpointCreate(
            session_id=session_id,
            checkpoint_name="Session Paused",
            description="Automatic checkpoint created when session was paused",
            checkpoint_type="auto",
            is_recovery_point=True,
            recovery_priority=5
        )
        
        await session_service.create_checkpoint(checkpoint_data)
        
    except Exception as e:
        logger.warning(f"Pause checkpoint creation failed for {session_id}: {e}")


async def _create_completion_checkpoint(session_id: str, completion_status: str):
    """Background task to create checkpoint when completing session."""
    try:
        checkpoint_data = SessionCheckpointCreate(
            session_id=session_id,
            checkpoint_name="Session Completed",
            description=f"Final checkpoint: {completion_status}",
            checkpoint_type="auto",
            is_recovery_point=True,
            recovery_priority=8
        )
        
        await session_service.create_checkpoint(checkpoint_data)
        
    except Exception as e:
        logger.warning(f"Completion checkpoint creation failed for {session_id}: {e}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Session management service health check."""
    try:
        # Basic service health check
        if not session_service._initialized:
            await session_service.initialize()
        
        return {
            "status": "healthy",
            "service": "session_management",
            "initialized": session_service._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "session_management",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )