"""Enhanced session lifecycle API endpoints

Provides advanced session management capabilities including:
- Session state transitions
- Session insights and analytics
- Automatic session linking
- Session health monitoring
"""

import logging
from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ....models import get_db
from ...core.session_manager import SessionManager
from ...core.session_lifecycle import SessionLifecycleManager
from ..schemas import SessionResponse
from .session import session_to_response

logger = logging.getLogger(__name__)
router = APIRouter()


class SessionStartRequest(BaseModel):
    """Request for starting a session with enhanced options"""
    user_id: str = Field(..., description="User identifier")
    project_id: Optional[UUID] = Field(None, description="Associated project")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    auto_link: bool = Field(True, description="Auto-link to recent sessions")
    context_window: int = Field(50, description="Number of memories to include in context")
    

class SessionEndRequest(BaseModel):
    """Request for ending a session"""
    reason: str = Field("normal", description="Reason for ending (normal, timeout, error, user_requested)")
    final_summary: Optional[str] = Field(None, description="Final session summary")
    

class SessionStateResponse(BaseModel):
    """Session state and health information"""
    session_id: UUID
    state: str
    is_active: bool
    health_status: str
    memory_count: int
    duration_minutes: float
    last_activity: datetime
    warnings: list[str] = Field(default_factory=list)
    

class SessionInsightsResponse(BaseModel):
    """Session insights and analytics"""
    session_id: UUID
    duration_minutes: float
    memory_count: int
    important_memories: int
    topics: list[str]
    key_decisions: list[str]
    errors_encountered: list[str]
    learning_points: list[str]
    action_items: list[str]
    entities: list[str]
    summary: str


@router.post("/lifecycle/start", response_model=SessionResponse)
async def start_enhanced_session(
    request: SessionStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> SessionResponse:
    """Start a new session with enhanced lifecycle management
    
    This endpoint provides advanced session initialization with:
    - Automatic linking to recent sessions
    - Context preservation from parent sessions
    - Background task scheduling for cleanup
    - Session state initialization
    """
    manager = SessionManager(db)
    
    try:
        # Create session through lifecycle manager
        from ..schemas import SessionCreate
        session_data = SessionCreate(
            user_id=request.user_id,
            project_id=request.project_id,
            metadata=request.metadata,
            tags=request.tags
        )
        
        # Set context window preference
        session_data.metadata['context_window'] = request.context_window
        session_data.metadata['auto_link'] = request.auto_link
        
        session = await manager.create_session(session_data)
        
        # Schedule background monitoring
        background_tasks.add_task(
            monitor_session_health,
            session.id,
            db
        )
        
        return session_to_response(session)
        
    except Exception as e:
        logger.error(f"Failed to start enhanced session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lifecycle/{session_id}/end", response_model=SessionResponse)
async def end_enhanced_session(
    session_id: UUID,
    request: SessionEndRequest,
    db: Session = Depends(get_db)
) -> SessionResponse:
    """End a session with full cleanup and insights extraction
    
    This endpoint handles:
    - Graceful session termination
    - Insights and summary extraction
    - Cleanup of temporary data
    - Analytics event triggering
    """
    manager = SessionManager(db)
    
    try:
        # End session through lifecycle manager
        lifecycle = SessionLifecycleManager(db)
        session = await lifecycle.end_session(
            session_id,
            reason=request.reason,
            final_summary=request.final_summary
        )
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_to_response(session)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lifecycle/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(
    session_id: UUID,
    db: Session = Depends(get_db)
) -> SessionStateResponse:
    """Get current session state and health status
    
    Returns detailed information about:
    - Current session state
    - Health indicators
    - Activity metrics
    - Potential warnings
    """
    manager = SessionManager(db)
    session = await manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate health status
    health_status = "healthy"
    warnings = []
    
    # Check for long duration
    if session.is_active and session.duration > 240:  # 4 hours
        health_status = "warning"
        warnings.append("Session has been active for over 4 hours")
    
    # Check for high memory count
    if session.memory_count > 500:
        health_status = "warning"
        warnings.append(f"High memory count: {session.memory_count}")
    
    # Check for inactivity
    last_memory = None
    if session.memories:
        last_memory = max(session.memories, key=lambda m: m.created_at)
    
    last_activity = last_memory.created_at if last_memory else session.started_at
    inactivity_minutes = (datetime.now(timezone.utc) - last_activity).total_seconds() / 60
    
    if session.is_active and inactivity_minutes > 60:
        health_status = "warning"
        warnings.append(f"No activity for {int(inactivity_minutes)} minutes")
    
    return SessionStateResponse(
        session_id=session.id,
        state=session.session_metadata.get('state', 'active' if session.is_active else 'ended'),
        is_active=session.is_active,
        health_status=health_status,
        memory_count=session.memory_count,
        duration_minutes=session.duration,
        last_activity=last_activity,
        warnings=warnings
    )


@router.get("/lifecycle/{session_id}/insights", response_model=SessionInsightsResponse)
async def get_session_insights(
    session_id: UUID,
    db: Session = Depends(get_db)
) -> SessionInsightsResponse:
    """Get insights and analytics for a session
    
    Extracts:
    - Key topics and themes
    - Important decisions made
    - Errors and solutions
    - Learning points
    - Action items
    """
    lifecycle = SessionLifecycleManager(db)
    session = db.query(MemorySession).filter_by(id=session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Extract insights
    insights = await lifecycle._extract_session_insights(session)
    
    # Get summary
    summary = session.session_metadata.get(
        'summary', 
        await lifecycle._generate_session_summary(session)
    )
    
    return SessionInsightsResponse(
        session_id=session.id,
        duration_minutes=session.duration,
        memory_count=session.memory_count,
        important_memories=len(session.important_memories),
        topics=insights.get('topics', []),
        key_decisions=insights.get('key_decisions', []),
        errors_encountered=insights.get('errors_encountered', []),
        learning_points=insights.get('learning_points', []),
        action_items=insights.get('action_items', []),
        entities=lifecycle._extract_unique_entities(session),
        summary=summary
    )


@router.post("/lifecycle/{session_id}/transition")
async def transition_session_state(
    session_id: UUID,
    new_state: str = Query(..., description="New state (paused, resumed, archived)"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Transition session to a new state
    
    Supported transitions:
    - active -> paused: Temporarily pause session
    - paused -> active: Resume paused session  
    - ended -> archived: Archive completed session
    """
    manager = SessionManager(db)
    session = await manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    current_state = session.session_metadata.get('state', 'active' if session.is_active else 'ended')
    
    # Validate transition
    valid_transitions = {
        'active': ['paused', 'ended'],
        'paused': ['active', 'ended'],
        'ended': ['archived']
    }
    
    if current_state not in valid_transitions:
        raise HTTPException(status_code=400, detail=f"Unknown current state: {current_state}")
    
    if new_state not in valid_transitions[current_state]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid transition from {current_state} to {new_state}"
        )
    
    # Apply transition
    session.add_metadata('state', new_state)
    session.add_metadata(f'{new_state}_at', datetime.now(timezone.utc).isoformat())
    
    # Handle specific transitions
    if new_state == 'paused':
        session.add_metadata('paused_reason', 'user_requested')
    elif new_state == 'active' and current_state == 'paused':
        session.add_metadata('resumed_at', datetime.now(timezone.utc).isoformat())
    elif new_state == 'archived':
        session.add_tag('archived')
    
    db.commit()
    
    return {
        "session_id": str(session_id),
        "previous_state": current_state,
        "new_state": new_state,
        "transitioned_at": datetime.now(timezone.utc).isoformat()
    }


@router.get("/lifecycle/health")
async def session_lifecycle_health() -> Dict[str, Any]:
    """Health check for session lifecycle management"""
    return {
        "status": "healthy",
        "service": "session_lifecycle",
        "features": [
            "enhanced_session_start",
            "graceful_session_end",
            "session_state_monitoring",
            "insights_extraction",
            "state_transitions",
            "automatic_cleanup"
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Background task functions
async def monitor_session_health(session_id: UUID, db: Session):
    """Background task to monitor session health"""
    import asyncio
    
    # Wait 30 minutes then check
    await asyncio.sleep(30 * 60)
    
    try:
        from ...models import MemorySession
        session = db.query(MemorySession).filter_by(id=session_id).first()
        
        if session and session.is_active:
            # Check for inactivity
            last_memory = None
            if session.memories:
                last_memory = max(session.memories, key=lambda m: m.created_at)
            
            last_activity = last_memory.created_at if last_memory else session.started_at
            inactivity_minutes = (datetime.now(timezone.utc) - last_activity).total_seconds() / 60
            
            if inactivity_minutes > 60:
                logger.warning(f"Session {session_id} inactive for {int(inactivity_minutes)} minutes")
                session.add_metadata('health_warning', f'Inactive for {int(inactivity_minutes)} minutes')
                db.commit()
                
    except Exception as e:
        logger.error(f"Error monitoring session health: {e}")


# Import required model
from ...models import MemorySession
