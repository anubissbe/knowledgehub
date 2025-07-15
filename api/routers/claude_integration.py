"""
Claude Code Integration API Router
Provides endpoints for bidirectional memory sync and real-time context sharing
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.claude_code_integration import (
    ClaudeCodeIntegration,
    MemorySyncRequest,
    ContextUpdateRequest,
    get_claude_integration
)
from ..services.realtime_learning_pipeline import get_learning_pipeline

router = APIRouter(prefix="/api/claude-integration", tags=["claude-integration"])


class SessionInitRequest(BaseModel):
    """Request to initialize a Claude session"""
    workspace_path: str
    user_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HandoffRequest(BaseModel):
    """Request to create session handoff"""
    session_id: str
    notes: str
    next_tasks: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextStreamRequest(BaseModel):
    """Request for real-time context stream"""
    session_id: str
    include_linked: bool = True
    stream_types: List[str] = Field(default_factory=lambda: ["all"])


@router.get("/health")
async def health_check(
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
):
    """Check Claude integration health"""
    return {
        "status": "healthy",
        "service": "claude-integration",
        "active_sessions": len(integration.active_sessions),
        "session_links": len(integration.session_links)
    }


@router.post("/session/init")
async def initialize_session(
    request: SessionInitRequest,
    db: Session = Depends(get_db),
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
) -> Dict[str, Any]:
    """
    Initialize a new Claude Code session with context restoration
    
    This endpoint:
    - Creates a new session with unique ID
    - Restores relevant memories from previous sessions
    - Links to parent session for bidirectional sync
    - Sets up real-time context streaming
    """
    try:
        result = await integration.initialize_session(
            workspace_path=request.workspace_path,
            user_id=request.user_id,
            parent_session_id=request.parent_session_id,
            db=db
        )
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/sync")
async def sync_memories(
    sync_request: MemorySyncRequest,
    db: Session = Depends(get_db),
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
) -> Dict[str, Any]:
    """
    Synchronize memories between Claude sessions
    
    Sync types:
    - full: Complete memory sync
    - incremental: Only new memories since last sync
    - selective: Filtered sync based on criteria
    """
    try:
        result = await integration.sync_memories(sync_request, db)
        
        return {
            "success": True,
            **result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to sync memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/update")
async def update_context(
    update_request: ContextUpdateRequest,
    background_tasks: BackgroundTasks,
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
) -> Dict[str, Any]:
    """
    Update Claude's context in real-time
    
    Update types:
    - file_change: File modifications
    - task_update: Task status changes
    - error_occurred: Error tracking
    - decision_made: Decision recording
    """
    try:
        result = await integration.update_context(update_request)
        
        # Background task to propagate updates
        background_tasks.add_task(
            integration._propagate_context_update,
            update_request.session_id,
            update_request
        )
        
        return {
            "success": True,
            **result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/{session_id}")
async def get_context(
    session_id: str,
    include_linked: bool = True,
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
) -> Dict[str, Any]:
    """Get unified context for a Claude session"""
    try:
        context = await integration.get_unified_context(
            session_id=session_id,
            include_linked=include_linked
        )
        
        return {
            "success": True,
            "context": context
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/handoff")
async def create_handoff(
    request: HandoffRequest,
    db: Session = Depends(get_db),
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
) -> Dict[str, Any]:
    """
    Create a handoff for the next Claude session
    
    This prepares everything the next instance needs:
    - Current context summary
    - Active tasks and their status
    - Recent patterns and decisions
    - Error history and learnings
    """
    try:
        result = await integration.handoff_session(
            current_session_id=request.session_id,
            notes=request.notes,
            next_tasks=request.next_tasks,
            db=db
        )
        
        return {
            "success": True,
            **result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create handoff: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/active")
async def get_active_sessions(
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
) -> Dict[str, Any]:
    """Get all active Claude sessions"""
    
    sessions = []
    for session_id, context in integration.active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "workspace": context.workspace_path,
            "started": context.timestamp.isoformat(),
            "summary": integration._summarize_context(context),
            "linked_to": list(integration.session_links.get(session_id, set()))
        })
        
    return {
        "active_sessions": sessions,
        "total": len(sessions)
    }


@router.post("/sessions/link")
async def link_sessions(
    session_ids: List[str],
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
) -> Dict[str, Any]:
    """Link multiple Claude sessions for bidirectional sync"""
    
    if len(session_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 session IDs required for linking"
        )
        
    # Verify all sessions exist
    for sid in session_ids:
        if sid not in integration.active_sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {sid} not found"
            )
            
    # Create bidirectional links
    for i, sid1 in enumerate(session_ids):
        for sid2 in session_ids[i+1:]:
            integration.session_links[sid1].add(sid2)
            integration.session_links[sid2].add(sid1)
            
    return {
        "success": True,
        "linked_sessions": session_ids,
        "total_links": sum(len(links) for links in integration.session_links.values()) // 2
    }


@router.delete("/sessions/{session_id}")
async def end_session(
    session_id: str,
    create_handoff: bool = True,
    handoff_notes: str = "Session ended",
    db: Session = Depends(get_db),
    integration: ClaudeCodeIntegration = Depends(get_claude_integration)
) -> Dict[str, Any]:
    """End a Claude session and optionally create handoff"""
    
    if session_id not in integration.active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    result = {"session_id": session_id}
    
    # Create handoff if requested
    if create_handoff:
        handoff_result = await integration.handoff_session(
            current_session_id=session_id,
            notes=handoff_notes,
            next_tasks=[],
            db=db
        )
        result["handoff"] = handoff_result
        
    # Remove from active sessions
    del integration.active_sessions[session_id]
    
    # Clean up links
    for linked_id in list(integration.session_links.get(session_id, set())):
        integration.session_links[linked_id].discard(session_id)
    del integration.session_links[session_id]
    
    # Clean up sync lock
    if session_id in integration.sync_locks:
        del integration.sync_locks[session_id]
        
    result["status"] = "ended"
    return result


@router.get("/insights/{session_id}")
async def get_session_insights(
    session_id: str,
    integration: ClaudeCodeIntegration = Depends(get_claude_integration),
    pipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Get AI insights for a Claude session"""
    
    try:
        # Get unified context
        context = await integration.get_unified_context(session_id)
        
        # Get real-time insights from pipeline
        realtime_context = await pipeline.get_real_time_context(session_id)
        
        insights = {
            "session_id": session_id,
            "workspace_insights": {
                "files_touched": len(context["all_files"]),
                "most_edited": context["all_files"][:5] if context["all_files"] else [],
                "session_duration": context.get("context_age")
            },
            "pattern_insights": {
                "total_patterns": len(context["combined_patterns"]),
                "top_patterns": context["combined_patterns"][:5],
                "pattern_categories": {}
            },
            "error_insights": context["error_insights"],
            "decision_insights": {
                "total_decisions": len(context["decision_history"]),
                "recent_decisions": context["decision_history"][:5]
            },
            "task_insights": {
                "active_tasks": len([t for t in context["active_tasks"] if t.get("status") != "completed"]),
                "completed_tasks": len([t for t in context["active_tasks"] if t.get("status") == "completed"]),
                "task_list": context["active_tasks"]
            },
            "realtime_insights": {
                "recent_patterns": realtime_context.get("recent_patterns", []),
                "recent_insights": realtime_context.get("recent_insights", [])
            }
        }
        
        # Categorize patterns
        for pattern in context["combined_patterns"]:
            ptype = pattern.get("type", "unknown")
            insights["pattern_insights"]["pattern_categories"][ptype] = \
                insights["pattern_insights"]["pattern_categories"].get(ptype, 0) + 1
                
        return insights
        
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import logger
import logging
logger = logging.getLogger(__name__)