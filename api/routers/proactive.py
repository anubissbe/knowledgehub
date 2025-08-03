"""
Proactive Assistant API - Anticipates needs and provides assistance
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.proactive_assistant import ProactiveAssistant

router = APIRouter(prefix="/api/proactive", tags=["proactive"])

# Global proactive assistant
assistant = ProactiveAssistant()


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "proactive-assistant",
        "description": "Anticipates needs and provides proactive assistance"
    }


@router.post("/analyze")
def analyze_session_post(
    context: Dict[str, Any] = Body(..., description="Context to analyze"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Analyze provided context and provide proactive insights (POST version)
    """
    try:
        # Extract session_id and project_id from context if provided
        session_id = context.get("session_id", "default")
        project_id = context.get("project_id")
        
        analysis = assistant.analyze_session_state(db, session_id, project_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze")
def analyze_session(
    session_id: str = Query(..., description="Current session ID"),
    project_id: Optional[str] = Query(None, description="Project ID for context"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Analyze current session and provide proactive insights
    
    Returns:
    - Work state analysis
    - Incomplete tasks with priorities
    - Unresolved errors with suggestions
    - Predicted next actions
    - Helpful reminders
    - Preloaded relevant context
    """
    try:
        analysis = assistant.analyze_session_state(db, session_id, project_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brief")
def get_session_brief(
    session_id: str = Query(..., description="Current session ID"),
    project_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get a brief proactive summary for session start
    
    Perfect for displaying when Claude Code starts
    """
    try:
        brief = assistant.get_session_brief(db, session_id, project_id)
        return {
            "brief": brief,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/incomplete-tasks")
def get_incomplete_tasks(
    session_id: str = Query(...),
    project_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get list of incomplete tasks with priorities
    
    Sorted by priority and age
    """
    try:
        # Use the analysis method but return just tasks
        analysis = assistant.analyze_session_state(db, session_id, project_id)
        return analysis.get("incomplete_tasks", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions")
def get_predictions(
    session_id: str = Query(...),
    project_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get predicted next actions based on current context
    
    Returns actions with confidence scores
    """
    try:
        analysis = assistant.analyze_session_state(db, session_id, project_id)
        return analysis.get("predictions", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reminders")
def get_reminders(
    session_id: str = Query(...),
    project_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get helpful reminders about work
    
    Includes overdue tasks, repeated errors, checkpoints
    """
    try:
        analysis = assistant.analyze_session_state(db, session_id, project_id)
        return analysis.get("reminders", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-interrupt")
def check_interrupt(
    action: str = Query(..., description="Action about to be performed"),
    context: Dict[str, Any] = Body({}, description="Current context")
) -> Dict[str, Any]:
    """
    Check if assistant should interrupt with help
    
    Use before actions to get proactive warnings
    """
    interrupt = assistant.should_interrupt(action, context)
    
    if interrupt:
        return interrupt
    
    return {
        "interrupt": False,
        "reason": None,
        "message": None
    }


@router.get("/context")
def get_preloaded_context(
    session_id: str = Query(...),
    project_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get preloaded context based on predictions
    
    Returns relevant files, errors, solutions, patterns
    """
    try:
        analysis = assistant.analyze_session_state(db, session_id, project_id)
        return analysis.get("preloaded_context", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggestions")
def get_suggestions_post(
    context: Dict[str, Any] = Body(..., description="Context for suggestions"),
    db: Session = Depends(get_db)
) -> List[str]:
    """
    Get proactive suggestions based on context (POST version)
    """
    try:
        session_id = context.get("session_id", "default")
        project_id = context.get("project_id")
        analysis = assistant.analyze_session_state(db, session_id, project_id)
        suggestions = analysis.get("proactive_suggestions", [])
        return suggestions[:5]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions")
def get_suggestions(
    session_id: str = Query(...),
    project_id: Optional[str] = Query(None),
    limit: int = Query(5, le=10),
    db: Session = Depends(get_db)
) -> List[str]:
    """
    Get proactive suggestions for current situation
    """
    try:
        analysis = assistant.analyze_session_state(db, session_id, project_id)
        suggestions = analysis.get("proactive_suggestions", [])
        return suggestions[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


