"""
Working Claude Code Enhancement API - bypasses dependency injection issues
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import asyncio

from ..models import get_db
from ..models.base import SessionLocal
from ..services.claude_simple import ClaudeEnhancementService

router = APIRouter(prefix="/api/claude-working", tags=["claude-working"])


async def get_service():
    """Get service without dependency injection"""
    db = SessionLocal()
    return ClaudeEnhancementService(db), db


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "claude-working"}


@router.post("/initialize")
async def initialize_claude(
    cwd: str = Query(..., description="Current working directory")
) -> Dict[str, Any]:
    """Initialize Claude with all enhancements"""
    service, db = await get_service()
    try:
        result = await service.initialize_claude(cwd)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/session/continue")
async def continue_session(
    previous_session_id: str = Query(..., description="Previous session ID")
) -> Dict[str, Any]:
    """Continue from a previous session"""
    service, db = await get_service()
    try:
        result = await service.continue_session(previous_session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/session/handoff")
async def create_handoff(
    session_id: str = Query(..., description="Current session ID"),
    content: str = Query(..., description="Handoff content"),
    next_tasks: Optional[List[str]] = Query(None, description="Next tasks")
) -> Dict[str, Any]:
    """Create handoff note for next session"""
    service, db = await get_service()
    try:
        memory = await service.create_handoff_note(session_id, content, next_tasks)
        return {
            "id": str(memory.id),
            "content": memory.content,
            "created": memory.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/project/detect")
async def detect_project(
    cwd: str = Query(..., description="Project directory")
) -> Dict[str, Any]:
    """Detect and profile project"""
    service, db = await get_service()
    try:
        result = await service.detect_project(cwd)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/project/{project_id}/context")
async def get_project_context(project_id: str) -> Dict[str, Any]:
    """Get project context"""
    service, db = await get_service()
    try:
        result = await service.get_project_context(project_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/error/record")
async def record_error(
    error_type: str = Query(..., description="Error type"),
    error_message: str = Query(..., description="Error message"),
    solution: Optional[str] = Query(None, description="Solution"),
    success: bool = Query(False, description="Was solution successful"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    project_id: Optional[str] = Query(None, description="Project ID")
) -> Dict[str, Any]:
    """Record an error and solution"""
    service, db = await get_service()
    try:
        memory = await service.record_error(
            error_type, error_message, solution, success, session_id, project_id
        )
        return {
            "id": str(memory.id),
            "content": memory.content,
            "success": success
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/error/similar")
async def find_similar_errors(
    error_type: str = Query(..., description="Error type"),
    error_message: str = Query(..., description="Error message")
) -> List[Dict[str, Any]]:
    """Find similar errors"""
    service, db = await get_service()
    try:
        result = await service.find_similar_errors(error_type, error_message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/task/predict")
async def predict_tasks(
    session_id: str = Query(..., description="Session ID"),
    project_id: Optional[str] = Query(None, description="Project ID")
) -> List[Dict[str, Any]]:
    """Predict next tasks"""
    service, db = await get_service()
    try:
        result = await service.predict_next_tasks(session_id, project_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()