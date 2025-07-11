"""
Simplified Claude Code Enhancement API
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.claude_simple import ClaudeEnhancementService


router = APIRouter(prefix="/api/claude", tags=["claude-enhancements"])


@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"status": "ok", "message": "Claude endpoint is working"}


@router.get("/test-detect")
async def test_detect_simple(cwd: str):
    """Test project detection without DB"""
    import hashlib
    from pathlib import Path
    
    project_path = Path(cwd)
    project_id = hashlib.md5(str(project_path).encode()).hexdigest()[:12]
    
    return {
        "id": project_id,
        "path": str(project_path),
        "name": project_path.name,
        "type": "python" if (project_path / "setup.py").exists() else "unknown"
    }


def get_claude_service(db: Session = Depends(get_db)) -> ClaudeEnhancementService:
    return ClaudeEnhancementService(db)


# ========== Main Endpoints ==========

@router.post("/initialize")
async def initialize_claude(
    cwd: str,
    previous_session_id: Optional[str] = None,
    service: ClaudeEnhancementService = Depends(get_claude_service)
) -> Dict[str, Any]:
    """Initialize Claude with all enhancements"""
    try:
        return await service.initialize_claude(cwd, previous_session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/continue")
async def continue_session(
    previous_session_id: str,
    service: ClaudeEnhancementService = Depends(get_claude_service)
) -> Dict[str, Any]:
    """Continue from a previous session"""
    try:
        return await service.continue_session(previous_session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/handoff")
async def create_handoff(
    session_id: str,
    content: str,
    next_tasks: Optional[List[str]] = None,
    service: ClaudeEnhancementService = Depends(get_claude_service)
) -> Dict[str, Any]:
    """Create handoff note for next session"""
    try:
        memory = await service.create_handoff_note(session_id, content, next_tasks)
        return {
            "id": str(memory.id),
            "content": memory.content,
            "created": memory.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project/detect")
async def detect_project(
    cwd: str,
    service: ClaudeEnhancementService = Depends(get_claude_service)
) -> Dict[str, Any]:
    """Detect and profile project"""
    try:
        return await service.detect_project(cwd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/context")
async def get_project_context(
    project_id: str,
    service: ClaudeEnhancementService = Depends(get_claude_service)
) -> Dict[str, Any]:
    """Get project context"""
    try:
        return await service.get_project_context(project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/error/record")
async def record_error(
    error_type: str,
    error_message: str,
    solution: Optional[str] = None,
    success: bool = False,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    service: ClaudeEnhancementService = Depends(get_claude_service)
) -> Dict[str, Any]:
    """Record an error and solution"""
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


@router.get("/error/similar")
async def find_similar_errors(
    error_type: str,
    error_message: str,
    service: ClaudeEnhancementService = Depends(get_claude_service)
) -> List[Dict[str, Any]]:
    """Find similar errors"""
    try:
        return await service.find_similar_errors(error_type, error_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/predict")
async def predict_tasks(
    session_id: str,
    project_id: Optional[str] = None,
    service: ClaudeEnhancementService = Depends(get_claude_service)
) -> List[Dict[str, Any]]:
    """Predict next tasks"""
    try:
        return await service.predict_next_tasks(session_id, project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))