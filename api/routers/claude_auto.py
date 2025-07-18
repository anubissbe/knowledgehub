"""
Claude Auto - Automatic session management for Claude Code
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.claude_session_manager import ClaudeSessionManager

router = APIRouter(prefix="/api/claude-auto", tags=["claude-auto"])

# Global session manager instance
session_manager = ClaudeSessionManager()


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "claude-auto", "description": "Automatic session management"}


@router.post("/session/start")
def start_session(
    cwd: str = Query(..., description="Current working directory"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Automatically start a Claude Code session with context restoration
    
    This endpoint:
    1. Creates a new session ID
    2. Detects project type and configuration
    3. Restores context from previous sessions
    4. Loads handoff notes and unfinished tasks
    5. Returns everything Claude needs to continue work
    """
    try:
        result = session_manager.start_session(cwd, db)
        
        # Add helpful instructions
        result["instructions"] = {
            "context_restored": True,
            "usage": "Review the restored context to understand project state",
            "handoff_notes": "Check handoff_notes for tasks from previous session",
            "recent_errors": "Review recent_errors for patterns to avoid",
            "unfinished_tasks": "Continue work on any unfinished tasks"
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/handoff")
def create_handoff(
    content: str = Query(..., description="Summary of work done"),
    next_tasks: List[str] = Query(None, description="Tasks for next session"),
    unresolved_issues: List[str] = Query(None, description="Issues that need attention")
) -> Dict[str, Any]:
    """
    Create a handoff note for the next Claude Code session
    
    Call this at the end of a session to:
    1. Summarize what was accomplished
    2. List tasks for next session
    3. Note any unresolved issues
    4. Create a checkpoint in memory
    """
    try:
        result = session_manager.create_handoff_note(content, next_tasks or [], unresolved_issues)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/error/record")
def record_error(
    error_type: str = Query(...),
    error_message: str = Query(...),
    solution: Optional[str] = Query(None),
    worked: bool = Query(False),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Record an error and its solution"""
    try:
        result = session_manager.record_error_with_solution(error_type, error_message, solution, worked, db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/error/similar")
def find_similar_errors(
    error_type: str = Query(...),
    error_message: str = Query(...)
) -> List[Dict[str, Any]]:
    """Find similar errors with working solutions"""
    try:
        return session_manager.get_similar_errors(error_type, error_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])


@router.get("/tasks/predict")
def predict_tasks() -> List[Dict[str, Any]]:
    """Predict next tasks based on current context"""
    try:
        return session_manager.predict_next_tasks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])


@router.post("/session/end")
def end_session(
    summary: Optional[str] = Query(None, description="Session summary")
) -> Dict[str, Any]:
    """End current session and save summary"""
    try:
        result = session_manager.end_session(summary)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/context/{project_path:path}")
def get_project_context(
    project_path: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get comprehensive project context from all memory sources"""
    try:
        return session_manager.get_project_context(f"/{project_path}", db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/current")
def get_current_session() -> Dict[str, Any]:
    """Get information about current session"""
    try:
        import json
        session_file = session_manager.session_file
        
        if not session_file.exists():
            return {"status": "no_active_session"}
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Add context if available
        context_file = session_manager.context_file
        if context_file.exists():
            with open(context_file, 'r') as f:
                context_data = json.load(f)
            session_data["context"] = context_data
        
        return session_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
def get_memory_stats() -> Dict[str, Any]:
    """Get memory system statistics"""
    try:
        import subprocess
        result = subprocess.run(
            [session_manager.memory_cli_path, "stats"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Parse stats output
            lines = result.stdout.strip().split('\n')
            stats = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    stats[key.strip()] = value.strip()
            return {"status": "success", "stats": stats}
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}