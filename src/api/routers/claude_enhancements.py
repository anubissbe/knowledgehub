"""
API endpoints for Claude Code enhancement features
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.memory_service import MemoryService
from ..services.session_continuity import SessionContinuityService
from ..services.project_profiles import ProjectProfilesService
from ..services.error_learning import ErrorLearningService
from ..services.task_prediction import TaskPredictionService
from ..schemas.memory import MemoryResponse


router = APIRouter(prefix="/api/claude", tags=["claude-enhancements"])


# Dependency injection
def get_memory_service(db: Session = Depends(get_db)) -> MemoryService:
    return MemoryService(db)


def get_session_continuity_service(
    db: Session = Depends(get_db),
    memory_service: MemoryService = Depends(get_memory_service)
) -> SessionContinuityService:
    return SessionContinuityService(db, memory_service)


def get_project_profiles_service(
    db: Session = Depends(get_db),
    memory_service: MemoryService = Depends(get_memory_service)
) -> ProjectProfilesService:
    return ProjectProfilesService(db, memory_service)


def get_error_learning_service(
    db: Session = Depends(get_db),
    memory_service: MemoryService = Depends(get_memory_service)
) -> ErrorLearningService:
    return ErrorLearningService(db, memory_service)


def get_task_prediction_service(
    db: Session = Depends(get_db),
    memory_service: MemoryService = Depends(get_memory_service)
) -> TaskPredictionService:
    return TaskPredictionService(db, memory_service)


# Session Continuity Endpoints
@router.post("/session/continue")
async def continue_session(
    previous_session_id: str,
    user_id: str = "claude-code",
    service: SessionContinuityService = Depends(get_session_continuity_service)
) -> Dict[str, Any]:
    """Continue from a previous Claude Code session"""
    try:
        result = await service.continue_session(previous_session_id, user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/context/{session_id}")
async def get_session_context(
    session_id: str,
    previous_session_id: Optional[str] = None,
    max_tokens: int = Query(4000, le=10000),
    service: SessionContinuityService = Depends(get_session_continuity_service)
) -> Dict[str, Any]:
    """Get relevant context for current session"""
    try:
        context = await service.get_relevant_context(
            session_id, previous_session_id, "claude-code", max_tokens
        )
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/handoff")
async def create_handoff_note(
    session_id: str,
    content: str,
    next_tasks: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    service: SessionContinuityService = Depends(get_session_continuity_service)
) -> MemoryResponse:
    """Create a handoff note for next session"""
    try:
        memory = await service.create_handoff_note(
            session_id, content, next_tasks, warnings
        )
        return memory
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/chain/{session_id}")
async def get_session_chain(
    session_id: str,
    max_depth: int = Query(5, le=20),
    service: SessionContinuityService = Depends(get_session_continuity_service)
) -> List[Dict[str, Any]]:
    """Get chain of linked sessions"""
    try:
        chain = await service.get_session_chain(session_id, max_depth)
        return chain
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project Profiles Endpoints
@router.post("/project/detect")
async def detect_project(
    cwd: str,
    service: ProjectProfilesService = Depends(get_project_profiles_service)
) -> Dict[str, Any]:
    """Detect project from working directory"""
    try:
        profile = await service.detect_project(cwd)
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/context")
async def load_project_context(
    project_id: str,
    session_id: Optional[str] = None,
    service: ProjectProfilesService = Depends(get_project_profiles_service)
) -> Dict[str, Any]:
    """Load project-specific context"""
    try:
        context = await service.load_context(project_id, session_id)
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project/{project_id}/preference")
async def save_project_preference(
    project_id: str,
    preference_key: str,
    preference_value: Any,
    session_id: Optional[str] = None,
    service: ProjectProfilesService = Depends(get_project_profiles_service)
) -> MemoryResponse:
    """Save project-specific preference"""
    try:
        memory = await service.save_project_preference(
            project_id, preference_key, preference_value, session_id
        )
        return memory
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project/{project_id}/pattern")
async def record_project_pattern(
    project_id: str,
    pattern: str,
    pattern_type: str,
    success: bool = True,
    session_id: Optional[str] = None,
    service: ProjectProfilesService = Depends(get_project_profiles_service)
) -> MemoryResponse:
    """Record project-specific pattern"""
    try:
        memory = await service.record_project_pattern(
            project_id, pattern, pattern_type, success, session_id
        )
        return memory
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/summary")
async def get_project_summary(
    project_id: str,
    service: ProjectProfilesService = Depends(get_project_profiles_service)
) -> Dict[str, Any]:
    """Get project knowledge summary"""
    try:
        summary = await service.get_project_summary(project_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error Learning Endpoints
@router.post("/error/record")
async def record_error(
    error_type: str,
    error_message: str,
    context: str,
    solution_applied: Optional[str] = None,
    success: bool = False,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    service: ErrorLearningService = Depends(get_error_learning_service)
) -> MemoryResponse:
    """Record an error and its solution"""
    try:
        memory = await service.record_error(
            error_type, error_message, context,
            solution_applied, success, session_id, project_id
        )
        return memory
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/error/find-similar")
async def find_similar_errors(
    error_type: str,
    error_message: str,
    context: Optional[str] = None,
    limit: int = Query(5, le=20),
    service: ErrorLearningService = Depends(get_error_learning_service)
) -> List[Dict[str, Any]]:
    """Find similar errors and solutions"""
    try:
        similar = await service.find_similar_errors(
            error_type, error_message, context, limit
        )
        return similar
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/error/patterns")
async def get_error_patterns(
    project_id: Optional[str] = None,
    time_range_days: int = Query(30, le=365),
    service: ErrorLearningService = Depends(get_error_learning_service)
) -> Dict[str, Any]:
    """Get error patterns and statistics"""
    try:
        patterns = await service.get_error_patterns(project_id, time_range_days)
        return patterns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/error/suggest-solution")
async def suggest_error_solution(
    error_type: str,
    error_message: str,
    context: str,
    service: ErrorLearningService = Depends(get_error_learning_service)
) -> Optional[Dict[str, Any]]:
    """Get solution suggestion for error"""
    try:
        suggestion = await service.suggest_solution(
            error_type, error_message, context
        )
        return suggestion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Task Prediction Endpoints
@router.post("/task/predict")
async def predict_next_tasks(
    current_context: Dict[str, Any],
    session_id: Optional[str] = None,
    limit: int = Query(5, le=10),
    service: TaskPredictionService = Depends(get_task_prediction_service)
) -> List[Dict[str, Any]]:
    """Predict likely next tasks"""
    try:
        predictions = await service.predict_next_tasks(
            current_context, session_id, limit
        )
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/task/prepare-context")
async def prepare_task_context(
    likely_tasks: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    service: TaskPredictionService = Depends(get_task_prediction_service)
) -> Dict[str, Any]:
    """Prepare context for predicted tasks"""
    try:
        context = await service.prepare_for_tasks(likely_tasks, session_id)
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/task/track-completion")
async def track_task_completion(
    task: str,
    completed: bool,
    session_id: str,
    time_taken: Optional[float] = None,
    obstacles: Optional[List[str]] = None,
    service: TaskPredictionService = Depends(get_task_prediction_service)
) -> Dict[str, str]:
    """Track task completion for learning"""
    try:
        await service.track_task_completion(
            task, completed, session_id, time_taken, obstacles
        )
        return {"status": "tracked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Combined endpoint for Claude Code initialization
@router.post("/initialize")
async def initialize_claude_session(
    cwd: str,
    previous_session_id: Optional[str] = None,
    user_id: str = "claude-code",
    session_service: SessionContinuityService = Depends(get_session_continuity_service),
    project_service: ProjectProfilesService = Depends(get_project_profiles_service),
    task_service: TaskPredictionService = Depends(get_task_prediction_service)
) -> Dict[str, Any]:
    """
    Initialize Claude Code with all enhancement features
    
    This endpoint:
    1. Detects the current project
    2. Continues from previous session if provided
    3. Loads project context
    4. Predicts likely tasks
    5. Returns everything Claude Code needs to start
    """
    try:
        # Detect project
        project = await project_service.detect_project(cwd)
        
        # Continue session or create new
        if previous_session_id:
            session_data = await session_service.continue_session(
                previous_session_id, user_id
            )
        else:
            # Create new session
            session_data = {
                "session_id": f"new-session-{datetime.utcnow().isoformat()}",
                "context": {"formatted_context": "Starting fresh session"},
                "handoff_summary": None
            }
        
        # Load project context
        project_context = await project_service.load_context(
            project["project_id"],
            session_data["session_id"]
        )
        
        # Predict tasks
        current_context = {
            "project_id": project["project_id"],
            "session_id": session_data["session_id"],
            "project_type": project.get("type"),
            "language": project.get("language")
        }
        
        predicted_tasks = await task_service.predict_next_tasks(
            current_context,
            session_data["session_id"]
        )
        
        # Prepare task context
        task_context = await task_service.prepare_for_tasks(
            predicted_tasks,
            session_data["session_id"]
        )
        
        return {
            "session": session_data,
            "project": project,
            "project_context": project_context,
            "predicted_tasks": predicted_tasks,
            "task_context": task_context,
            "initialized_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))