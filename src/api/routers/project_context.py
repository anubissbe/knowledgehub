"""
Project Context API - Automatic context switching and project isolation
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.project_context_manager import ProjectContextManager

router = APIRouter(prefix="/api/project-context", tags=["project-context"])

# Global project context manager
context_manager = ProjectContextManager()


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "project-context",
        "description": "Project context isolation and management"
    }


@router.post("/switch")
def switch_project(
    project_path: str = Query(..., description="Path to project directory"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Switch to a different project context
    
    This automatically:
    1. Saves current project state
    2. Loads new project profile
    3. Switches memory namespace
    4. Restores project-specific context
    """
    try:
        result = context_manager.switch_project_context(project_path, db)
        
        # Add usage hints
        result["hints"] = {
            "conventions": "Project conventions loaded - I'll follow these patterns",
            "memories": f"Loaded {len(result['memories'])} project-specific memories",
            "preferences": "Project preferences applied"
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
def get_current_project() -> Dict[str, Any]:
    """Get information about the currently active project"""
    active = context_manager.get_active_project()
    
    if not active:
        return {"status": "no_active_project"}
    
    # Load full project info
    project_id = active.get("id")
    if project_id:
        config_path = context_manager._get_project_config_path(project_id)
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                project_data = json.load(f)
            return {
                "active": active,
                "profile": project_data
            }
    
    return {"active": active}


@router.get("/list")
def list_projects() -> List[Dict[str, Any]]:
    """List all known projects sorted by last access"""
    return context_manager.list_all_projects()


@router.post("/preference")
def add_preference(
    project_path: str = Query(...),
    key: str = Query(..., description="Preference key"),
    value: str = Query(..., description="Preference value")
) -> Dict[str, Any]:
    """
    Add a project-specific preference
    
    Examples:
    - test_command: "pytest -v"
    - format_command: "black . && isort ."
    - preferred_import_style: "absolute"
    """
    try:
        preferences = context_manager.add_project_preference(project_path, key, value)
        return {"preferences": preferences, "updated": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pattern")
def record_pattern(
    project_path: str = Query(...),
    pattern_type: str = Query(..., description="Type of pattern (e.g., naming_convention)"),
    pattern_value: str = Query(..., description="Pattern value"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Record a discovered pattern for the project
    
    Pattern types:
    - naming_convention: snake_case, camelCase, PascalCase
    - import_style: absolute, relative
    - error_handling: try_except, result_types
    - async_pattern: asyncio, threading
    """
    try:
        patterns = context_manager.record_project_pattern(
            project_path, pattern_type, pattern_value, db
        )
        return {"patterns": patterns, "recorded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conventions/{project_path:path}")
def get_conventions(project_path: str) -> Dict[str, Any]:
    """Get all conventions and patterns for a project"""
    try:
        conventions = context_manager.get_project_conventions(f"/{project_path}")
        return conventions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory")
def add_project_memory(
    project_path: str = Query(...),
    content: str = Body(...),
    memory_type: str = Query("fact"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Add a memory specific to this project"""
    try:
        project_id = context_manager._get_project_id(project_path)
        memory = context_manager._store_project_memory(
            db,
            project_id,
            content,
            {
                "memory_type": memory_type,
                "importance": 0.7,
                "source": "api"
            }
        )
        return {
            "id": str(memory.id),
            "project_id": project_id,
            "stored": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memories")
def get_project_memories(
    project_path: str = Query(...),
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get memories specific to a project"""
    try:
        project_id = context_manager._get_project_id(project_path)
        memories = context_manager._load_project_memories(project_id, db, limit)
        return memories
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])


@router.post("/auto-detect")
def auto_detect_project(
    cwd: str = Query(..., description="Current working directory"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Automatically detect and switch to project based on current directory
    
    This is called automatically by Claude Code on session start
    """
    try:
        # Find project root (look for .git, package.json, etc.)
        from pathlib import Path
        current_path = Path(cwd)
        project_root = current_path
        
        # Search up to 5 levels for project indicators
        for _ in range(5):
            if any((project_root / indicator).exists() for indicator in [
                ".git", "package.json", "requirements.txt", "Cargo.toml", 
                "go.mod", "pom.xml", ".project"
            ]):
                break
            if project_root.parent == project_root:
                break
            project_root = project_root.parent
        
        # Switch to detected project
        result = context_manager.switch_project_context(str(project_root), db)
        result["auto_detected"] = True
        result["detected_root"] = str(project_root)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-active")
def clear_active_project():
    """Clear the active project (useful for testing)"""
    if context_manager.active_project_file.exists():
        context_manager.active_project_file.unlink()
    return {"cleared": True}