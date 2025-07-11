"""
Synchronous Claude Code Enhancement API - simple working version
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import cast, String, and_, desc

from ..models.base import SessionLocal
from ..models.memory import MemoryItem

router = APIRouter(prefix="/api/claude-sync", tags=["claude-sync"])


def create_memory_directly(db: Session, content: str, tags: List[str], metadata: Dict[str, Any]) -> MemoryItem:
    """Create memory item directly without async"""
    memory_hash = hashlib.sha256(content.encode()).hexdigest()
    
    # Check if exists
    existing = db.query(MemoryItem).filter(MemoryItem.content_hash == memory_hash).first()
    if existing:
        existing.access_count += 1
        existing.accessed_at = datetime.utcnow()
        db.commit()
        return existing
    
    # Create new
    memory = MemoryItem(
        content=content,
        content_hash=memory_hash,
        tags=tags,
        meta_data=metadata,
        access_count=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        accessed_at=datetime.utcnow()
    )
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "claude-sync"}


@router.post("/initialize")
def initialize_claude(cwd: str = Query(..., description="Current working directory")) -> Dict[str, Any]:
    """Initialize Claude with all enhancements"""
    db = SessionLocal()
    try:
        # Detect project
        project_path = Path(cwd)
        project_id = hashlib.md5(str(project_path).encode()).hexdigest()[:12]
        
        project_info = {
            "id": project_id,
            "path": str(project_path),
            "name": project_path.name,
            "type": "python" if (project_path / "setup.py").exists() or (project_path / "requirements.txt").exists() else "unknown",
            "language": "python"
        }
        
        # Store project profile
        create_memory_directly(
            db,
            f"Project Profile: {project_info['name']} ({project_info['type']})",
            ["project", project_info["type"]],
            {
                "memory_type": "entity",
                "importance": 0.8,
                "project_id": project_id,
                **project_info
            }
        )
        
        # Create session
        session_id = f"session-{datetime.utcnow().isoformat()}"
        
        # Get project context
        project_memories = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
        ).order_by(MemoryItem.created_at.desc()).limit(10).all()
        
        # Check for handoff notes
        handoff_memories = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains('"handoff": true')
        ).order_by(desc(MemoryItem.created_at)).limit(5).all()
        
        predictions = []
        for mem in handoff_memories:
            next_tasks = mem.meta_data.get("next_tasks", [])
            for task in next_tasks[:2]:
                predictions.append({
                    "task": task,
                    "type": "handoff",
                    "confidence": 0.9
                })
        
        return {
            "initialized_at": datetime.utcnow().isoformat(),
            "session": {
                "session_id": session_id,
                "context": "Starting fresh session"
            },
            "project": project_info,
            "project_context": {
                "total_memories": len(project_memories),
                "recent": [m.content[:100] for m in project_memories[:3]]
            },
            "predicted_tasks": predictions[:5]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/session/continue")
def continue_session(previous_session_id: str = Query(...)) -> Dict[str, Any]:
    """Continue from a previous session"""
    db = SessionLocal()
    try:
        new_session_id = f"session-{datetime.utcnow().isoformat()}"
        
        # Get previous session memories
        previous_memories = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"session_id": "{previous_session_id}"')
        ).order_by(MemoryItem.created_at.desc()).limit(20).all()
        
        # Create continuation memory
        handoff_content = f"SESSION CONTINUATION: Continuing from {previous_session_id}"
        if previous_memories:
            handoff_content += f"\nLast activity: {previous_memories[0].content[:100]}..."
        
        create_memory_directly(
            db,
            handoff_content,
            ["session", "continuation"],
            {
                "memory_type": "decision",
                "importance": 0.9,
                "session_id": new_session_id,
                "previous_session": previous_session_id
            }
        )
        
        return {
            "session_id": new_session_id,
            "previous_session_id": previous_session_id,
            "context": handoff_content,
            "memory_count": len(previous_memories)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/error/record")
def record_error(
    error_type: str = Query(...),
    error_message: str = Query(...),
    solution: Optional[str] = Query(None),
    success: bool = Query(False),
    session_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Record an error and solution"""
    db = SessionLocal()
    try:
        content = f"ERROR [{error_type}]: {error_message}"
        if solution:
            content += f"\nSOLUTION: {solution} ({'✓' if success else '✗'})"
        
        memory = create_memory_directly(
            db,
            content,
            ["error", error_type.lower()],
            {
                "memory_type": "error",
                "importance": 0.8 if not success else 0.9,
                "error_type": error_type,
                "solution": solution,
                "success": success,
                "session_id": session_id,
                "project_id": project_id
            }
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


@router.post("/session/handoff")
def create_handoff(
    session_id: str = Query(...),
    content: str = Query(...),
    next_tasks: Optional[List[str]] = Query(None)
) -> Dict[str, Any]:
    """Create handoff note for next session"""
    db = SessionLocal()
    try:
        handoff_content = f"HANDOFF NOTE: {content}"
        if next_tasks:
            handoff_content += "\nNext tasks:\n" + "\n".join(f"- {task}" for task in next_tasks)
        
        memory = create_memory_directly(
            db,
            handoff_content,
            ["handoff", "session"],
            {
                "memory_type": "decision",
                "importance": 0.95,
                "handoff": True,
                "next_tasks": next_tasks,
                "session_id": session_id
            }
        )
        
        return {
            "id": str(memory.id),
            "content": memory.content,
            "created": memory.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/error/similar")
def find_similar_errors(
    error_type: str = Query(...),
    error_message: str = Query(...)
) -> List[Dict[str, Any]]:
    """Find similar errors"""
    db = SessionLocal()
    try:
        # Search for similar errors
        error_memories = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains(f'"memory_type": "error"'),
                MemoryItem.content.ilike(f"%{error_type}%")
            )
        ).order_by(desc(MemoryItem.access_count)).limit(10).all()
        
        similar = []
        for mem in error_memories:
            similar.append({
                "id": str(mem.id),
                "error": mem.content[:200],
                "solution": mem.meta_data.get("solution"),
                "success": mem.meta_data.get("success", False),
                "created": mem.created_at.isoformat()
            })
        
        return similar
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/task/predict")
def predict_tasks(
    session_id: str = Query(...),
    project_id: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """Predict next tasks"""
    db = SessionLocal()
    try:
        predictions = []
        
        # Check for unfinished tasks (handoff notes)
        handoff_memories = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains('"handoff": true')
        ).order_by(desc(MemoryItem.created_at)).limit(5).all()
        
        for mem in handoff_memories:
            next_tasks = mem.meta_data.get("next_tasks", [])
            for task in next_tasks[:2]:
                predictions.append({
                    "task": task,
                    "type": "handoff",
                    "confidence": 0.9,
                    "from_session": mem.meta_data.get("session_id")
                })
        
        # Check recent errors without solutions
        unsolved_errors = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "error"'),
                cast(MemoryItem.meta_data, String).contains('"success": false')
            )
        ).order_by(desc(MemoryItem.created_at)).limit(3).all()
        
        for error in unsolved_errors:
            predictions.append({
                "task": f"Fix error: {error.meta_data.get('error_type', 'Unknown')}",
                "type": "error_fix",
                "confidence": 0.7,
                "error_id": str(error.id)
            })
        
        return predictions[:5]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/test-db")
def test_database():
    """Test database connection"""
    db = SessionLocal()
    try:
        # Simple query
        count = db.query(MemoryItem).count()
        return {"status": "connected", "memory_count": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


@router.get("/test-simple")
def test_simple():
    """Test without database"""
    return {"status": "ok", "message": "Simple endpoint works"}


@router.post("/test-project")
def test_project_only(cwd: str = Query(...)):
    """Test project detection only"""
    try:
        project_path = Path(cwd)
        project_id = hashlib.md5(str(project_path).encode()).hexdigest()[:12]
        
        return {
            "id": project_id,
            "path": str(project_path),
            "name": project_path.name,
            "exists": project_path.exists()
        }
    except Exception as e:
        return {"error": str(e)}