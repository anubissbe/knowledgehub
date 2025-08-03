"""
Mistake Learning API - Track, analyze, and prevent repeated errors
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Depends, Body
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.mistake_learning_system import MistakeLearningSystem

router = APIRouter(prefix="/api/mistake-learning", tags=["mistake-learning"])

# Global mistake learning system
learning_system = MistakeLearningSystem()


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mistake-learning",
        "description": "Learning from mistakes to prevent repetition"
    }


@router.post("/track")
def track_mistake(
    data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Track a mistake with full context and solutions
    
    This helps Claude learn from errors and avoid repetition
    """
    try:
        # Extract parameters from JSON body
        error_type = data.get("error_type")
        error_message = data.get("error_message")
        context = data.get("context", {})
        attempted_solution = data.get("attempted_solution")
        successful_solution = data.get("successful_solution") or data.get("solution")
        resolved = data.get("resolved")
        project_id = data.get("project_id")
        
        # Validate required fields
        if not error_type or not error_message:
            raise HTTPException(
                status_code=422,
                detail="Missing required fields: error_type, error_message"
            )
            
        try:
            result = learning_system.track_mistake(
                db, error_type, error_message, context,
                attempted_solution, successful_solution, project_id
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        
        # Add advice based on tracking
        if result["is_repeated"]:
            result["advice"] = f"This mistake has occurred {result['repetition_count']} times. Check lessons learned."
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-action")
def check_action(
    action: str = Query(..., description="Action about to be performed"),
    context: Dict[str, Any] = Body({}, description="Current context")
) -> Dict[str, Any]:
    """
    Check if an action might trigger a known mistake
    
    Call this before performing risky operations
    """
    prevention = learning_system.check_for_prevention(action, context)
    
    if prevention:
        return {
            "should_proceed": False,
            "warning": prevention,
            "message": f"Warning: This action matches pattern '{prevention['pattern']}'. Suggestion: {prevention['suggestion']}"
        }
    
    return {
        "should_proceed": True,
        "warning": None,
        "message": "No known issues with this action"
    }


@router.get("/lessons")
def get_lessons(
    category: Optional[str] = Query(None, description="Filter by error category"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    days: int = Query(30, description="Look back N days"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get lessons learned from past mistakes
    
    Categories: dependency, api_misuse, type_mismatch, syntax, data_access, performance, security
    """
    try:
        lessons = learning_system.get_lessons_learned(db, category, project_id, days)
        return lessons
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])


@router.get("/patterns")
def get_patterns(
    min_occurrences: int = Query(2, description="Minimum occurrences to include"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get analysis of mistake patterns
    
    Shows which types of errors occur most frequently
    """
    try:
        patterns = learning_system.get_mistake_patterns(db, min_occurrences)
        return patterns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
def get_report(
    days: int = Query(7, description="Report period in days"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Generate comprehensive mistake analysis report
    
    Includes:
    - Total mistakes and repetition rate
    - Solution effectiveness
    - Category breakdown
    - Top lessons learned
    """
    try:
        report = learning_system.generate_mistake_report(db, days)
        
        # Add recommendations
        recommendations = []
        if report["repetition_rate"] > 0.3:
            recommendations.append("High repetition rate - review prevention rules")
        if report["solution_rate"] < 0.5:
            recommendations.append("Low solution rate - need better debugging approaches")
        
        report["recommendations"] = recommendations
        
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn-pattern")
def learn_pattern(
    pattern_name: str = Query(..., description="Name for the pattern"),
    regex: str = Query(..., description="Regex to match the error"),
    category: str = Query(..., description="Error category"),
    severity: str = Query("medium", description="Severity: low, medium, high, critical")
) -> Dict[str, Any]:
    """
    Teach the system a new error pattern to recognize
    """
    try:
        # Add to learned patterns
        learning_system.error_patterns[pattern_name] = {
            "regex": regex,
            "category": category,
            "severity": severity,
            "learned": True
        }
        
        # Save to persistent storage
        with open(learning_system.patterns_file, 'w') as f:
            learned_patterns = {
                k: v for k, v in learning_system.error_patterns.items()
                if v.get("learned", False)
            }
            json.dump(learned_patterns, f, indent=2)
        
        return {
            "pattern_name": pattern_name,
            "learned": True,
            "total_patterns": len(learning_system.error_patterns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar-mistakes/{error_type}")
def find_similar_mistakes(
    error_type: str,
    error_message: str = Query(...),
    project_id: Optional[str] = Query(None),
    limit: int = Query(5, le=20),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Find similar mistakes and their solutions
    """
    try:
        similar = learning_system._find_similar_mistakes(
            db, error_type, error_message, project_id
        )
        
        results = []
        for mistake in similar[:limit]:
            results.append({
                "id": str(mistake.id),
                "error_type": mistake.meta_data.get("error_type"),
                "error_message": mistake.meta_data.get("error_message", "")[:200],
                "solution": mistake.meta_data.get("successful_solution"),
                "lesson": mistake.meta_data.get("lesson", {}).get("summary"),
                "created": mistake.created_at.isoformat()
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])


@router.post("/search")
def search_mistakes(
    query: Dict[str, Any] = Body(..., description="Search query"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Search for mistakes based on query
    """
    try:
        # Extract query string from body
        query_str = query.get("query", "")
        
        # Use the learning system to find similar mistakes
        from ..models.mistake_tracking import MistakeTracking
        
        # Search in database
        mistakes = db.query(MistakeTracking).filter(
            (MistakeTracking.error_type.contains(query_str)) |
            (MistakeTracking.error_message.contains(query_str))
        ).limit(10).all()
        
        results = []
        for mistake in mistakes:
            results.append({
                "id": str(mistake.id),
                "error_type": mistake.error_type,
                "error_message": mistake.error_message[:200],
                "solution": mistake.solution,
                "resolved": mistake.resolved,
                "created_at": mistake.created_at.isoformat()
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])


@router.delete("/reset-patterns")
def reset_patterns():
    """Reset learned patterns (for testing)"""
    import os
    for file in [
        learning_system.patterns_file,
        learning_system.prevention_rules_file,
        Path.home() / ".claude_pattern_stats.json"
    ]:
        if file.exists():
            os.unlink(file)
    
    return {"reset": True}