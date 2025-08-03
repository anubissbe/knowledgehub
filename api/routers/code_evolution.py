"""
Code Evolution API - Track code changes, refactoring patterns, and improvements
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body, File, UploadFile
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.code_evolution_tracker import CodeEvolutionTracker

router = APIRouter(prefix="/api/code-evolution", tags=["code-evolution"])

# Global code evolution tracker
evolution_tracker = CodeEvolutionTracker()


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "code-evolution",
        "description": "Track code changes, refactoring patterns, and improvements over time"
    }


@router.post("/track")
def track_code_change_simple(
    data: Dict[str, Any] = Body(..., description="Code change data"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Track a code change (simplified endpoint)
    
    Expected data format:
    {
        "file_path": "test.py",
        "change_type": "refactor",
        "description": "Test refactoring",
        "user_id": "test_user"
    }
    """
    try:
        file_path = data.get("file_path", "")
        change_type = data.get("change_type", "modification")
        description = data.get("description", "")
        user_id = data.get("user_id", "unknown")
        
        # For simple tracking, we'll just record metadata
        result = evolution_tracker.track_simple_change(
            db, file_path, change_type, description, user_id
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-change")
def track_code_change(
    file_path: str = Query(..., description="Path to the file that changed"),
    change_description: str = Query(..., description="Description of the change"),
    change_reason: str = Query(..., description="Why the change was made"),
    project_id: Optional[str] = Query(None, description="Project identifier"),
    session_id: Optional[str] = Query(None, description="Session identifier"),
    change_data: Dict[str, str] = Body(..., description="Before and after code"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Track a code change with before/after comparison
    
    change_data format:
    {
        "before_code": "original code content",
        "after_code": "modified code content"
    }
    """
    try:
        before_code = change_data.get("before_code", "")
        after_code = change_data.get("after_code", "")
        
        if not before_code or not after_code:
            raise HTTPException(
                status_code=400, 
                detail="Both before_code and after_code must be provided"
            )
        
        result = evolution_tracker.track_code_change(
            db, file_path, before_code, after_code,
            change_description, change_reason, project_id, session_id
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def get_evolution_history(
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    change_type: Optional[str] = Query(None, description="Filter by change type"),
    limit: int = Query(10, le=50, description="Maximum results"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get code evolution history with optional filters
    """
    try:
        history = evolution_tracker.get_evolution_history(
            db, file_path, project_id, change_type, limit
        )
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_path:path}/history")
def get_file_history(
    file_path: str,
    limit: int = Query(20, le=100, description="Maximum results"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get evolution history for a specific file
    """
    try:
        # Get all changes for this file
        history = evolution_tracker.get_evolution_history(
            db, file_path=file_path, project_id=None, change_type=None, limit=limit
        )
        
        return {
            "file_path": file_path,
            "changes": history,
            "total_changes": len(history),
            "change_types": list(set(change.get("change_type", "unknown") for change in history))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare/{change_id}")
def compare_code_versions(
    change_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed before/after comparison for a specific change
    """
    comparison = evolution_tracker.compare_code_versions(db, change_id)
    
    if not comparison:
        raise HTTPException(status_code=404, detail="Change not found")
    
    return comparison


@router.post("/suggest-refactoring")
def get_refactoring_suggestions(
    file_path: str = Query(..., description="Path to the file to analyze"),
    project_id: Optional[str] = Query(None, description="Project identifier"),
    code_data: Dict[str, str] = Body(..., description="Current code to analyze"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get refactoring suggestions based on learned patterns
    
    code_data format:
    {
        "code": "current code content to analyze"
    }
    """
    try:
        code = code_data.get("code", "")
        if not code:
            raise HTTPException(status_code=400, detail="Code content must be provided")
        
        suggestions = evolution_tracker.get_refactoring_suggestions(
            db, code, file_path, project_id
        )
        
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-impact")
def update_change_impact(
    change_id: str = Query(..., description="Change ID to update"),
    success_rating: float = Query(..., ge=0.0, le=1.0, description="Success rating (0-1)"),
    impact_data: Dict[str, Any] = Body(..., description="Impact measurement data"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Update a change record with measured impact
    
    impact_data format:
    {
        "impact_notes": "Description of actual impact",
        "performance_impact": {
            "execution_time_change": "-15%",
            "memory_usage_change": "+5%",
            "bug_count_change": "-3"
        }
    }
    """
    try:
        impact_notes = impact_data.get("impact_notes", "")
        performance_impact = impact_data.get("performance_impact")
        
        result = evolution_tracker.update_change_impact(
            db, change_id, success_rating, impact_notes, performance_impact
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/analytics")
def get_pattern_analytics(
    project_id: Optional[str] = Query(None, description="Filter by project")
) -> Dict[str, Any]:
    """
    Get analytics on refactoring patterns and code evolution trends
    """
    try:
        analytics = evolution_tracker.get_pattern_analytics(project_id)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/learned")
def get_learned_patterns() -> Dict[str, Any]:
    """
    Get all learned refactoring patterns
    """
    return {
        "refactoring_patterns": evolution_tracker.refactoring_patterns,
        "learned_patterns": dict(evolution_tracker.learned_patterns)
    }


@router.get("/suggestions/file")
def analyze_file_for_improvements(
    file_path: str = Query(..., description="Path to file to analyze"),
    project_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Analyze a file and suggest improvements based on evolution patterns
    """
    try:
        # This would typically read the file from the filesystem
        # For now, return analysis based on historical patterns
        
        # Find evolution history for this file
        history = evolution_tracker.get_evolution_history(
            db, file_path=file_path, project_id=project_id, limit=5
        )
        
        # Get pattern analytics
        analytics = evolution_tracker.get_pattern_analytics(project_id)
        
        suggestions = {
            "file_path": file_path,
            "evolution_history": history,
            "pattern_recommendations": [],
            "improvement_opportunities": []
        }
        
        # Generate recommendations based on history
        if history:
            successful_patterns = set()
            for change in history:
                if change.get("quality_improvement", 0) > 0.1:
                    successful_patterns.update(change.get("patterns", []))
            
            for pattern in successful_patterns:
                if pattern in evolution_tracker.refactoring_patterns:
                    pattern_info = evolution_tracker.refactoring_patterns[pattern]
                    suggestions["pattern_recommendations"].append({
                        "pattern": pattern,
                        "description": pattern_info["description"],
                        "benefits": pattern_info["benefits"],
                        "confidence": 0.8  # Based on historical success
                    })
        
        # General improvement opportunities
        common_improvements = [
            {
                "type": "add_type_hints",
                "description": "Consider adding type hints for better IDE support",
                "pattern": "improve_typing"
            },
            {
                "type": "add_error_handling", 
                "description": "Consider adding error handling for robustness",
                "pattern": "add_error_handling"
            },
            {
                "type": "extract_functions",
                "description": "Consider extracting large functions for better modularity",
                "pattern": "extract_method"
            }
        ]
        
        suggestions["improvement_opportunities"] = common_improvements
        
        return suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
def get_evolution_trends(
    project_id: Optional[str] = Query(None),
    days: int = Query(30, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get code evolution trends over time
    """
    try:
        # Get recent evolution history
        history = evolution_tracker.get_evolution_history(
            db, project_id=project_id, limit=100
        )
        
        # Filter by date range (simplified)
        recent_history = history[:min(days, len(history))]
        
        trends = {
            "total_changes": len(recent_history),
            "change_types": {},
            "quality_trend": [],
            "pattern_usage": {},
            "avg_quality_improvement": 0
        }
        
        # Analyze change types
        for change in recent_history:
            change_type = change.get("type", "unknown")
            trends["change_types"][change_type] = trends["change_types"].get(change_type, 0) + 1
        
        # Analyze pattern usage
        for change in recent_history:
            for pattern in change.get("patterns", []):
                trends["pattern_usage"][pattern] = trends["pattern_usage"].get(pattern, 0) + 1
        
        # Calculate average quality improvement
        quality_improvements = [c.get("quality_improvement", 0) for c in recent_history]
        if quality_improvements:
            trends["avg_quality_improvement"] = sum(quality_improvements) / len(quality_improvements)
        
        # Quality trend over time (simplified)
        trends["quality_trend"] = [
            {
                "period": f"Period {i+1}",
                "avg_improvement": quality_improvements[i] if i < len(quality_improvements) else 0
            }
            for i in range(min(10, len(quality_improvements)))
        ]
        
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-diff")
async def upload_diff_file(
    file: UploadFile = File(...),
    change_description: str = Query(...),
    change_reason: str = Query(...),
    file_path: str = Query(...),
    project_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload a diff file to track code evolution
    """
    try:
        # Read the uploaded diff file
        diff_content = await file.read()
        diff_text = diff_content.decode('utf-8')
        
        # Parse the diff to extract before and after code
        # This is a simplified implementation
        lines = diff_text.split('\n')
        before_lines = []
        after_lines = []
        
        current_section = None
        for line in lines:
            if line.startswith('---'):
                current_section = 'before'
                continue
            elif line.startswith('+++'):
                current_section = 'after'
                continue
            elif line.startswith('@@'):
                continue
            elif line.startswith('-') and current_section == 'before':
                before_lines.append(line[1:])
            elif line.startswith('+') and current_section == 'after':
                after_lines.append(line[1:])
            elif not line.startswith(('+', '-')):
                # Context line
                before_lines.append(line)
                after_lines.append(line)
        
        before_code = '\n'.join(before_lines)
        after_code = '\n'.join(after_lines)
        
        # Track the change
        result = evolution_tracker.track_code_change(
            db, file_path, before_code, after_code,
            change_description, change_reason, project_id, session_id
        )
        
        result["diff_processed"] = True
        result["diff_size"] = len(diff_text)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
def search_evolution_records(
    query: str = Query(..., description="Search query"),
    change_type: Optional[str] = Query(None),
    min_quality_improvement: Optional[float] = Query(None, ge=0.0, le=1.0),
    project_id: Optional[str] = Query(None),
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Search through code evolution records
    """
    try:
        # Get all evolution history and filter
        history = evolution_tracker.get_evolution_history(
            db, project_id=project_id, change_type=change_type, limit=limit * 2
        )
        
        # Filter by search query
        filtered_history = []
        query_lower = query.lower()
        
        for record in history:
            # Search in description, file path, and patterns
            searchable_text = f"{record.get('description', '')} {record.get('file_path', '')} {' '.join(record.get('patterns', []))}"
            
            if query_lower in searchable_text.lower():
                # Apply quality filter if specified
                if min_quality_improvement is None or record.get('quality_improvement', 0) >= min_quality_improvement:
                    filtered_history.append(record)
        
        return filtered_history[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])