"""
Error Tracking & Learning API Router.

This module provides REST API endpoints for AI-powered error tracking,
pattern recognition, and solution discovery.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session as DBSession

from ..models.base import get_db
from ..models.error_pattern import (
    ErrorOccurrenceCreate, ErrorSolutionCreate, ErrorFeedbackCreate,
    ErrorPatternResponse, ErrorPredictionResponse, ErrorAnalytics,
    ErrorCategory, ErrorSeverity
)
from ..services.error_learning_service import error_learning_service
from ..workers.error_analyzer import start_error_analyzer, stop_error_analyzer
from shared.logging import setup_logging

logger = setup_logging("error_tracking_router")

router = APIRouter(prefix="/api/errors", tags=["error-tracking"])


@router.post("/occurrences", response_model=Dict[str, Any])
async def record_error_occurrence(
    error_data: ErrorOccurrenceCreate,
    user_id: str = Query(..., description="User ID"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Record a new error occurrence and get suggested solutions.
    
    Features:
    - Automatic pattern matching (exact, fuzzy, semantic)
    - Solution suggestions ranked by effectiveness
    - Background pattern analysis
    - Predictive error prevention
    """
    try:
        match_result = await error_learning_service.record_error(
            error_data=error_data,
            user_id=user_id,
            session_id=session_id
        )
        
        # Schedule background analysis
        background_tasks.add_task(_analyze_error_context, user_id, session_id)
        
        logger.info(f"Recorded error for user {user_id}: {match_result.pattern.error_type}")
        
        return {
            "pattern_id": str(match_result.pattern.id),
            "error_type": match_result.pattern.error_type,
            "error_category": match_result.pattern.error_category,
            "severity": match_result.pattern.severity,
            "match_type": match_result.match_type,
            "similarity_score": match_result.similarity_score,
            "confidence": match_result.confidence,
            "primary_solution": match_result.pattern.primary_solution,
            "suggested_solutions": [
                {
                    "solution_id": str(sol.id),
                    "solution_text": sol.solution_text,
                    "effectiveness_score": sol.effectiveness_score,
                    "success_count": sol.success_count
                }
                for sol in match_result.suggested_solutions[:5]
            ],
            "prevention_tips": match_result.pattern.prerequisites.get("prevention", [])
                if match_result.pattern.prerequisites else []
        }
        
    except Exception as e:
        logger.error(f"Failed to record error occurrence: {e}")
        raise HTTPException(status_code=500, detail="Failed to record error")


@router.post("/feedback")
async def provide_error_feedback(
    feedback_data: ErrorFeedbackCreate,
    user_id: str = Query(..., description="User ID")
):
    """
    Provide feedback on error pattern or solution effectiveness.
    
    Updates:
    - Pattern confidence scores
    - Solution effectiveness ratings
    - Learning metrics
    """
    try:
        success = await error_learning_service.provide_feedback(
            feedback_data=feedback_data,
            user_id=user_id
        )
        
        if success:
            logger.info(f"Recorded feedback for pattern {feedback_data.pattern_id}")
            return {"status": "success", "message": "Feedback recorded successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to record feedback")
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")


@router.get("/predictions", response_model=Dict[str, Any])
async def get_error_predictions(
    user_id: str = Query(..., description="User ID"),
    session_id: str = Query(..., description="Session ID"),
    context: Optional[str] = Query(None, description="JSON context string")
):
    """
    Get error predictions based on current context and history.
    
    Provides:
    - Likely error patterns
    - Risk assessment
    - Prevention recommendations
    """
    try:
        # Parse context if provided
        import json
        context_dict = json.loads(context) if context else {}
        
        prediction_result = await error_learning_service.predict_errors(
            user_id=user_id,
            session_id=session_id,
            context=context_dict
        )
        
        return {
            "risk_score": prediction_result.risk_score,
            "confidence": prediction_result.confidence,
            "predicted_errors": [
                {
                    "pattern_id": str(pattern.id),
                    "error_type": pattern.error_type,
                    "error_message": pattern.error_message,
                    "severity": pattern.severity,
                    "likelihood": risk_factor.get("likelihood", 0.0)
                }
                for pattern, risk_factor in zip(
                    prediction_result.predicted_patterns[:5],
                    prediction_result.risk_factors[:5]
                )
            ],
            "prevention_steps": prediction_result.prevention_steps,
            "risk_factors": prediction_result.risk_factors
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON context")
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get error predictions")


@router.get("/analytics", response_model=ErrorAnalytics)
async def get_error_analytics(
    user_id: Optional[str] = Query(None, description="User ID filter"),
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    time_window_hours: int = Query(168, description="Time window in hours", ge=1, le=8760)
):
    """
    Get comprehensive error analytics.
    
    Includes:
    - Error pattern statistics
    - Resolution metrics
    - Learning progress
    - Prediction accuracy
    """
    try:
        analytics = await error_learning_service.get_error_analytics(
            user_id=user_id,
            project_id=project_id,
            time_window_hours=time_window_hours
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get error analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve error analytics")


@router.get("/patterns", response_model=List[ErrorPatternResponse])
async def list_error_patterns(
    category: Optional[ErrorCategory] = Query(None, description="Error category filter"),
    severity: Optional[ErrorSeverity] = Query(None, description="Severity filter"),
    min_occurrences: int = Query(1, description="Minimum occurrences", ge=1),
    min_confidence: float = Query(0.0, description="Minimum confidence score", ge=0.0, le=1.0),
    limit: int = Query(50, description="Maximum results", ge=1, le=200),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    db: DBSession = Depends(get_db)
):
    """
    List error patterns with filtering and pagination.
    """
    try:
        from ..models.error_pattern import EnhancedErrorPattern
        
        # Build query
        query = db.query(EnhancedErrorPattern)
        
        if category:
            query = query.filter(EnhancedErrorPattern.error_category == category.value)
        if severity:
            query = query.filter(EnhancedErrorPattern.severity == severity.value)
        
        query = query.filter(
            EnhancedErrorPattern.occurrences >= min_occurrences,
            EnhancedErrorPattern.confidence_score >= min_confidence
        )
        
        # Apply ordering and pagination
        patterns = query.order_by(
            EnhancedErrorPattern.occurrences.desc(),
            EnhancedErrorPattern.confidence_score.desc()
        ).offset(offset).limit(limit).all()
        
        # Convert to response format
        return [
            ErrorPatternResponse(
                id=str(pattern.id),
                pattern_hash=pattern.pattern_hash,
                error_type=pattern.error_type,
                error_category=ErrorCategory(pattern.error_category),
                error_message=pattern.error_message,
                error_code=pattern.error_code,
                severity=ErrorSeverity(pattern.severity),
                primary_solution=pattern.primary_solution,
                alternative_solutions=pattern.alternative_solutions or [],
                solution_steps=pattern.solution_steps or [],
                occurrences=pattern.occurrences,
                success_rate=pattern.success_rate,
                confidence_score=pattern.confidence_score,
                first_seen=pattern.first_seen,
                last_seen=pattern.last_seen
            )
            for pattern in patterns
        ]
        
    except Exception as e:
        logger.error(f"Failed to list error patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to list error patterns")


@router.get("/patterns/{pattern_id}", response_model=Dict[str, Any])
async def get_error_pattern_details(
    pattern_id: str = Path(..., description="Pattern ID"),
    include_solutions: bool = Query(True, description="Include solutions"),
    include_recent_occurrences: bool = Query(False, description="Include recent occurrences"),
    db: DBSession = Depends(get_db)
):
    """Get detailed information about a specific error pattern."""
    try:
        from ..models.error_pattern import EnhancedErrorPattern, ErrorSolution, ErrorOccurrence
        
        # Get pattern
        pattern = db.query(EnhancedErrorPattern).filter_by(id=pattern_id).first()
        
        if not pattern:
            raise HTTPException(status_code=404, detail="Error pattern not found")
        
        response = {
            "pattern": ErrorPatternResponse(
                id=str(pattern.id),
                pattern_hash=pattern.pattern_hash,
                error_type=pattern.error_type,
                error_category=ErrorCategory(pattern.error_category),
                error_message=pattern.error_message,
                error_code=pattern.error_code,
                severity=ErrorSeverity(pattern.severity),
                primary_solution=pattern.primary_solution,
                alternative_solutions=pattern.alternative_solutions or [],
                solution_steps=pattern.solution_steps or [],
                occurrences=pattern.occurrences,
                success_rate=pattern.success_rate,
                confidence_score=pattern.confidence_score,
                first_seen=pattern.first_seen,
                last_seen=pattern.last_seen
            ),
            "metadata": {
                "key_indicators": pattern.key_indicators,
                "prerequisites": pattern.prerequisites,
                "environment_factors": pattern.environment_factors,
                "avg_resolution_time": pattern.avg_resolution_time,
                "false_positive_rate": pattern.false_positive_rate
            }
        }
        
        if include_solutions:
            solutions = db.query(ErrorSolution).filter_by(
                pattern_id=pattern_id
            ).order_by(
                ErrorSolution.effectiveness_score.desc()
            ).limit(10).all()
            
            response["solutions"] = [
                {
                    "solution_id": str(sol.id),
                    "solution_text": sol.solution_text,
                    "solution_steps": sol.solution_steps,
                    "effectiveness_score": sol.effectiveness_score,
                    "success_count": sol.success_count,
                    "failure_count": sol.failure_count,
                    "avg_resolution_time": sol.avg_resolution_time,
                    "status": sol.status
                }
                for sol in solutions
            ]
        
        if include_recent_occurrences:
            recent_occurrences = db.query(ErrorOccurrence).filter_by(
                pattern_id=pattern_id
            ).order_by(
                ErrorOccurrence.timestamp.desc()
            ).limit(10).all()
            
            response["recent_occurrences"] = [
                {
                    "occurrence_id": str(occ.id),
                    "user_id": occ.user_id,
                    "timestamp": occ.timestamp.isoformat(),
                    "resolved": occ.resolved,
                    "resolution_time": occ.resolution_time
                }
                for occ in recent_occurrences
            ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get error pattern details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pattern details")


@router.post("/solutions", response_model=Dict[str, str])
async def create_error_solution(
    solution_data: ErrorSolutionCreate,
    created_by: str = Query(..., description="Creator user ID"),
    db: DBSession = Depends(get_db)
):
    """Create a new solution for an error pattern."""
    try:
        from ..models.error_pattern import ErrorSolution
        
        solution = ErrorSolution(
            pattern_id=solution_data.pattern_id,
            solution_text=solution_data.solution_text,
            solution_code=solution_data.solution_code,
            solution_steps=solution_data.solution_steps,
            prerequisites=solution_data.prerequisites,
            created_by=created_by
        )
        
        db.add(solution)
        db.commit()
        
        logger.info(f"Created solution {solution.id} for pattern {solution_data.pattern_id}")
        return {"solution_id": str(solution.id), "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create solution: {e}")
        raise HTTPException(status_code=500, detail="Failed to create solution")


@router.post("/analyzer/start")
async def start_analyzer_worker():
    """Start the background error analyzer worker."""
    try:
        await start_error_analyzer()
        logger.info("Started error analyzer worker")
        return {"status": "started", "message": "Error analyzer worker started successfully"}
        
    except Exception as e:
        logger.error(f"Failed to start analyzer: {e}")
        raise HTTPException(status_code=500, detail="Failed to start analyzer worker")


@router.post("/analyzer/stop")
async def stop_analyzer_worker():
    """Stop the background error analyzer worker."""
    try:
        await stop_error_analyzer()
        logger.info("Stopped error analyzer worker")
        return {"status": "stopped", "message": "Error analyzer worker stopped successfully"}
        
    except Exception as e:
        logger.error(f"Failed to stop analyzer: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop analyzer worker")


# Background task functions

async def _analyze_error_context(user_id: str, session_id: Optional[str]):
    """Background task to analyze error context."""
    try:
        # Perform additional analysis in background
        if session_id:
            logger.debug(f"Analyzing error context for session {session_id}")
            # Could trigger additional pattern analysis, predictions, etc.
        
    except Exception as e:
        logger.warning(f"Background error analysis failed: {e}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Error tracking service health check."""
    try:
        # Basic service health check
        if not error_learning_service._initialized:
            await error_learning_service.initialize()
        
        return {
            "status": "healthy",
            "service": "error_tracking",
            "initialized": error_learning_service._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "error_tracking",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )