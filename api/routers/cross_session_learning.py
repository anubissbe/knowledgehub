"""Cross-Session Learning API Endpoints

Provides API endpoints for managing cross-session learning functionality,
including learning sessions, knowledge transfers, and pattern analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..models import get_db
from ..learning_system.services.learning_session_persistence_sync import LearningSessionPersistence
from ..learning_system.services.cross_session_pattern_recognition_sync import CrossSessionPatternRecognition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cross-session-learning", tags=["cross-session-learning"])


# Pydantic models for API
class CreateLearningSessionRequest(BaseModel):
    session_type: str = Field(..., description="Type of learning session")
    session_name: Optional[str] = Field(None, description="Human-readable name")
    conversation_session_id: Optional[UUID] = Field(None, description="Associated conversation session")
    learning_objectives: Optional[Dict[str, Any]] = Field(None, description="Learning objectives")
    learning_context: Optional[Dict[str, Any]] = Field(None, description="Learning context")
    parent_session_id: Optional[UUID] = Field(None, description="Parent session for continuity")


class KnowledgeTransferRequest(BaseModel):
    source_session_id: UUID = Field(..., description="Source learning session")
    destination_session_id: UUID = Field(..., description="Destination learning session")
    transfer_type: str = Field(..., description="Type of knowledge transfer")
    transfer_data: Dict[str, Any] = Field(..., description="Data to transfer")
    transfer_name: Optional[str] = Field(None, description="Human-readable name")


class LearningSessionResponse(BaseModel):
    session_id: str
    session_type: str
    session_name: Optional[str]
    user_id: str
    status: str
    started_at: str
    ended_at: Optional[str]
    patterns_learned: int
    patterns_reinforced: int
    knowledge_units_created: int
    success_rate: Optional[float]
    learning_effectiveness: Optional[float]


class CrossSessionAnalysisResponse(BaseModel):
    cross_session_patterns: List[Dict[str, Any]]
    pattern_evolution_trends: List[Dict[str, Any]]
    learning_consistency_score: float
    sessions_analyzed: int
    patterns_analyzed: int
    analysis_time_window_days: int


@router.post("/sessions", response_model=LearningSessionResponse)
def create_learning_session(
    user_id: str,
    request: CreateLearningSessionRequest,
    db: Session = Depends(get_db)
):
    """Create a new learning session"""
    
    try:
        persistence = LearningSessionPersistence(db)
        
        session = persistence.create_learning_session(
            user_id=user_id,
            session_type=request.session_type,
            session_name=request.session_name,
            conversation_session_id=request.conversation_session_id,
            learning_objectives=request.learning_objectives,
            learning_context=request.learning_context,
            parent_session_id=request.parent_session_id
        )
        
        return LearningSessionResponse(
            session_id=str(session.id),
            session_type=session.session_type,
            session_name=session.session_name,
            user_id=session.user_id,
            status=session.status,
            started_at=session.started_at.isoformat(),
            ended_at=session.ended_at.isoformat() if session.ended_at else None,
            patterns_learned=session.patterns_learned,
            patterns_reinforced=session.patterns_reinforced,
            knowledge_units_created=session.knowledge_units_created,
            success_rate=session.success_rate,
            learning_effectiveness=session.learning_effectiveness
        )
        
    except Exception as e:
        logger.error(f"Error creating learning session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/active", response_model=List[LearningSessionResponse])
def get_active_sessions(
    user_id: str,
    session_types: Optional[List[str]] = Query(None),
    db: Session = Depends(get_db)
):
    """Get active learning sessions for a user"""
    
    try:
        persistence = LearningSessionPersistence(db)
        sessions = persistence.get_active_learning_sessions(user_id, session_types)
        
        return [
            LearningSessionResponse(
                session_id=str(session.id),
                session_type=session.session_type,
                session_name=session.session_name,
                user_id=session.user_id,
                status=session.status,
                started_at=session.started_at.isoformat(),
                ended_at=session.ended_at.isoformat() if session.ended_at else None,
                patterns_learned=session.patterns_learned,
                patterns_reinforced=session.patterns_reinforced,
                knowledge_units_created=session.knowledge_units_created,
                success_rate=session.success_rate,
                learning_effectiveness=session.learning_effectiveness
            )
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}/continue")
def continue_learning_session(
    session_id: UUID,
    conversation_session_id: Optional[UUID] = None,
    db: Session = Depends(get_db)
):
    """Continue an existing learning session"""
    
    try:
        persistence = LearningSessionPersistence(db)
        session = persistence.continue_learning_session(session_id, conversation_session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Learning session not found")
        
        return {"message": "Learning session continued", "session_id": str(session.id)}
        
    except Exception as e:
        logger.error(f"Error continuing learning session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}/pause")
def pause_learning_session(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Pause a learning session"""
    
    try:
        persistence = LearningSessionPersistence(db)
        success = persistence.pause_learning_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Learning session not found")
        
        return {"message": "Learning session paused"}
        
    except Exception as e:
        logger.error(f"Error pausing learning session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}/complete")
def complete_learning_session(
    session_id: UUID,
    success_rate: Optional[float] = None,
    final_summary: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Complete a learning session"""
    
    try:
        persistence = LearningSessionPersistence(db)
        success = persistence.complete_learning_session(session_id, success_rate, final_summary)
        
        if not success:
            raise HTTPException(status_code=404, detail="Learning session not found")
        
        return {"message": "Learning session completed"}
        
    except Exception as e:
        logger.error(f"Error completing learning session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-transfer")
def create_knowledge_transfer(
    request: KnowledgeTransferRequest,
    db: Session = Depends(get_db)
):
    """Create a knowledge transfer between sessions"""
    
    try:
        persistence = LearningSessionPersistence(db)
        
        transfer = persistence.transfer_knowledge_between_sessions(
            source_session_id=request.source_session_id,
            destination_session_id=request.destination_session_id,
            transfer_type=request.transfer_type,
            transfer_data=request.transfer_data,
            transfer_name=request.transfer_name
        )
        
        if not transfer:
            raise HTTPException(status_code=400, detail="Failed to create knowledge transfer")
        
        return {
            "transfer_id": str(transfer.id),
            "message": "Knowledge transfer created",
            "knowledge_units_transferred": transfer.knowledge_units_transferred
        }
        
    except Exception as e:
        logger.error(f"Error creating knowledge transfer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def get_learning_history(
    user_id: str,
    max_sessions: int = Query(20, le=100),
    session_types: Optional[List[str]] = Query(None),
    db: Session = Depends(get_db)
):
    """Get learning history for a user"""
    
    try:
        persistence = LearningSessionPersistence(db)
        history = persistence.get_session_learning_history(user_id, max_sessions, session_types)
        
        return {
            "user_id": user_id,
            "history": history,
            "sessions_count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting learning history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/cross-session")
def get_cross_session_patterns(
    user_id: str,
    pattern_types: Optional[List[str]] = Query(None),
    min_sessions: int = Query(2, ge=2),
    db: Session = Depends(get_db)
):
    """Get patterns that appear across multiple sessions"""
    
    try:
        persistence = LearningSessionPersistence(db)
        patterns = persistence.get_cross_session_patterns(user_id, pattern_types, min_sessions)
        
        return {
            "user_id": user_id,
            "cross_session_patterns": patterns,
            "patterns_count": len(patterns)
        }
        
    except Exception as e:
        logger.error(f"Error getting cross-session patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis", response_model=CrossSessionAnalysisResponse)
def analyze_cross_session_patterns(
    user_id: str,
    time_window_days: int = Query(30, ge=1, le=365),
    min_pattern_count: int = Query(3, ge=2),
    db: Session = Depends(get_db)
):
    """Analyze patterns across learning sessions"""
    
    try:
        recognizer = CrossSessionPatternRecognition(db)
        analysis = recognizer.analyze_cross_session_patterns(
            user_id, time_window_days, min_pattern_count
        )
        
        return CrossSessionAnalysisResponse(**analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing cross-session patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/recurring")
def get_recurring_patterns(
    user_id: str,
    pattern_type: Optional[str] = None,
    recurrence_threshold: int = Query(3, ge=2),
    db: Session = Depends(get_db)
):
    """Get patterns that recur across sessions"""
    
    try:
        recognizer = CrossSessionPatternRecognition(db)
        patterns = recognizer.detect_recurring_patterns(
            user_id, pattern_type, recurrence_threshold
        )
        
        return {
            "user_id": user_id,
            "recurring_patterns": patterns,
            "patterns_count": len(patterns)
        }
        
    except Exception as e:
        logger.error(f"Error getting recurring patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/clusters")
def get_pattern_clusters(
    user_id: str,
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """Get clusters of similar patterns"""
    
    try:
        recognizer = CrossSessionPatternRecognition(db)
        clusters = recognizer.identify_pattern_clusters(user_id, similarity_threshold)
        
        return {
            "user_id": user_id,
            "pattern_clusters": clusters,
            "clusters_count": len(clusters)
        }
        
    except Exception as e:
        logger.error(f"Error getting pattern clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progression")
def analyze_learning_progression(
    user_id: str,
    session_sequence: Optional[List[UUID]] = Query(None),
    db: Session = Depends(get_db)
):
    """Analyze learning progression across sessions"""
    
    try:
        recognizer = CrossSessionPatternRecognition(db)
        progression = recognizer.analyze_learning_progression(user_id, session_sequence)
        
        return {
            "user_id": user_id,
            "progression_analysis": progression
        }
        
    except Exception as e:
        logger.error(f"Error analyzing learning progression: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predictions")
def predict_learning_outcomes(
    user_id: str,
    current_session_context: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Predict learning outcomes for current session"""
    
    try:
        recognizer = CrossSessionPatternRecognition(db)
        predictions = recognizer.predict_learning_outcomes(user_id, current_session_context)
        
        return {
            "user_id": user_id,
            "predictions": predictions,
            "context": current_session_context
        }
        
    except Exception as e:
        logger.error(f"Error predicting learning outcomes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
def get_learning_session(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """Get detailed information about a learning session"""
    
    try:
        persistence = LearningSessionPersistence(db)
        session = persistence.get_learning_session_by_id(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Learning session not found")
        
        return session.to_dict()
        
    except Exception as e:
        logger.error(f"Error getting learning session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/related")
def find_related_sessions(
    session_id: UUID,
    max_age_days: int = Query(30, ge=1, le=365),
    max_results: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Find sessions related to a specific learning session"""
    
    try:
        persistence = LearningSessionPersistence(db)
        session = persistence.get_learning_session_by_id(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Learning session not found")
        
        related_sessions = persistence.find_related_learning_sessions(
            user_id=session.user_id,
            session_type=session.session_type,
            learning_context=session.learning_context,
            max_age_days=max_age_days,
            max_results=max_results
        )
        
        return {
            "session_id": str(session_id),
            "related_sessions": [s.get_learning_summary() for s in related_sessions],
            "related_count": len(related_sessions)
        }
        
    except Exception as e:
        logger.error(f"Error finding related sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/cleanup")
def cleanup_old_sessions(
    max_age_days: int = Query(90, ge=30, le=365),
    keep_successful: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Clean up old learning sessions"""
    
    try:
        persistence = LearningSessionPersistence(db)
        deleted_count = persistence.cleanup_old_sessions(max_age_days, keep_successful)
        
        return {
            "message": f"Cleaned up {deleted_count} old learning sessions",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))