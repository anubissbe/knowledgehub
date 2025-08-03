"""
Enhanced Decision Recording and Analysis API Router.

This router provides comprehensive endpoints for decision tracking, analysis,
visualization, and AI-powered recommendations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session as DBSession

from ..models.base import get_db
from ..models.enhanced_decision import (
    DecisionCreate, AlternativeCreate, OutcomeCreate, FeedbackCreate,
    DecisionResponse, DecisionTreeNode, DecisionAnalytics,
    DecisionType, DecisionStatus, OutcomeStatus, ImpactLevel
)
from ..services.decision_service import decision_service, DecisionRecommendation, DecisionImpact
from ..analytics.decision_analyzer import decision_analyzer, DecisionTrend, DecisionInsight
from shared.logging import setup_logging

logger = setup_logging("enhanced_decisions_router")

router = APIRouter(prefix="/api/decisions", tags=["decision-recording"])


@router.post("/record", response_model=DecisionResponse)
async def record_decision(
    decision_data: DecisionCreate,
    user_id: str = Query(..., description="User ID"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    project_id: Optional[str] = Query(None, description="Project ID"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Record a new decision with alternatives and reasoning.
    
    Features:
    - Comprehensive decision tracking
    - Alternative evaluation
    - Pattern recognition
    - Success prediction
    - Automatic pattern mining
    """
    try:
        decision = await decision_service.record_decision(
            decision_data=decision_data,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id
        )
        
        # Schedule background pattern analysis
        background_tasks.add_task(_analyze_decision_patterns, str(decision.id))
        
        logger.info(f"Recorded decision {decision.id} for user {user_id}")
        
        # Convert to response format
        return DecisionResponse(
            id=str(decision.id),
            decision_type=decision.decision_type,
            category=decision.category,
            title=decision.title,
            description=decision.description,
            chosen_option=decision.chosen_option,
            reasoning=decision.reasoning,
            confidence_score=decision.confidence_score,
            status=decision.status,
            impact_level=decision.impact_level,
            created_at=decision.created_at,
            decided_at=decision.decided_at,
            alternatives_count=len(decision.alternatives),
            has_outcome=decision.outcome is not None,
            success_metrics=decision.calculate_success_metrics() if decision.outcome else None
        )
        
    except Exception as e:
        logger.error(f"Failed to record decision: {e}")
        raise HTTPException(status_code=500, detail="Failed to record decision")


@router.post("/recommend", response_model=Dict[str, Any])
async def get_decision_recommendations(
    decision_type: DecisionType,
    context: Dict[str, Any] = Body(..., description="Decision context"),
    constraints: Dict[str, Any] = Body(default={}, description="Decision constraints"),
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    project_id: Optional[str] = Query(None, description="Project ID")
):
    """
    Get AI-powered recommendations for a decision.
    
    Analyzes historical decisions to provide:
    - Recommended option
    - Success probability
    - Estimated effort
    - Best practices
    - Risk factors
    """
    try:
        recommendation = await decision_service.get_recommendations(
            decision_type=decision_type,
            context=context,
            constraints=constraints,
            user_id=user_id,
            project_id=project_id
        )
        
        return {
            "recommendation": {
                "recommended_option": recommendation.recommended_option,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "success_probability": recommendation.success_probability,
                "estimated_effort_hours": recommendation.estimated_effort,
                "risks": recommendation.risks,
                "best_practices": recommendation.best_practices
            },
            "similar_decisions": [
                {
                    "id": str(d.id),
                    "title": d.title,
                    "chosen_option": d.chosen_option,
                    "confidence_score": d.confidence_score,
                    "outcome_status": d.outcome.status if d.outcome else None
                }
                for d in recommendation.similar_decisions
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


@router.post("/outcomes", response_model=Dict[str, str])
async def record_decision_outcome(
    outcome_data: OutcomeCreate,
    validated_by: Optional[str] = Query(None, description="Validator ID")
):
    """
    Record the outcome of a decision.
    
    Updates:
    - Decision status
    - Success metrics
    - Pattern statistics
    - Learning data
    """
    try:
        outcome = await decision_service.record_outcome(
            outcome_data=outcome_data,
            validated_by=validated_by
        )
        
        logger.info(f"Recorded outcome for decision {outcome_data.decision_id}")
        
        return {
            "outcome_id": str(outcome.id),
            "status": "recorded",
            "overall_success": str(outcome.calculate_overall_success())
        }
        
    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")
        raise HTTPException(status_code=500, detail="Failed to record outcome")


@router.post("/feedback", response_model=Dict[str, str])
async def provide_decision_feedback(
    feedback_data: FeedbackCreate,
    user_id: str = Query(..., description="User ID")
):
    """
    Provide feedback on a decision.
    
    Captures:
    - Effectiveness ratings
    - Implementation feedback
    - Improvement suggestions
    - Alternative approaches
    """
    try:
        from ..models.enhanced_decision import EnhancedDecisionFeedback
        
        db = next(get_db())
        
        feedback = EnhancedDecisionFeedback(
            decision_id=UUID(feedback_data.decision_id),
            user_id=user_id,
            feedback_type=feedback_data.feedback_type,
            rating=feedback_data.rating,
            comment=feedback_data.comment,
            effectiveness_rating=feedback_data.effectiveness_rating,
            implementation_rating=feedback_data.implementation_rating,
            maintainability_rating=feedback_data.maintainability_rating,
            improvement_suggestions=feedback_data.improvement_suggestions,
            alternative_approach=feedback_data.alternative_approach
        )
        
        db.add(feedback)
        db.commit()
        
        logger.info(f"Recorded feedback for decision {feedback_data.decision_id}")
        
        return {"feedback_id": str(feedback.id), "status": "recorded"}
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")


@router.get("/tree/{decision_id}", response_model=DecisionTreeNode)
async def get_decision_tree(
    decision_id: str = Path(..., description="Root decision ID"),
    max_depth: int = Query(5, description="Maximum tree depth", ge=1, le=10)
):
    """
    Get decision tree visualization data.
    
    Returns hierarchical structure showing:
    - Parent-child relationships
    - Decision outcomes
    - Success ratings
    - Impact levels
    """
    try:
        tree = await decision_service.get_decision_tree(
            root_decision_id=decision_id,
            max_depth=max_depth
        )
        
        return tree
        
    except Exception as e:
        logger.error(f"Failed to get decision tree: {e}")
        raise HTTPException(status_code=500, detail="Failed to get decision tree")


@router.get("/impact/{decision_id}", response_model=Dict[str, Any])
async def analyze_decision_impact(
    decision_id: str = Path(..., description="Decision ID")
):
    """
    Analyze the impact of a decision.
    
    Provides:
    - Affected components
    - Downstream decisions
    - Risk assessment
    - Scope estimation
    """
    try:
        impact = await decision_service.analyze_decision_impact(decision_id)
        
        return {
            "direct_components": impact.direct_components,
            "indirect_components": impact.indirect_components,
            "downstream_decisions_count": len(impact.downstream_decisions),
            "downstream_decisions": [
                {
                    "id": str(d.id),
                    "title": d.title,
                    "decision_type": d.decision_type,
                    "impact_level": d.impact_level
                }
                for d in impact.downstream_decisions[:10]  # Limit to 10
            ],
            "estimated_scope": impact.estimated_scope,
            "risk_assessment": impact.risk_assessment
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze impact: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze decision impact")


@router.get("/analytics", response_model=DecisionAnalytics)
async def get_decision_analytics(
    user_id: Optional[str] = Query(None, description="User ID filter"),
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    time_window_days: int = Query(30, description="Time window in days", ge=1, le=365)
):
    """
    Get comprehensive decision analytics.
    
    Includes:
    - Decision statistics
    - Success metrics
    - Pattern analysis
    - Trend data
    - Impact distribution
    """
    try:
        analytics = await decision_service.get_decision_analytics(
            user_id=user_id,
            project_id=project_id,
            time_window_days=time_window_days
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get decision analytics")


@router.get("/trends", response_model=List[Dict[str, Any]])
async def analyze_decision_trends(
    decision_type: Optional[DecisionType] = Query(None, description="Filter by type"),
    user_id: Optional[str] = Query(None, description="User ID filter"),
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    period: str = Query("daily", description="Analysis period", pattern="^(daily|weekly|monthly)$"),
    window_days: int = Query(30, description="Time window", ge=7, le=180)
):
    """
    Analyze decision trends over time.
    
    Tracks:
    - Volume trends
    - Confidence trends
    - Success rate trends
    - Impact distribution trends
    """
    try:
        trends = decision_analyzer.analyze_trends(
            decision_type=decision_type,
            user_id=user_id,
            project_id=project_id,
            period=period,
            window_days=window_days
        )
        
        return [
            {
                "period": trend.period,
                "trend_type": trend.trend_type,
                "values": trend.values,
                "timestamps": [t.isoformat() for t in trend.timestamps],
                "direction": trend.direction,
                "change_percentage": trend.change_percentage
            }
            for trend in trends
        ]
        
    except Exception as e:
        logger.error(f"Failed to analyze trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze trends")


@router.get("/success-factors", response_model=Dict[str, Any])
async def identify_success_factors(
    decision_type: Optional[DecisionType] = Query(None, description="Filter by type"),
    min_decisions: int = Query(10, description="Minimum decisions for analysis", ge=5, le=100)
):
    """
    Identify factors that contribute to decision success.
    
    Analyzes:
    - Confidence correlation
    - Impact level effects
    - Context patterns
    - Timing factors
    - Alternative evaluation impact
    """
    try:
        factors = decision_analyzer.identify_success_factors(
            decision_type=decision_type,
            min_decisions=min_decisions
        )
        
        return factors
        
    except Exception as e:
        logger.error(f"Failed to identify success factors: {e}")
        raise HTTPException(status_code=500, detail="Failed to identify success factors")


@router.get("/insights", response_model=List[Dict[str, Any]])
async def generate_decision_insights(
    user_id: Optional[str] = Query(None, description="User ID filter"),
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    lookback_days: int = Query(30, description="Days to analyze", ge=7, le=90)
):
    """
    Generate actionable insights from decision history.
    
    Provides:
    - Warnings about patterns
    - Improvement opportunities
    - Recommendations
    - Action items
    """
    try:
        insights = decision_analyzer.generate_insights(
            user_id=user_id,
            project_id=project_id,
            lookback_days=lookback_days
        )
        
        return [
            {
                "insight_type": insight.insight_type,
                "title": insight.title,
                "description": insight.description,
                "confidence": insight.confidence,
                "supporting_data": insight.supporting_data,
                "recommended_actions": insight.recommended_actions
            }
            for insight in insights
        ]
        
    except Exception as e:
        logger.error(f"Failed to generate insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate insights")


@router.post("/predict", response_model=Dict[str, Any])
async def predict_decision_outcome(
    decision_type: DecisionType,
    confidence_score: float = Body(..., ge=0.0, le=1.0),
    impact_level: ImpactLevel = Body(...),
    context: Dict[str, Any] = Body(default={}),
    alternatives_count: int = Body(0, ge=0, le=20)
):
    """
    Predict the likely outcome of a decision.
    
    Based on:
    - Historical patterns
    - Similar decisions
    - Success factors
    - Context analysis
    """
    try:
        prediction = decision_analyzer.predict_decision_outcome(
            decision_type=decision_type,
            confidence_score=confidence_score,
            impact_level=impact_level,
            context=context,
            alternatives_count=alternatives_count
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Failed to predict outcome: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict decision outcome")


@router.get("/search", response_model=List[DecisionResponse])
async def search_decisions(
    query: Optional[str] = Query(None, description="Search query"),
    decision_type: Optional[DecisionType] = Query(None, description="Filter by type"),
    status: Optional[DecisionStatus] = Query(None, description="Filter by status"),
    impact_level: Optional[ImpactLevel] = Query(None, description="Filter by impact"),
    user_id: Optional[str] = Query(None, description="Filter by user"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    has_outcome: Optional[bool] = Query(None, description="Filter by outcome presence"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: DBSession = Depends(get_db)
):
    """
    Search and filter decisions.
    
    Supports:
    - Text search
    - Multiple filters
    - Pagination
    - Sorting by relevance
    """
    try:
        from ..models.enhanced_decision import EnhancedDecision
        
        # Build query
        query_obj = db.query(EnhancedDecision)
        
        # Apply filters
        if query:
            # Simple text search - in production, use full-text search
            query_obj = query_obj.filter(
                (EnhancedDecision.title.ilike(f"%{query}%")) |
                (EnhancedDecision.reasoning.ilike(f"%{query}%"))
            )
        
        if decision_type:
            query_obj = query_obj.filter(EnhancedDecision.decision_type == decision_type.value)
        if status:
            query_obj = query_obj.filter(EnhancedDecision.status == status.value)
        if impact_level:
            query_obj = query_obj.filter(EnhancedDecision.impact_level == impact_level.value)
        if user_id:
            query_obj = query_obj.filter(EnhancedDecision.user_id == user_id)
        if project_id:
            query_obj = query_obj.filter(EnhancedDecision.project_id == project_id)
        
        query_obj = query_obj.filter(EnhancedDecision.confidence_score >= min_confidence)
        
        if has_outcome is not None:
            if has_outcome:
                query_obj = query_obj.filter(EnhancedDecision.outcome.isnot(None))
            else:
                query_obj = query_obj.filter(EnhancedDecision.outcome.is_(None))
        
        # Order by creation date desc
        query_obj = query_obj.order_by(EnhancedDecision.created_at.desc())
        
        # Apply pagination
        decisions = query_obj.offset(offset).limit(limit).all()
        
        # Convert to response format
        return [
            DecisionResponse(
                id=str(decision.id),
                decision_type=decision.decision_type,
                category=decision.category,
                title=decision.title,
                description=decision.description,
                chosen_option=decision.chosen_option,
                reasoning=decision.reasoning,
                confidence_score=decision.confidence_score,
                status=decision.status,
                impact_level=decision.impact_level,
                created_at=decision.created_at,
                decided_at=decision.decided_at,
                alternatives_count=len(decision.alternatives),
                has_outcome=decision.outcome is not None,
                success_metrics=decision.calculate_success_metrics() if decision.outcome else None
            )
            for decision in decisions
        ]
        
    except Exception as e:
        logger.error(f"Failed to search decisions: {e}")
        raise HTTPException(status_code=500, detail="Failed to search decisions")


@router.get("/patterns", response_model=List[Dict[str, Any]])
async def list_decision_patterns(
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    min_occurrences: int = Query(3, description="Minimum occurrences", ge=1, le=100),
    min_success_rate: float = Query(0.6, description="Minimum success rate", ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
    db: DBSession = Depends(get_db)
):
    """
    List learned decision patterns.
    
    Shows:
    - Common decision patterns
    - Success rates
    - Best practices
    - Recommendations
    """
    try:
        from ..models.enhanced_decision import DecisionPattern
        from sqlalchemy import desc
        
        # Build query
        query = db.query(DecisionPattern)
        
        if pattern_type:
            query = query.filter(DecisionPattern.pattern_type == pattern_type)
        
        query = query.filter(
            DecisionPattern.occurrence_count >= min_occurrences,
            DecisionPattern.success_rate >= min_success_rate
        )
        
        # Order by occurrence count and success rate
        patterns = query.order_by(
            desc(DecisionPattern.occurrence_count),
            desc(DecisionPattern.success_rate)
        ).limit(limit).all()
        
        return [
            {
                "pattern_id": str(pattern.id),
                "pattern_type": pattern.pattern_type,
                "pattern_name": pattern.pattern_name,
                "description": pattern.description,
                "occurrence_count": pattern.occurrence_count,
                "success_rate": pattern.success_rate,
                "avg_confidence": pattern.avg_confidence,
                "avg_implementation_time_hours": pattern.avg_implementation_time,
                "recommended_approach": pattern.recommended_approach,
                "best_practices": pattern.best_practices,
                "common_pitfalls": pattern.common_pitfalls,
                "first_seen": pattern.first_seen.isoformat(),
                "last_seen": pattern.last_seen.isoformat()
            }
            for pattern in patterns
        ]
        
    except Exception as e:
        logger.error(f"Failed to list patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to list decision patterns")


# Background task functions

async def _analyze_decision_patterns(decision_id: str):
    """Background task to analyze decision patterns."""
    try:
        # Perform additional pattern analysis
        logger.debug(f"Analyzing patterns for decision {decision_id}")
        # Pattern mining happens in the service
        
    except Exception as e:
        logger.warning(f"Background pattern analysis failed: {e}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Decision recording service health check."""
    try:
        # Basic service health check
        if not decision_service._initialized:
            await decision_service.initialize()
        
        return {
            "status": "healthy",
            "service": "decision_recording",
            "initialized": decision_service._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "decision_recording",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )