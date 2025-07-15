"""
Decision Reasoning API - Track decisions, alternatives, and reasoning
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.decision_reasoning_system import DecisionReasoningSystem

router = APIRouter(prefix="/api/decisions", tags=["decisions"])

# Global decision reasoning system
reasoning_system = DecisionReasoningSystem()


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "decision-reasoning",
        "description": "Track decisions, alternatives, and reasoning with confidence"
    }


@router.post("/record")
def record_decision(
    decision_title: str = Query(..., description="Title of the decision"),
    chosen_solution: str = Query(..., description="What was ultimately chosen"),
    reasoning: str = Query(..., description="Why this solution was chosen"),
    confidence: float = Query(..., ge=0.0, le=1.0, description="Confidence level (0-1)"),
    alternatives: List[Dict[str, Any]] = Body(..., description="Alternative solutions considered"),
    context: Dict[str, Any] = Body({}, description="Context when decision was made"),
    evidence: Optional[List[str]] = Body(None, description="Evidence supporting the decision"),
    trade_offs: Optional[Dict[str, Any]] = Body(None, description="Trade-offs analyzed"),
    project_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Record a decision with full reasoning, alternatives, and confidence
    
    alternatives format:
    [
      {
        "solution": "Alternative 1",
        "pros": ["Pro 1", "Pro 2"],
        "cons": ["Con 1", "Con 2"],
        "reason_rejected": "Why this wasn't chosen"
      }
    ]
    """
    try:
        result = reasoning_system.record_decision(
            db, decision_title, chosen_solution, reasoning,
            alternatives, context, confidence, evidence,
            trade_offs, project_id, session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explain/{decision_id}")
def explain_decision(
    decision_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get full explanation of a past decision
    
    Returns complete reasoning, alternatives, evidence, and outcome
    """
    explanation = reasoning_system.explain_decision(db, decision_id)
    
    if not explanation:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    return explanation


@router.post("/update-outcome")
def update_outcome(
    decision_id: str = Query(..., description="Decision ID to update"),
    outcome: str = Query(..., description="Actual outcome (successful/failed/mixed)"),
    impact: Dict[str, Any] = Body(..., description="Measured impact of the decision"),
    lessons_learned: Optional[str] = Body(None, description="What was learned"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Update a decision with its actual outcome and impact
    
    This helps improve future confidence calibration
    """
    try:
        result = reasoning_system.update_decision_outcome(
            db, decision_id, outcome, impact, lessons_learned
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar")
def find_similar_decisions(
    category: str = Query(..., description="Decision category"),
    keywords: List[str] = Query(..., description="Keywords to match"),
    limit: int = Query(5, le=20, description="Maximum results"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Find similar past decisions for reference
    """
    try:
        results = reasoning_system.find_similar_decisions(db, category, keywords, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])


@router.get("/suggest")
def suggest_decision(
    problem: str = Query(..., description="Problem description"),
    context: Dict[str, Any] = Body({}, description="Current context"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get decision suggestion based on past experience
    
    Returns suggested approach with confidence and reasoning
    """
    try:
        suggestion = reasoning_system.suggest_decision(db, problem, context)
        return suggestion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confidence-report")
def get_confidence_report(
    category: Optional[str] = Query(None, description="Filter by category"),
) -> Dict[str, Any]:
    """
    Get report on confidence accuracy over time
    
    Shows how well-calibrated confidence scores are
    """
    try:
        report = reasoning_system.get_confidence_report(category)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
def get_decision_categories() -> Dict[str, List[str]]:
    """Get all decision categories and their keywords"""
    return reasoning_system.decision_categories


@router.get("/patterns/{category}")
def get_reasoning_patterns(category: str) -> Dict[str, str]:
    """Get learned reasoning patterns for a category"""
    patterns = reasoning_system.reasoning_patterns.get(category, {})
    return patterns


@router.get("/search")
def search_decisions(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    outcome: Optional[str] = Query(None, description="Filter by outcome"),
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Search through past decisions
    """
    try:
        # Basic implementation - can be enhanced
        keywords = query.split()
        category_filter = category or "architecture"  # Default category
        
        results = reasoning_system.find_similar_decisions(
            db, category_filter, keywords, limit
        )
        
        # Apply additional filters
        if min_confidence:
            results = [r for r in results if r["confidence"] >= min_confidence]
        
        if outcome:
            results = [r for r in results if r.get("outcome") == outcome]
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=[])