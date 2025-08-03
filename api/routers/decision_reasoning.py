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
    data: Dict[str, Any] = Body(...),
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
        # Extract parameters from JSON body
        decision_title = data.get("decision_title")
        chosen_solution = data.get("chosen_solution")
        reasoning = data.get("reasoning")
        confidence = data.get("confidence", 0.8)
        alternatives_raw = data.get("alternatives", [])
        context = data.get("context", {})
        evidence = data.get("evidence")
        trade_offs = data.get("trade_offs")
        project_id = data.get("project_id")
        session_id = data.get("session_id")
        
        # Handle alternatives - convert strings to dict format if needed
        alternatives = []
        for alt in alternatives_raw:
            if isinstance(alt, str):
                # Convert simple string to dict format
                alternatives.append({
                    "solution": alt,
                    "pros": [],
                    "cons": [],
                    "reason_rejected": "Not chosen"
                })
            elif isinstance(alt, dict):
                alternatives.append(alt)
            else:
                continue
        
        # Validate required fields
        if not all([decision_title, chosen_solution, reasoning]):
            raise HTTPException(
                status_code=422,
                detail="Missing required fields: decision_title, chosen_solution, reasoning"
            )
        
        # Validate confidence range
        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(
                status_code=422,
                detail="Confidence must be between 0.0 and 1.0"
            )
        
        result = reasoning_system.record_decision(
            db, decision_title, chosen_solution, reasoning,
            alternatives, context, confidence, evidence,
            trade_offs, project_id, session_id
        )
        return result
    except HTTPException:
        raise
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


@router.get("/history")
def get_decision_history(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, le=100, description="Maximum number of decisions to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get decision history with filtering and pagination
    """
    try:
        # Get recent decisions from the reasoning system
        all_decisions = []
        
        # Collect decisions from all categories
        for cat, decisions in reasoning_system.decision_history.items():
            for decision in decisions:
                if not category or cat == category:
                    # Handle both dict and string formats
                    if isinstance(decision, dict):
                        decision_data = {
                            "category": cat,
                            "title": decision.get("title", "Untitled"),
                            "chosen_solution": decision.get("chosen_solution"),
                            "alternatives": decision.get("alternatives", []),
                            "reasoning": decision.get("reasoning", ""),
                            "confidence_score": decision.get("confidence_score", 0.5),
                            "timestamp": decision.get("timestamp", ""),
                            "outcome": decision.get("outcome", "pending"),
                            "tags": decision.get("tags", [])
                        }
                    else:
                        # Handle legacy string format
                        decision_data = {
                            "category": cat,
                            "title": str(decision),
                            "chosen_solution": str(decision),
                            "alternatives": [],
                            "reasoning": "",
                            "confidence_score": 0.5,
                            "timestamp": "",
                            "outcome": "legacy",
                            "tags": []
                        }
                    all_decisions.append(decision_data)
        
        # Sort by timestamp (newest first)
        all_decisions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Apply pagination
        paginated_decisions = all_decisions[offset:offset + limit]
        
        return {
            "decisions": paginated_decisions,
            "total": len(all_decisions),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(all_decisions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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