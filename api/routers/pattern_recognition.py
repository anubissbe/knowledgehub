"""
Pattern Recognition API Router
Exposes endpoints for code pattern analysis and learning
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..services.pattern_recognition_engine import (
    PatternRecognitionEngine,
    CodePattern,
    PatternCategory,
    get_pattern_engine
)
from ..services.realtime_learning_pipeline import (
    get_learning_pipeline,
    StreamEvent,
    EventType
)

router = APIRouter(prefix="/api/patterns", tags=["pattern-recognition"])


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis"""
    code: str = Field(..., description="Code to analyze")
    language: str = Field("python", description="Programming language")
    session_id: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PatternFeedback(BaseModel):
    """Feedback on detected patterns"""
    pattern_id: str
    correct: bool
    user_feedback: Optional[str] = None
    suggested_pattern_type: Optional[str] = None


class LearnPatternRequest(BaseModel):
    """Request to learn a new pattern"""
    code: str
    pattern_type: str
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.get("/health")
async def health_check():
    """Check pattern recognition engine health"""
    engine = get_pattern_engine()
    return {
        "status": "healthy",
        "service": "pattern-recognition",
        "patterns_loaded": len(engine.patterns_db),
        "learned_patterns": sum(len(p) for p in engine.learned_patterns.values())
    }


@router.post("/analyze")
async def analyze_code(
    request: CodeAnalysisRequest,
    engine: PatternRecognitionEngine = Depends(get_pattern_engine),
    pipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Analyze code for patterns and anti-patterns"""
    try:
        # Analyze code
        patterns = await engine.analyze_code(request.code, request.language)
        
        # Get improvement suggestions
        suggestions = await engine.suggest_improvements(patterns)
        
        # Publish pattern detection events to real-time pipeline
        for pattern in patterns:
            event = StreamEvent(
                event_type=EventType.PATTERN_DETECTED,
                session_id=request.session_id,
                data={
                    "pattern_type": pattern.pattern_type,
                    "pattern_name": pattern.name,
                    "confidence": pattern.confidence,
                    "file_path": request.file_path,
                    "language": request.language
                },
                metadata=request.metadata
            )
            await pipeline.publish_event(event)
        
        return {
            "patterns": [p.dict() for p in patterns],
            "suggestions": suggestions,
            "summary": {
                "total_patterns": len(patterns),
                "anti_patterns": len([p for p in patterns if p.pattern_type == PatternCategory.ANTI_PATTERN]),
                "code_smells": len([p for p in patterns if p.pattern_type == PatternCategory.CODE_SMELL]),
                "best_practices": len([p for p in patterns if p.pattern_type == PatternCategory.BEST_PRACTICE]),
                "security_issues": len([p for p in patterns if p.pattern_type == PatternCategory.SECURITY_PATTERN])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    language: str = "python",
    session_id: Optional[str] = None,
    engine: PatternRecognitionEngine = Depends(get_pattern_engine)
) -> Dict[str, Any]:
    """Analyze uploaded file for patterns"""
    try:
        # Read file content
        content = await file.read()
        code = content.decode('utf-8')
        
        # Analyze
        patterns = await engine.analyze_code(code, language)
        suggestions = await engine.suggest_improvements(patterns)
        
        return {
            "file_name": file.filename,
            "patterns": [p.dict() for p in patterns],
            "suggestions": suggestions,
            "lines_analyzed": len(code.split('\n'))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn")
async def learn_pattern(
    request: LearnPatternRequest,
    engine: PatternRecognitionEngine = Depends(get_pattern_engine)
) -> Dict[str, Any]:
    """Teach the engine a new pattern"""
    try:
        # Learn the pattern
        await engine.learn_pattern(
            request.code,
            request.pattern_type,
            {
                "name": request.name,
                "description": request.description,
                **request.metadata
            }
        )
        
        # Create pattern object
        pattern = CodePattern(
            pattern_type=request.pattern_type,
            name=request.name,
            description=request.description or f"User-defined {request.pattern_type}",
            examples=[{"code": request.code}],
            confidence=0.8,
            frequency=1,
            metadata=request.metadata
        )
        
        # Add to patterns database
        engine.patterns_db[pattern.pattern_id] = pattern
        
        return {
            "success": True,
            "pattern_id": pattern.pattern_id,
            "message": f"Learned new pattern: {request.name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def provide_feedback(
    feedback: PatternFeedback,
    engine: PatternRecognitionEngine = Depends(get_pattern_engine)
) -> Dict[str, Any]:
    """Provide feedback on pattern detection accuracy"""
    try:
        # Process feedback
        await engine.evolve_patterns({
            "pattern_id": feedback.pattern_id,
            "correct": feedback.correct,
            "user_feedback": feedback.user_feedback,
            "suggested_type": feedback.suggested_pattern_type
        })
        
        return {
            "success": True,
            "message": "Feedback processed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def list_patterns(
    pattern_type: Optional[str] = None,
    min_confidence: float = 0.0,
    engine: PatternRecognitionEngine = Depends(get_pattern_engine)
) -> Dict[str, Any]:
    """List known patterns"""
    patterns = []
    
    for pattern in engine.patterns_db.values():
        if pattern.confidence >= min_confidence:
            if pattern_type is None or pattern.pattern_type == pattern_type:
                patterns.append(pattern.dict())
    
    return {
        "patterns": patterns,
        "total": len(patterns),
        "categories": list(set(p["pattern_type"] for p in patterns))
    }


@router.get("/patterns/{pattern_id}")
async def get_pattern(
    pattern_id: str,
    engine: PatternRecognitionEngine = Depends(get_pattern_engine)
) -> Dict[str, Any]:
    """Get details of a specific pattern"""
    if pattern_id not in engine.patterns_db:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    pattern = engine.patterns_db[pattern_id]
    return pattern.dict()


@router.get("/statistics")
async def get_statistics(
    engine: PatternRecognitionEngine = Depends(get_pattern_engine)
) -> Dict[str, Any]:
    """Get pattern recognition statistics"""
    stats = {
        "total_patterns": len(engine.patterns_db),
        "learned_patterns": sum(len(p) for p in engine.learned_patterns.values()),
        "pattern_types": {},
        "confidence_distribution": {
            "high": 0,
            "medium": 0,
            "low": 0
        }
    }
    
    # Count by type
    for pattern in engine.patterns_db.values():
        ptype = pattern.pattern_type
        stats["pattern_types"][ptype] = stats["pattern_types"].get(ptype, 0) + 1
        
        # Confidence distribution
        if pattern.confidence >= 0.8:
            stats["confidence_distribution"]["high"] += 1
        elif pattern.confidence >= 0.5:
            stats["confidence_distribution"]["medium"] += 1
        else:
            stats["confidence_distribution"]["low"] += 1
    
    return stats


@router.post("/batch-analyze")
async def batch_analyze(
    files: List[UploadFile] = File(...),
    language: str = "python",
    engine: PatternRecognitionEngine = Depends(get_pattern_engine)
) -> Dict[str, Any]:
    """Analyze multiple files for patterns"""
    results = []
    total_patterns = []
    
    for file in files:
        try:
            content = await file.read()
            code = content.decode('utf-8')
            
            patterns = await engine.analyze_code(code, language)
            total_patterns.extend(patterns)
            
            results.append({
                "file": file.filename,
                "patterns": len(patterns),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "file": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    # Aggregate suggestions
    all_suggestions = await engine.suggest_improvements(total_patterns)
    
    return {
        "files_analyzed": len(files),
        "results": results,
        "total_patterns": len(total_patterns),
        "aggregated_suggestions": all_suggestions
    }