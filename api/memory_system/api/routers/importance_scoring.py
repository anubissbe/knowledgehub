"""Importance scoring API router"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import time

from ...services.importance_scoring import (
    importance_scorer,
    calculate_content_importance,
    score_memories_by_importance,
    ImportanceScore,
    ImportanceFactors
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ImportanceRequest(BaseModel):
    """Request for importance scoring"""
    content: str = Field(..., description="Text content to score", min_length=1)
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    entities: Optional[List[str]] = Field(None, description="Extracted entities")
    facts: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted facts")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Historical content for analysis")


class ImportanceResponse(BaseModel):
    """Response for importance scoring"""
    total_score: float
    normalized_score: float
    confidence: float
    factor_scores: Dict[str, float]
    reasoning: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: float


class BatchImportanceRequest(BaseModel):
    """Request for batch importance scoring"""
    memories: List[Dict[str, Any]] = Field(..., description="List of memory objects to score")
    context: Optional[Dict[str, Any]] = Field(None, description="Shared context for all memories")


class BatchImportanceResponse(BaseModel):
    """Response for batch importance scoring"""
    results: List[Dict[str, Any]]
    total_memories: int
    processing_time_ms: float
    statistics: Dict[str, Any]


@router.get("/health")
async def importance_scoring_health():
    """Health check for importance scoring service"""
    return {
        "status": "healthy",
        "service": "importance_scoring",
        "features": [
            "content_importance_scoring",
            "multi_factor_analysis",
            "pattern_based_detection",
            "contextual_relevance_scoring",
            "historical_repetition_analysis",
            "entity_density_calculation",
            "temporal_proximity_detection",
            "batch_scoring",
            "confidence_calculation",
            "statistics_generation"
        ],
        "importance_factors": [factor.value for factor in ImportanceFactors],
        "factor_count": len(ImportanceFactors)
    }


@router.post("/score", response_model=ImportanceResponse)
async def score_content_importance(request: ImportanceRequest):
    """
    Calculate importance score for content.
    
    This endpoint analyzes content using multiple importance factors:
    - Explicit importance markers (important, critical, urgent)
    - User emphasis patterns (bold text, exclamation marks)
    - Technical complexity indicators
    - Decision and action urgency
    - Error severity levels
    - Contextual relevance
    - Recency and temporal proximity
    - Entity density
    - Historical repetition patterns
    """
    start_time = time.time()
    
    try:
        score = await calculate_content_importance(
            content=request.content,
            context=request.context,
            entities=request.entities,
            facts=request.facts,
            history=request.history
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert factor scores to string keys for JSON serialization
        factor_scores = {factor.value: score for factor, score in score.factor_scores.items()}
        
        return ImportanceResponse(
            total_score=score.total_score,
            normalized_score=score.normalized_score,
            confidence=score.confidence,
            factor_scores=factor_scores,
            reasoning=score.reasoning,
            metadata=score.metadata,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Importance scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Importance scoring failed: {str(e)}")


@router.post("/score-batch", response_model=BatchImportanceResponse)
async def score_memories_batch(request: BatchImportanceRequest):
    """
    Score multiple memories for importance in batch.
    
    This endpoint efficiently processes multiple memories and provides
    comparative importance scores along with statistics.
    """
    start_time = time.time()
    
    try:
        if not request.memories:
            raise HTTPException(status_code=400, detail="No memories provided for scoring")
        
        scored_results = await score_memories_by_importance(
            request.memories,
            request.context
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert results to response format
        results = []
        scores = []
        
        for memory, score in scored_results:
            factor_scores = {factor.value: score_val for factor, score_val in score.factor_scores.items()}
            
            result = {
                "memory": memory,
                "importance_score": {
                    "total_score": score.total_score,
                    "normalized_score": score.normalized_score,
                    "confidence": score.confidence,
                    "factor_scores": factor_scores,
                    "reasoning": score.reasoning,
                    "metadata": score.metadata
                }
            }
            results.append(result)
            scores.append(score)
        
        # Generate statistics
        statistics = importance_scorer.get_importance_statistics(scores)
        
        return BatchImportanceResponse(
            results=results,
            total_memories=len(request.memories),
            processing_time_ms=processing_time,
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch importance scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch importance scoring failed: {str(e)}")


@router.get("/factors")
async def get_importance_factors():
    """Get available importance factors and their weights"""
    factor_info = {}
    
    for factor in ImportanceFactors:
        weight = importance_scorer.factor_weights.get(factor, 0.0)
        factor_info[factor.value] = {
            "weight": weight,
            "description": _get_factor_description(factor)
        }
    
    return {
        "importance_factors": factor_info,
        "total_factors": len(ImportanceFactors),
        "weight_sum": sum(importance_scorer.factor_weights.values())
    }


def _get_factor_description(factor: ImportanceFactors) -> str:
    """Get human-readable description for importance factor"""
    descriptions = {
        ImportanceFactors.EXPLICITNESS: "Explicit importance markers like 'important', 'critical', 'urgent'",
        ImportanceFactors.USER_EMPHASIS: "User emphasis patterns like bold text, caps, exclamation marks",
        ImportanceFactors.TECHNICAL_COMPLEXITY: "Technical depth and complexity indicators",
        ImportanceFactors.DECISION_WEIGHT: "Decision-related content and choices made",
        ImportanceFactors.ACTION_URGENCY: "Urgency of action items and time-sensitive tasks",
        ImportanceFactors.ERROR_SEVERITY: "Severity of errors, issues, and problems",
        ImportanceFactors.CONTEXTUAL_RELEVANCE: "Relevance to current context and project",
        ImportanceFactors.RECENCY: "How recent the information is",
        ImportanceFactors.ENTITY_DENSITY: "Density of important entities in the content",
        ImportanceFactors.TEMPORAL_PROXIMITY: "Time-sensitive and temporal information",
        ImportanceFactors.REPETITION: "How often similar content appears in history",
        ImportanceFactors.FREQUENCY_PATTERN: "Frequency patterns of key terms"
    }
    return descriptions.get(factor, "Unknown factor")


class FactorAnalysisRequest(BaseModel):
    """Request for factor analysis"""
    content: str = Field(..., description="Content to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    entities: Optional[List[str]] = Field(None, description="Entities in content")


@router.post("/analyze-factors")
async def analyze_importance_factors(request: FactorAnalysisRequest):
    """
    Analyze which importance factors contribute to content scoring.
    
    This endpoint provides detailed breakdown of how each factor
    contributes to the overall importance score.
    """
    try:
        score = await calculate_content_importance(
            content=request.content,
            context=request.context,
            entities=request.entities
        )
        
        # Calculate weighted contributions
        weighted_contributions = {}
        for factor, factor_score in score.factor_scores.items():
            weight = importance_scorer.factor_weights.get(factor, 0.0)
            contribution = factor_score * weight
            weighted_contributions[factor.value] = {
                "raw_score": factor_score,
                "weight": weight,
                "weighted_contribution": contribution,
                "percentage": (contribution / score.total_score * 100) if score.total_score > 0 else 0
            }
        
        # Sort by contribution
        sorted_contributions = sorted(
            weighted_contributions.items(),
            key=lambda x: x[1]["weighted_contribution"],
            reverse=True
        )
        
        return {
            "content_length": len(request.content),
            "total_score": score.total_score,
            "normalized_score": score.normalized_score,
            "confidence": score.confidence,
            "factor_analysis": dict(sorted_contributions),
            "top_contributing_factors": [factor for factor, _ in sorted_contributions[:3]],
            "reasoning": score.reasoning,
            "metadata": score.metadata
        }
        
    except Exception as e:
        logger.error(f"Factor analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Factor analysis failed: {str(e)}")


class ContentRankingRequest(BaseModel):
    """Request for content ranking"""
    content_list: List[Dict[str, Any]] = Field(..., description="List of content items to rank")
    context: Optional[Dict[str, Any]] = Field(None, description="Shared context")


@router.post("/rank-content")
async def rank_content_by_importance(request: ContentRankingRequest):
    """
    Rank multiple pieces of content by importance.
    
    Each content item should have at minimum a 'content' field.
    Optional fields: 'entities', 'facts', 'metadata'
    """
    try:
        if not request.content_list:
            raise HTTPException(status_code=400, detail="No content provided for ranking")
        
        scored_items = []
        
        for i, item in enumerate(request.content_list):
            if 'content' not in item:
                raise HTTPException(status_code=400, detail=f"Item {i} missing 'content' field")
            
            # Merge item metadata with shared context
            item_context = {**(request.context or {}), **item.get('metadata', {})}
            
            score = await calculate_content_importance(
                content=item['content'],
                context=item_context,
                entities=item.get('entities'),
                facts=item.get('facts')
            )
            
            scored_items.append({
                "original_item": item,
                "importance_score": score.normalized_score,
                "confidence": score.confidence,
                "rank": 0  # Will be set after sorting
            })
        
        # Sort by importance score (descending)
        scored_items.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Assign ranks
        for i, item in enumerate(scored_items):
            item['rank'] = i + 1
        
        # Calculate ranking statistics
        scores = [item['importance_score'] for item in scored_items]
        
        return {
            "ranked_items": scored_items,
            "total_items": len(request.content_list),
            "statistics": {
                "highest_score": max(scores) if scores else 0,
                "lowest_score": min(scores) if scores else 0,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "high_importance_count": sum(1 for score in scores if score >= 0.8),
                "medium_importance_count": sum(1 for score in scores if 0.5 <= score < 0.8),
                "low_importance_count": sum(1 for score in scores if score < 0.5)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content ranking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content ranking failed: {str(e)}")


class WeightOptimizationRequest(BaseModel):
    """Request for weight optimization"""
    sample_content: List[Dict[str, Any]] = Field(..., description="Sample content for optimization")
    expected_rankings: Optional[List[int]] = Field(None, description="Expected rankings for samples")


@router.post("/optimize-weights")
async def suggest_weight_optimization(request: WeightOptimizationRequest):
    """
    Suggest weight optimizations based on sample content.
    
    This is an experimental endpoint that can help tune importance
    factor weights based on expected importance rankings.
    """
    try:
        if not request.sample_content:
            raise HTTPException(status_code=400, detail="No sample content provided")
        
        # Score all content with current weights
        current_scores = []
        factor_distributions = {factor: [] for factor in ImportanceFactors}
        
        for item in request.sample_content:
            score = await calculate_content_importance(
                content=item.get('content', ''),
                context=item.get('context'),
                entities=item.get('entities'),
                facts=item.get('facts')
            )
            
            current_scores.append(score.normalized_score)
            
            # Collect factor scores for analysis
            for factor, factor_score in score.factor_scores.items():
                factor_distributions[factor].append(factor_score)
        
        # Analyze factor variance and effectiveness
        factor_analysis = {}
        for factor in ImportanceFactors:
            scores = factor_distributions[factor]
            if scores:
                variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
                factor_analysis[factor.value] = {
                    "current_weight": importance_scorer.factor_weights.get(factor, 0.0),
                    "average_score": sum(scores) / len(scores),
                    "variance": variance,
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "effectiveness": variance * (sum(scores) / len(scores))  # Simple effectiveness metric
                }
        
        # Sort factors by effectiveness
        sorted_factors = sorted(
            factor_analysis.items(),
            key=lambda x: x[1]["effectiveness"],
            reverse=True
        )
        
        return {
            "current_performance": {
                "average_score": sum(current_scores) / len(current_scores),
                "score_variance": sum((s - sum(current_scores)/len(current_scores))**2 for s in current_scores) / len(current_scores),
                "high_importance_count": sum(1 for s in current_scores if s >= 0.8)
            },
            "factor_analysis": dict(sorted_factors),
            "recommendations": {
                "most_effective_factors": [factor for factor, _ in sorted_factors[:3]],
                "least_effective_factors": [factor for factor, _ in sorted_factors[-3:]],
                "note": "Consider increasing weights for most effective factors and decreasing for least effective"
            },
            "sample_count": len(request.sample_content)
        }
        
    except Exception as e:
        logger.error(f"Weight optimization analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Weight optimization analysis failed: {str(e)}")