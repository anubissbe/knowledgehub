"""Context Compression API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.context_compression import (
    context_compression_service,
    CompressionStrategy,
    CompressionLevel,
    CompressedContext
)
from ....models import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class CompressionRequest(BaseModel):
    """Request model for context compression"""
    session_id: UUID = Field(..., description="Session to compress")
    target_tokens: int = Field(4000, ge=500, le=16000, description="Target token count")
    strategy: CompressionStrategy = Field(
        CompressionStrategy.HYBRID,
        description="Compression strategy to use"
    )
    level: CompressionLevel = Field(
        CompressionLevel.MODERATE,
        description="Compression intensity level"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "target_tokens": 4000,
                "strategy": "hybrid",
                "level": "moderate"
            }
        }


class CompressionResponse(BaseModel):
    """Response model for compressed context"""
    session_id: UUID
    compressed_context: Dict[str, Any]
    strategy_used: str
    compression_ratio: float
    token_estimate: int
    original_memory_count: int
    compressed_memory_count: int
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "compressed_context": {
                    "memories": [],
                    "summary": "Session focused on authentication implementation",
                    "key_entities": ["OAuth2", "JWT", "authentication"],
                    "important_facts": ["JWT tokens expire in 1 hour"],
                    "recent_decisions": ["Chose OAuth2 over basic auth"],
                    "token_estimate": 3500
                },
                "strategy_used": "hybrid",
                "compression_ratio": 0.65,
                "token_estimate": 3500,
                "original_memory_count": 150,
                "compressed_memory_count": 45
            }
        }


class CompressionStatsResponse(BaseModel):
    """Response model for compression statistics"""
    session_id: UUID
    original_tokens: int
    available_strategies: List[str]
    compression_estimates: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "original_tokens": 12000,
                "available_strategies": ["importance_based", "recency_weighted", "hybrid"],
                "compression_estimates": {
                    "moderate": {
                        "estimated_tokens": 4000,
                        "compression_ratio": 0.67,
                        "time_estimate": "2-3 seconds"
                    }
                },
                "recommendations": [
                    "Use hybrid strategy for balanced results",
                    "Consider moderate compression for good balance"
                ]
            }
        }


@router.post("/compress", response_model=CompressionResponse)
async def compress_context(
    request: CompressionRequest,
    db: Session = Depends(get_db)
):
    """
    Compress session context using specified strategy
    
    This endpoint applies context compression to reduce memory usage while
    preserving the most important information from a session.
    """
    try:
        # Compress the context
        compressed = await context_compression_service.compress_context(
            db=db,
            session_id=request.session_id,
            target_tokens=request.target_tokens,
            strategy=request.strategy,
            level=request.level
        )
        
        # Get original memory count for comparison
        from ...models import Memory
        original_count = db.query(MemorySystemMemory).filter_by(
            session_id=request.session_id
        ).count()
        
        return CompressionResponse(
            session_id=request.session_id,
            compressed_context=compressed.to_dict(),
            strategy_used=compressed.strategy_used,
            compression_ratio=compressed.compression_ratio,
            token_estimate=compressed.token_estimate,
            original_memory_count=original_count,
            compressed_memory_count=len(compressed.memories)
        )
        
    except ValueError as e:
        logger.error(f"Compression error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during compression: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats/{session_id}", response_model=CompressionStatsResponse)
async def get_compression_stats(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get compression statistics and recommendations for a session
    
    This endpoint provides information about the session's current state
    and recommendations for compression strategies.
    """
    try:
        from ...models import MemorySystemMemory, MemorySession
        
        # Check if session exists
        session = db.query(MemorySession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get memory count and estimate tokens
        memories = db.query(MemorySystemMemory).filter_by(session_id=session_id).all()
        if not memories:
            raise HTTPException(status_code=404, detail="No memories found in session")
        
        # Estimate original tokens
        original_tokens = sum(
            context_compression_service._estimate_memory_tokens(memory)
            for memory in memories
        )
        
        # Generate compression estimates
        compression_estimates = {}
        for level in CompressionLevel:
            estimated_ratio = {
                CompressionLevel.LIGHT: 0.15,
                CompressionLevel.MODERATE: 0.4,
                CompressionLevel.AGGRESSIVE: 0.7,
                CompressionLevel.EXTREME: 0.85
            }[level]
            
            estimated_tokens = int(original_tokens * (1 - estimated_ratio))
            compression_estimates[level.value] = {
                "estimated_tokens": estimated_tokens,
                "compression_ratio": estimated_ratio,
                "time_estimate": "1-2 seconds" if level in [CompressionLevel.LIGHT, CompressionLevel.MODERATE] else "2-5 seconds"
            }
        
        # Generate recommendations
        recommendations = []
        if original_tokens > 8000:
            recommendations.append("High token count - consider aggressive compression")
        elif original_tokens > 4000:
            recommendations.append("Use moderate compression for good balance")
        else:
            recommendations.append("Light compression may be sufficient")
        
        if len(memories) > 100:
            recommendations.append("High memory count - entity consolidation recommended")
        
        # Check for decision/error heavy sessions
        decision_count = sum(1 for m in memories if m.memory_type == "DECISION")
        error_count = sum(1 for m in memories if m.memory_type == "ERROR")
        
        if decision_count > 20:
            recommendations.append("Many decisions - hierarchical strategy recommended")
        if error_count > 10:
            recommendations.append("Many errors - importance-based strategy recommended")
        
        return CompressionStatsResponse(
            session_id=session_id,
            original_tokens=original_tokens,
            available_strategies=[strategy.value for strategy in CompressionStrategy],
            compression_estimates=compression_estimates,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting compression stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/compress/preview", response_model=Dict[str, Any])
async def preview_compression(
    request: CompressionRequest,
    db: Session = Depends(get_db)
):
    """
    Preview compression results without caching
    
    This endpoint shows what compression would look like without
    actually storing the results.
    """
    try:
        # Temporarily disable caching for preview
        original_cache_ttl = context_compression_service.cache_ttl
        context_compression_service.cache_ttl = 0
        
        # Get compression
        compressed = await context_compression_service.compress_context(
            db=db,
            session_id=request.session_id,
            target_tokens=request.target_tokens,
            strategy=request.strategy,
            level=request.level
        )
        
        # Restore cache TTL
        context_compression_service.cache_ttl = original_cache_ttl
        
        # Return preview data
        return {
            "preview": True,
            "strategy": compressed.strategy_used,
            "compression_ratio": compressed.compression_ratio,
            "token_estimate": compressed.token_estimate,
            "memory_count": len(compressed.memories),
            "summary_length": len(compressed.summary),
            "key_entities": compressed.key_entities[:5],  # Top 5 entities
            "important_facts_count": len(compressed.important_facts),
            "recent_decisions_count": len(compressed.recent_decisions),
            "context_stats": compressed.context_stats
        }
        
    except Exception as e:
        logger.error(f"Error previewing compression: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/cache/{session_id}")
async def clear_compression_cache(
    session_id: UUID,
    strategy: Optional[CompressionStrategy] = Query(None),
    level: Optional[CompressionLevel] = Query(None),
    target_tokens: Optional[int] = Query(None)
):
    """
    Clear compression cache for a session
    
    This endpoint allows clearing cached compression results,
    optionally for specific parameters.
    """
    try:
        from ...services.cache import redis_client
        
        if not redis_client.client:
            raise HTTPException(status_code=503, detail="Cache not available")
        
        # If specific parameters provided, clear that specific cache
        if strategy and level and target_tokens:
            cache_key = context_compression_service._get_cache_key(
                session_id, target_tokens, strategy, level
            )
            await redis_client.delete(cache_key)
            return {"message": "Specific cache entry cleared"}
        
        # Otherwise, clear all compression cache for this session
        pattern = f"compression:*{session_id}*"
        deleted_count = 0
        
        # Note: This is a simplified implementation
        # In production, you'd want to use Redis SCAN for better performance
        for key in await redis_client.keys(pattern):
            await redis_client.delete(key)
            deleted_count += 1
        
        return {"message": f"Cleared {deleted_count} cache entries"}
        
    except Exception as e:
        logger.error(f"Error clearing compression cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/strategies", response_model=Dict[str, Any])
async def get_compression_strategies():
    """
    Get available compression strategies and their descriptions
    
    This endpoint provides information about all available compression
    strategies and their characteristics.
    """
    return {
        "strategies": {
            "importance_based": {
                "description": "Selects memories based on importance scores",
                "best_for": "Sessions with varying importance levels",
                "characteristics": ["Preserves high-importance memories", "Good for critical information"]
            },
            "recency_weighted": {
                "description": "Prioritizes recent memories with exponential decay",
                "best_for": "Sessions where recent context is most important",
                "characteristics": ["Time-based prioritization", "Exponential decay weighting"]
            },
            "summarization": {
                "description": "Creates summaries of memory groups",
                "best_for": "Sessions with repetitive or groupable content",
                "characteristics": ["Groups similar memories", "Generates summaries"]
            },
            "entity_consolidation": {
                "description": "Consolidates information about entities",
                "best_for": "Sessions with many entity references",
                "characteristics": ["Entity-focused", "Reduces redundancy"]
            },
            "semantic_clustering": {
                "description": "Groups semantically similar memories",
                "best_for": "Sessions with thematic content",
                "characteristics": ["Semantic similarity", "Cluster representatives"]
            },
            "hierarchical": {
                "description": "Uses importance tiers for selection",
                "best_for": "Sessions with clear importance hierarchy",
                "characteristics": ["Tier-based selection", "Critical memory preservation"]
            },
            "hybrid": {
                "description": "Combines multiple strategies for optimal results",
                "best_for": "Most sessions - provides balanced compression",
                "characteristics": ["Multi-strategy approach", "Balanced results"]
            }
        },
        "levels": {
            "light": {"reduction": "10-20%", "use_case": "Minimal compression, preserve most content"},
            "moderate": {"reduction": "30-50%", "use_case": "Balanced compression, good for most use cases"},
            "aggressive": {"reduction": "60-80%", "use_case": "High compression, focus on essentials"},
            "extreme": {"reduction": "80-90%", "use_case": "Maximum compression, only critical information"}
        }
    }