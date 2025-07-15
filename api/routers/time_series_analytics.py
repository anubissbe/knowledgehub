"""
Time-Series Analytics API router.

Provides endpoints for time-series data management, trend analysis,
and temporal pattern detection.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from ..services.time_series_analytics import (
    TimeSeriesAnalyticsService,
    MetricType,
    AggregationWindow,
    TrendAnalysis
)
from ..dependencies import get_current_user
from ...shared.logging import setup_logging

logger = setup_logging("api.time_series_analytics")

router = APIRouter(prefix="/api/time-series", tags=["time-series"])


class RecordMetricRequest(BaseModel):
    """Request to record a metric"""
    metric_type: str = Field(..., description="Type of metric")
    value: float = Field(..., description="Metric value")
    tags: Optional[Dict[str, str]] = Field(None, description="Optional tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    timestamp: Optional[datetime] = Field(None, description="Optional timestamp")


class KnowledgeEvolutionRequest(BaseModel):
    """Request to record knowledge evolution"""
    entity_type: str = Field(..., description="Type of entity")
    entity_id: str = Field(..., description="Entity ID")
    change_type: str = Field(..., description="Type of change")
    old_value: Optional[Dict[str, Any]] = Field(None, description="Previous value")
    new_value: Optional[Dict[str, Any]] = Field(None, description="New value")
    confidence: float = Field(0.0, description="Confidence score")
    impact_score: float = Field(0.0, description="Impact score")


class PatternTrendRequest(BaseModel):
    """Request to record pattern trend"""
    pattern_type: str = Field(..., description="Pattern type")
    pattern_name: str = Field(..., description="Pattern name")
    occurrence_count: int = Field(1, description="Number of occurrences")
    effectiveness_score: float = Field(0.0, description="Effectiveness score")
    context: Optional[Dict[str, Any]] = Field(None, description="Context data")


class TrendAnalysisResponse(BaseModel):
    """Response for trend analysis"""
    trend_direction: str = Field(..., description="Trend direction")
    trend_strength: float = Field(..., description="Trend strength")
    change_rate: float = Field(..., description="Change rate percentage")
    correlation_score: float = Field(..., description="Correlation score")
    seasonal_patterns: List[Dict[str, Any]] = Field(..., description="Seasonal patterns")
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")


class MetricTrendsResponse(BaseModel):
    """Response for metric trends"""
    trends: List[Dict[str, Any]] = Field(..., description="Trend data points")
    total_points: int = Field(..., description="Total data points")
    time_range: Dict[str, str] = Field(..., description="Time range")


class AnalyticsDashboardResponse(BaseModel):
    """Response for analytics dashboard"""
    overview: Dict[str, Any] = Field(..., description="Overview metrics")
    trends: Dict[str, Any] = Field(..., description="Trend data")
    patterns: Dict[str, Any] = Field(..., description="Pattern data")
    evolution: List[Dict[str, Any]] = Field(..., description="Evolution timeline")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")


# Initialize service
time_series_service = None


async def get_time_series_service():
    """Get or create time-series analytics service instance"""
    global time_series_service
    if time_series_service is None:
        time_series_service = TimeSeriesAnalyticsService()
        await time_series_service.initialize()
    return time_series_service


@router.post("/metrics")
async def record_metric(
    request: RecordMetricRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """
    Record a time-series metric.
    
    Metric types: knowledge_creation, decision_making, error_rate, solution_effectiveness,
    pattern_evolution, user_engagement, code_quality, learning_progress, performance, usage
    """
    try:
        # Validate metric type
        try:
            metric_type = MetricType(request.metric_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric type: {request.metric_type}"
            )
        
        # Record metric
        success = await service.record_metric(
            metric_type=metric_type,
            value=request.value,
            tags=request.tags,
            metadata=request.metadata,
            user_id=current_user['id'],
            timestamp=request.timestamp
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record metric")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording metric: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-evolution")
async def record_knowledge_evolution(
    request: KnowledgeEvolutionRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """Record knowledge evolution event."""
    try:
        success = await service.record_knowledge_evolution(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            change_type=request.change_type,
            old_value=request.old_value,
            new_value=request.new_value,
            confidence=request.confidence,
            impact_score=request.impact_score,
            user_id=current_user['id']
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record evolution")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording evolution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pattern-trends")
async def record_pattern_trend(
    request: PatternTrendRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """Record pattern trend data."""
    try:
        success = await service.record_pattern_trend(
            pattern_type=request.pattern_type,
            pattern_name=request.pattern_name,
            occurrence_count=request.occurrence_count,
            effectiveness_score=request.effectiveness_score,
            context=request.context
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record pattern trend")
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording pattern trend: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{metric_type}/trends", response_model=MetricTrendsResponse)
async def get_metric_trends(
    metric_type: str,
    start_time: datetime = Query(..., description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    window: str = Query("1 hour", description="Aggregation window"),
    tags: Optional[str] = Query(None, description="Tags filter (JSON)"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """Get metric trends over time."""
    try:
        # Validate metric type
        try:
            metric_type_enum = MetricType(metric_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric type: {metric_type}"
            )
        
        # Validate window
        try:
            window_enum = AggregationWindow(window)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid window: {window}"
            )
        
        # Parse tags filter
        tags_filter = None
        if tags:
            import json
            try:
                tags_filter = json.loads(tags)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid tags JSON")
        
        # Get trends
        trends = await service.get_metric_trends(
            metric_type=metric_type_enum,
            start_time=start_time,
            end_time=end_time,
            window=window_enum,
            tags_filter=tags_filter
        )
        
        return MetricTrendsResponse(
            trends=trends,
            total_points=len(trends),
            time_range={
                "start": start_time.isoformat(),
                "end": (end_time or datetime.utcnow()).isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{metric_type}/analysis", response_model=TrendAnalysisResponse)
async def analyze_metric_trends(
    metric_type: str,
    start_time: datetime = Query(..., description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    window: str = Query("1 day", description="Aggregation window"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """Analyze trends in metric data."""
    try:
        # Validate inputs
        try:
            metric_type_enum = MetricType(metric_type)
            window_enum = AggregationWindow(window)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Analyze trends
        analysis = await service.analyze_trends(
            metric_type=metric_type_enum,
            start_time=start_time,
            end_time=end_time,
            window=window_enum
        )
        
        return TrendAnalysisResponse(
            trend_direction=analysis.trend_direction,
            trend_strength=analysis.trend_strength,
            change_rate=analysis.change_rate,
            correlation_score=analysis.correlation_score,
            seasonal_patterns=analysis.seasonal_patterns,
            anomalies=analysis.anomalies
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-evolution")
async def get_knowledge_evolution_timeline(
    entity_type: Optional[str] = Query(None, description="Entity type filter"),
    entity_id: Optional[str] = Query(None, description="Entity ID filter"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """Get knowledge evolution timeline."""
    try:
        evolution = await service.get_knowledge_evolution_timeline(
            entity_type=entity_type,
            entity_id=entity_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return {
            "evolution": evolution,
            "total": len(evolution),
            "filters": {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting evolution timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/effectiveness")
async def get_pattern_effectiveness_trends(
    pattern_type: Optional[str] = Query(None, description="Pattern type filter"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    window: str = Query("1 day", description="Aggregation window"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """Get pattern effectiveness trends."""
    try:
        # Validate window
        try:
            window_enum = AggregationWindow(window)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid window: {window}")
        
        trends = await service.get_pattern_effectiveness_trends(
            pattern_type=pattern_type,
            start_time=start_time,
            end_time=end_time,
            window=window_enum
        )
        
        return {
            "trends": trends,
            "total_patterns": len(trends),
            "time_range": {
                "start": (start_time or (datetime.utcnow() - timedelta(days=30))).isoformat(),
                "end": (end_time or datetime.utcnow()).isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pattern trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard", response_model=AnalyticsDashboardResponse)
async def get_analytics_dashboard(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """Get analytics dashboard data."""
    try:
        dashboard_data = await service.generate_analytics_dashboard_data()
        
        return AnalyticsDashboardResponse(**dashboard_data)
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/types")
async def get_metric_types(
    current_user: dict = Depends(get_current_user)
):
    """Get available metric types."""
    return {
        "metric_types": [mt.value for mt in MetricType],
        "aggregation_windows": [aw.value for aw in AggregationWindow]
    }


@router.get("/health")
async def health_check(
    service: TimeSeriesAnalyticsService = Depends(get_time_series_service)
):
    """Health check for time-series service."""
    try:
        # Simple query to check database connectivity
        async with service.async_engine.begin() as conn:
            await conn.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )