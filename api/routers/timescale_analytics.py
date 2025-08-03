"""
Time-Series Analytics API router.

Provides endpoints for time-series data analytics, trend analysis,
and performance monitoring using TimescaleDB.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel, Field
import uuid

from api.services.time_series_analytics import (
    TimeSeriesAnalyticsService,
    MetricType,
    AggregationWindow,
    TimeSeriesPoint,
    TrendAnalysis
)
from api.dependencies import get_current_user
from shared.logging import setup_logging

logger = setup_logging("api.time_series_analytics")

router = APIRouter(prefix="/api/analytics", tags=["time-series-analytics"])


class MetricRecordRequest(BaseModel):
    """Request to record a metric"""
    metric_type: str = Field(..., description="Type of metric to record")
    value: float = Field(..., description="Metric value")
    tags: Optional[Dict[str, str]] = Field(None, description="Optional tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    project_id: Optional[str] = Field(None, description="Project ID")
    timestamp: Optional[datetime] = Field(None, description="Timestamp (defaults to now)")


class KnowledgeEvolutionRequest(BaseModel):
    """Request to record knowledge evolution"""
    entity_type: str = Field(..., description="Type of entity")
    entity_id: str = Field(..., description="Entity ID")
    change_type: str = Field(..., description="Type of change")
    old_value: Optional[Dict[str, Any]] = Field(None, description="Previous value")
    new_value: Optional[Dict[str, Any]] = Field(None, description="New value")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    impact_score: float = Field(0.0, ge=0.0, le=1.0, description="Impact score")


class PatternTrendRequest(BaseModel):
    """Request to record pattern trend"""
    pattern_type: str = Field(..., description="Type of pattern")
    pattern_name: str = Field(..., description="Pattern name")
    occurrence_count: int = Field(1, ge=1, description="Number of occurrences")
    effectiveness_score: float = Field(0.0, ge=0.0, le=1.0, description="Effectiveness score")
    context: Optional[Dict[str, Any]] = Field(None, description="Pattern context")
    project_id: Optional[str] = Field(None, description="Project ID")


class PerformanceRecordRequest(BaseModel):
    """Request to record performance metrics"""
    endpoint: str = Field(..., description="API endpoint")
    response_time: float = Field(..., ge=0.0, description="Response time in seconds")
    memory_usage: Optional[float] = Field(None, ge=0.0, description="Memory usage in MB")
    cpu_usage: Optional[float] = Field(None, ge=0.0, le=100.0, description="CPU usage percentage")
    error_count: int = Field(0, ge=0, description="Number of errors")
    request_count: int = Field(1, ge=1, description="Number of requests")
    user_count: int = Field(0, ge=0, description="Number of unique users")


class TrendAnalysisResponse(BaseModel):
    """Response for trend analysis"""
    trend_direction: str = Field(..., description="Trend direction")
    trend_strength: float = Field(..., description="Trend strength (0-1)")
    change_rate: float = Field(..., description="Change rate percentage")
    correlation_score: float = Field(..., description="Correlation score")
    seasonal_patterns: List[Dict[str, Any]] = Field(..., description="Seasonal patterns")
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")


# Initialize service
analytics_service = None


async def get_analytics_service():
    """Get or create analytics service instance"""
    global analytics_service
    if analytics_service is None:
        logger.info("Creating new TimeSeriesAnalyticsService instance")
        analytics_service = TimeSeriesAnalyticsService()
        logger.info("Initializing TimeSeriesAnalyticsService...")
        await analytics_service.initialize()
        logger.info("TimeSeriesAnalyticsService initialized")
    return analytics_service


@router.post("/metrics", response_model=Dict[str, str])
async def record_metric(
    request: MetricRecordRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Record a time-series metric."""
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
            metric_type,
            request.value,
            request.tags,
            request.metadata,
            current_user['id'],
            None,  # session_id can be added later
            request.project_id,
            request.timestamp
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to record metric"
            )
        
        return {"status": "recorded"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording metric: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolution", response_model=Dict[str, str])
async def record_evolution(
    request: KnowledgeEvolutionRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Record a knowledge evolution event."""
    try:
        success = await service.record_knowledge_evolution(
            request.entity_type,
            request.entity_id,
            request.change_type,
            request.old_value,
            request.new_value,
            request.confidence,
            request.impact_score,
            current_user['id']
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to record evolution"
            )
        
        return {"status": "recorded"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording evolution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns", response_model=Dict[str, str])
async def record_pattern(
    request: PatternTrendRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Record a pattern trend."""
    try:
        success = await service.record_pattern_trend(
            request.pattern_type,
            request.pattern_name,
            request.occurrence_count,
            request.effectiveness_score,
            request.context,
            request.project_id
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to record pattern"
            )
        
        return {"status": "recorded"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance", response_model=Dict[str, str])
async def record_performance(
    request: PerformanceRecordRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Record performance metrics."""
    try:
        success = await service.record_performance(
            request.endpoint,
            request.response_time,
            request.memory_usage,
            request.cpu_usage,
            request.error_count,
            request.request_count,
            request.user_count
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to record performance"
            )
        
        return {"status": "recorded"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{metric_type}/trends")
async def get_metric_trends(
    metric_type: str,
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    window: str = Query("hour", description="Aggregation window"),
    tags: Optional[str] = Query(None, description="Tag filters (JSON)"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get metric trends over time."""
    try:
        # Validate metric type
        try:
            mt = MetricType(metric_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric type: {metric_type}"
            )
        
        # Validate window
        try:
            agg_window = AggregationWindow(window)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid aggregation window: {window}"
            )
        
        # Default time range
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # Parse tags filter
        tags_filter = None
        if tags:
            import json
            try:
                tags_filter = json.loads(tags)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid tags JSON format"
                )
        
        trends = await service.get_metric_trends(
            mt, start_time, end_time, agg_window, tags_filter
        )
        
        return {
            "metric_type": metric_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "window": window,
            "trends": trends
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{metric_type}/analysis", response_model=TrendAnalysisResponse)
async def analyze_metric_trends(
    metric_type: str,
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    window: str = Query("day", description="Aggregation window"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Analyze trends in metric data."""
    try:
        # Validate metric type
        try:
            mt = MetricType(metric_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric type: {metric_type}"
            )
        
        # Validate window
        try:
            agg_window = AggregationWindow(window)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid aggregation window: {window}"
            )
        
        # Default time range
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=30)
        
        analysis = await service.analyze_trends(
            mt, start_time, end_time, agg_window
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


@router.get("/evolution/timeline")
async def get_evolution_timeline(
    entity_type: Optional[str] = Query(None, description="Entity type filter"),
    entity_id: Optional[str] = Query(None, description="Entity ID filter"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get knowledge evolution timeline."""
    try:
        evolution = await service.get_knowledge_evolution_timeline(
            entity_type, entity_id, start_time, end_time, limit
        )
        
        return {
            "evolution": evolution,
            "count": len(evolution)
        }
        
    except Exception as e:
        logger.error(f"Error getting evolution timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/effectiveness")
async def get_pattern_effectiveness(
    pattern_type: Optional[str] = Query(None, description="Pattern type filter"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    window: str = Query("day", description="Aggregation window"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get pattern effectiveness trends."""
    try:
        # Validate window
        try:
            agg_window = AggregationWindow(window)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid aggregation window: {window}"
            )
        
        trends = await service.get_pattern_effectiveness_trends(
            pattern_type, start_time, end_time, agg_window
        )
        
        return {
            "pattern_trends": trends,
            "count": len(trends)
        }
        
    except Exception as e:
        logger.error(f"Error getting pattern effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard_data(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get comprehensive analytics dashboard data."""
    try:
        dashboard_data = await service.generate_analytics_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error generating dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retention-policies")
async def get_retention_policies(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get current retention policy status."""
    try:
        policies = await service.get_retention_policy_status()
        return policies
        
    except Exception as e:
        logger.error(f"Error getting retention policies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hypertables")
async def get_hypertable_info(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get hypertable information and statistics."""
    try:
        info = await service.get_hypertable_info()
        return info
        
    except Exception as e:
        logger.error(f"Error getting hypertable info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for the analytics service."""
    try:
        service = await get_analytics_service()
        
        # Test basic functionality
        test_time = datetime.utcnow()
        hypertables = await service.get_hypertable_info()
        
        return {
            "status": "healthy",
            "service": "time_series_analytics",
            "timestamp": test_time.isoformat(),
            "hypertables": hypertables['total_hypertables']
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )