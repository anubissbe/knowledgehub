"""
Phase 3: Advanced Analytics & Insights API Router
François Coppens - Performance Profiling Expert

Enhanced analytics endpoints with real-time performance monitoring,
GPU acceleration support, and sub-10ms query response times.
"""

import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, Request, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uuid
import asyncio

# Enhanced imports for Phase 3
from api.services.timescale_analytics import (
    TimeSeriesAnalyticsService,
    MetricType,
    AggregationWindow,
    PerformanceMetrics,
    TimeSeriesPoint,
    TrendAnalysis,
    QUERY_LATENCY_TARGET_MS
)
from api.dependencies import get_current_user
from shared.logging import setup_logging

logger = setup_logging("api.analytics_advanced")

router = APIRouter(prefix="/api/analytics", tags=["advanced-analytics"])

# Phase 3 Enhanced Request Models

class AdvancedMetricRequest(BaseModel):
    """Enhanced metric recording with performance tracking"""
    metric_type: str = Field(..., description="Type of metric to record")
    value: float = Field(..., description="Metric value")
    tags: Optional[Dict[str, str]] = Field(None, description="Optional tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    performance_data: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    project_id: Optional[str] = Field(None, description="Project ID")
    timestamp: Optional[datetime] = Field(None, description="Timestamp (defaults to now)")
    
    @validator('metric_type')
    def validate_metric_type(cls, v):
        try:
            MetricType(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid metric type: {v}")


class RAGPerformanceRequest(BaseModel):
    """RAG pipeline performance recording"""
    query_id: str = Field(..., description="Query identifier")
    query_type: str = Field(..., description="Type of query")
    retrieval_time_ms: float = Field(..., ge=0.0, description="Retrieval time in milliseconds")
    generation_time_ms: float = Field(..., ge=0.0, description="Generation time in milliseconds")
    chunk_count: int = Field(..., ge=0, description="Number of chunks processed")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    tokens_processed: Optional[int] = Field(None, ge=0, description="Tokens processed")
    model_used: Optional[str] = Field(None, description="Model identifier")
    success: bool = Field(True, description="Whether operation was successful")


class PerformanceAnalyticsRequest(BaseModel):
    """System performance analytics recording"""
    component: str = Field(..., description="System component")
    operation: str = Field(..., description="Operation performed")
    latency_ms: float = Field(..., ge=0.0, description="Latency in milliseconds")
    throughput_ops_sec: Optional[float] = Field(None, ge=0.0, description="Throughput in ops/sec")
    memory_usage_mb: Optional[float] = Field(None, ge=0.0, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="CPU usage percentage")
    gpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU usage percentage")
    error_rate: float = Field(0.0, ge=0.0, le=100.0, description="Error rate percentage")
    success_rate: float = Field(100.0, ge=0.0, le=100.0, description="Success rate percentage")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RealTimeQueryRequest(BaseModel):
    """Real-time metrics query"""
    metric_types: List[str] = Field(..., description="List of metric types to query")
    time_range_minutes: int = Field(60, ge=1, le=1440, description="Time range in minutes")
    aggregation_window: Optional[str] = Field("minute", description="Aggregation window")
    tags_filter: Optional[Dict[str, str]] = Field(None, description="Tag filters")
    
    @validator('metric_types')
    def validate_metric_types(cls, v):
        for mt in v:
            try:
                MetricType(mt)
            except ValueError:
                raise ValueError(f"Invalid metric type: {mt}")
        return v


class PerformanceDashboardResponse(BaseModel):
    """Performance dashboard data"""
    system_performance: Dict[str, Any]
    rag_performance: Dict[str, Any]
    gpu_utilization: Dict[str, Any]
    real_time_status: Dict[str, Any]
    performance_alerts: List[Dict[str, Any]]
    optimization_recommendations: List[str]
    generation_time_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response"""
    service: str
    status: str
    initialized: bool
    gpu_available: bool
    performance_targets_met: bool
    database_connectivity: bool
    continuous_aggregates_active: bool
    health_check_latency_ms: float
    issues: List[str]


# Global service instance
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


# Phase 3 Enhanced Endpoints

@router.post("/metrics/advanced", response_model=Dict[str, Any])
async def record_advanced_metric(
    request: AdvancedMetricRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """
    Record advanced metric with performance tracking
    François Coppens: Target sub-10ms recording latency
    """
    start_time = time.time()
    
    try:
        # Convert performance data to PerformanceMetrics if provided
        perf_metrics = None
        if request.performance_data:
            perf_metrics = PerformanceMetrics(
                query_latency_ms=request.performance_data.get('query_latency_ms', 0.0),
                memory_usage_mb=request.performance_data.get('memory_usage_mb', 0.0),
                gpu_utilization=request.performance_data.get('gpu_utilization', 0.0),
                cpu_utilization=request.performance_data.get('cpu_utilization', 0.0),
                throughput_ops_sec=request.performance_data.get('throughput_ops_sec', 0.0),
                cache_hit_rate=request.performance_data.get('cache_hit_rate', 0.0),
                compression_ratio=request.performance_data.get('compression_ratio', 1.0),
                error_count=int(request.performance_data.get('error_count', 0))
            )
        
        # Record metric
        success = await service.record_metric(
            MetricType(request.metric_type),
            request.value,
            request.tags,
            request.metadata,
            current_user['id'],
            None,  # session_id can be added later
            request.project_id,
            request.timestamp
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record metric")
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Background task to record this API call's performance
        background_tasks.add_task(
            service.record_performance_metric,
            component="analytics_api",
            operation="record_advanced_metric",
            latency_ms=processing_time_ms,
            metadata={"metric_type": request.metric_type}
        )
        
        response = {
            "status": "recorded",
            "processing_time_ms": processing_time_ms,
            "performance_target_met": processing_time_ms <= QUERY_LATENCY_TARGET_MS
        }
        
        if processing_time_ms <= QUERY_LATENCY_TARGET_MS:
            logger.debug(f"Advanced metric recorded in {processing_time_ms:.2f}ms - TARGET MET")
        else:
            logger.warning(f"Advanced metric recording took {processing_time_ms:.2f}ms - ABOVE TARGET")
            response["warning"] = f"Processing time {processing_time_ms:.2f}ms exceeded target {QUERY_LATENCY_TARGET_MS}ms"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording advanced metric: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/performance", response_model=Dict[str, str])
async def record_rag_performance(
    request: RAGPerformanceRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Record RAG pipeline performance metrics"""
    try:
        success = await service.record_rag_performance(
            request.query_id,
            request.query_type,
            request.retrieval_time_ms,
            request.generation_time_ms,
            request.chunk_count,
            request.relevance_score,
            request.tokens_processed,
            request.model_used,
            request.success
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record RAG performance")
        
        # Background task to track API performance
        background_tasks.add_task(
            service.record_performance_metric,
            component="rag_pipeline",
            operation=request.query_type,
            latency_ms=request.retrieval_time_ms + request.generation_time_ms,
            metadata={"model_used": request.model_used}
        )
        
        return {"status": "recorded"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording RAG performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance", response_model=Dict[str, str])
async def record_performance_analytics(
    request: PerformanceAnalyticsRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Record system performance analytics"""
    try:
        success = await service.record_performance_metric(
            request.component,
            request.operation,
            request.latency_ms,
            request.throughput_ops_sec,
            request.memory_usage_mb,
            request.cpu_usage_percent,
            request.gpu_usage_percent,
            request.error_rate,
            request.success_rate,
            request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record performance analytics")
        
        return {"status": "recorded"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording performance analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/realtime/query")
async def query_realtime_metrics(
    request: RealTimeQueryRequest,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """
    Query real-time metrics with sub-10ms response time
    Uses continuous aggregates for optimal performance
    """
    start_time = time.time()
    
    try:
        # Convert string metric types to enums
        metric_types = [MetricType(mt) for mt in request.metric_types]
        
        # Execute real-time query
        metrics = await service.get_real_time_metrics(
            metric_types,
            request.time_range_minutes
        )
        
        query_time_ms = (time.time() - start_time) * 1000
        
        response = {
            "metrics": metrics,
            "query_time_ms": query_time_ms,
            "performance_target_met": query_time_ms <= QUERY_LATENCY_TARGET_MS,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if query_time_ms > QUERY_LATENCY_TARGET_MS:
            response["warning"] = f"Query time {query_time_ms:.2f}ms exceeded target {QUERY_LATENCY_TARGET_MS}ms"
        
        return response
        
    except Exception as e:
        logger.error(f"Error querying real-time metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/performance", response_model=PerformanceDashboardResponse)
async def get_performance_dashboard(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """
    Get comprehensive performance dashboard data
    François Coppens: Optimized for performance profiling insights
    """
    try:
        dashboard_data = await service.generate_analytics_dashboard_data()
        
        return PerformanceDashboardResponse(**dashboard_data)
        
    except Exception as e:
        logger.error(f"Error generating performance dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gpu/status")
async def get_gpu_status(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get GPU acceleration status and utilization"""
    try:
        gpu_info = service.gpu_analytics.get_device_info()
        
        # Get recent GPU metrics
        gpu_metrics = await service._get_gpu_utilization_metrics()
        
        return {
            "gpu_info": gpu_info,
            "utilization_metrics": gpu_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting GPU status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bottlenecks/analysis")
async def analyze_system_bottlenecks(
    time_range_hours: int = Query(1, ge=1, le=24, description="Analysis time range in hours"),
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """
    Analyze system bottlenecks using François Coppens methodology
    Identifies CPU, memory, GPU, and database bottlenecks
    """
    try:
        start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Analyze bottlenecks across different components
        bottleneck_analysis = {
            'analysis_period': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'duration_hours': time_range_hours
            },
            'cpu_bottlenecks': [],
            'memory_bottlenecks': [],
            'gpu_bottlenecks': [],
            'database_bottlenecks': [],
            'network_bottlenecks': [],
            'recommendations': []
        }
        
        # Get performance data for analysis
        query = """
        SELECT 
            component,
            operation,
            AVG(latency_ms) as avg_latency,
            MAX(latency_ms) as max_latency,
            AVG(cpu_usage_percent) as avg_cpu,
            AVG(memory_usage_mb) as avg_memory,
            AVG(gpu_usage_percent) as avg_gpu,
            COUNT(*) as operation_count
        FROM ts_performance_analytics
        WHERE time >= $1
        GROUP BY component, operation
        HAVING AVG(latency_ms) > $2 OR AVG(cpu_usage_percent) > 80 OR AVG(memory_usage_mb) > 1000
        ORDER BY avg_latency DESC
        """
        
        async with service.async_engine.connect() as conn:
            result = await conn.execute(text(query), (start_time, QUERY_LATENCY_TARGET_MS))
            
            for row in result:
                bottleneck_entry = {
                    'component': row.component,
                    'operation': row.operation,
                    'avg_latency_ms': float(row.avg_latency),
                    'max_latency_ms': float(row.max_latency),
                    'avg_cpu_percent': float(row.avg_cpu) if row.avg_cpu else 0,
                    'avg_memory_mb': float(row.avg_memory) if row.avg_memory else 0,
                    'avg_gpu_percent': float(row.avg_gpu) if row.avg_gpu else 0,
                    'operation_count': int(row.operation_count),
                    'severity': 'high' if row.avg_latency > QUERY_LATENCY_TARGET_MS * 5 else 'medium'
                }
                
                # Categorize bottlenecks
                if row.avg_cpu and row.avg_cpu > 80:
                    bottleneck_analysis['cpu_bottlenecks'].append(bottleneck_entry)
                
                if row.avg_memory and row.avg_memory > 1000:
                    bottleneck_analysis['memory_bottlenecks'].append(bottleneck_entry)
                
                if row.avg_gpu and row.avg_gpu > 90:
                    bottleneck_analysis['gpu_bottlenecks'].append(bottleneck_entry)
                elif row.avg_gpu and row.avg_gpu < 10:
                    bottleneck_analysis['gpu_bottlenecks'].append({
                        **bottleneck_entry,
                        'issue_type': 'underutilized'
                    })
                
                if row.avg_latency > QUERY_LATENCY_TARGET_MS * 2:
                    bottleneck_analysis['database_bottlenecks'].append(bottleneck_entry)
        
        # Generate François Coppens recommendations
        recommendations = []
        
        if bottleneck_analysis['cpu_bottlenecks']:
            recommendations.append("High CPU usage detected. Consider optimizing compute-intensive operations or scaling horizontally.")
        
        if bottleneck_analysis['memory_bottlenecks']:
            recommendations.append("Memory pressure detected. Review memory allocation and implement caching strategies.")
        
        if any(b.get('issue_type') == 'underutilized' for b in bottleneck_analysis['gpu_bottlenecks']):
            recommendations.append("GPU underutilization detected. Enable GPU acceleration for analytics workloads.")
        
        if bottleneck_analysis['database_bottlenecks']:
            recommendations.append("Database latency issues detected. Optimize queries, add indexes, or review connection pooling.")
        
        if not any(bottleneck_analysis.values()):
            recommendations.append("No significant bottlenecks detected. System performing within François Coppens standards.")
        
        bottleneck_analysis['recommendations'] = recommendations
        
        return bottleneck_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing bottlenecks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/recommendations")
async def get_optimization_recommendations(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get François Coppens optimization recommendations"""
    try:
        recommendations = await service._get_optimization_recommendations()
        
        return {
            "recommendations": recommendations,
            "performance_standards": {
                "query_latency_target_ms": QUERY_LATENCY_TARGET_MS,
                "memory_efficiency_target": 0.8,
                "gpu_utilization_target": 0.9
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Backward compatibility endpoints

@router.post("/metrics", response_model=Dict[str, str])
async def record_metric(
    request: dict,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Backward compatibility endpoint for metric recording"""
    try:
        # Convert to enhanced request format
        enhanced_request = AdvancedMetricRequest(**request)
        response = await record_advanced_metric(enhanced_request, BackgroundTasks(), current_user, service)
        return {"status": response["status"]}
        
    except Exception as e:
        logger.error(f"Error in backward compatibility metric endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard_data(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Backward compatibility dashboard endpoint"""
    try:
        return await service.generate_analytics_dashboard_data()
        
    except Exception as e:
        logger.error(f"Error generating dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retention-policies")
async def get_retention_policies(
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Get current retention policy status"""
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
    """Get hypertable information and statistics"""
    try:
        info = await service.get_hypertable_info()
        return info
        
    except Exception as e:
        logger.error(f"Error getting hypertable info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check for advanced analytics service
    François Coppens: Validates sub-10ms performance targets
    """
    start_time = time.time()
    
    try:
        service = await get_analytics_service()
        
        # Basic functionality test
        test_time = datetime.utcnow()
        
        # Test database connectivity
        database_connectivity = False
        try:
            async with service.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                database_connectivity = True
        except Exception as e:
            logger.error(f"Database connectivity test failed: {e}")
        
        # Test continuous aggregates
        continuous_aggregates_active = False
        try:
            async with service.async_engine.connect() as conn:
                result = await conn.execute(text("""
                    SELECT COUNT(*) as ca_count 
                    FROM timescaledb_information.continuous_aggregates
                    WHERE view_name IN ('ts_metrics_1min', 'ts_performance_5min', 'ts_rag_hourly')
                """))
                ca_count = result.scalar()
                continuous_aggregates_active = ca_count >= 3
        except Exception as e:
            logger.warning(f"Continuous aggregates check failed: {e}")
        
        # Performance test
        health_check_latency_ms = (time.time() - start_time) * 1000
        performance_targets_met = health_check_latency_ms <= QUERY_LATENCY_TARGET_MS
        
        issues = []
        if not database_connectivity:
            issues.append("Database connectivity failed")
        
        if not continuous_aggregates_active:
            issues.append("Continuous aggregates not fully active")
        
        if not performance_targets_met:
            issues.append(f"Health check latency {health_check_latency_ms:.2f}ms exceeded target {QUERY_LATENCY_TARGET_MS}ms")
        
        # Determine overall status
        if not database_connectivity:
            status = "unhealthy"
        elif issues:
            status = "degraded"
        else:
            status = "healthy"
        
        return HealthCheckResponse(
            service="advanced_timescale_analytics",
            status=status,
            initialized=service._initialized,
            gpu_available=service.gpu_analytics.device_count > 0,
            performance_targets_met=performance_targets_met,
            database_connectivity=database_connectivity,
            continuous_aggregates_active=continuous_aggregates_active,
            health_check_latency_ms=health_check_latency_ms,
            issues=issues
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            service="advanced_timescale_analytics",
            status="unhealthy",
            initialized=False,
            gpu_available=False,
            performance_targets_met=False,
            database_connectivity=False,
            continuous_aggregates_active=False,
            health_check_latency_ms=(time.time() - start_time) * 1000,
            issues=[f"Health check failed: {str(e)}"]
        )


# Integration endpoints for RAG pipeline and FPGA workflow

@router.post("/integrations/rag/track")
async def track_rag_integration(
    request: dict,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Track RAG pipeline integration metrics"""
    try:
        # Record RAG performance from pipeline
        await service.record_rag_performance(
            query_id=request.get('query_id', str(uuid.uuid4())),
            query_type=request.get('query_type', 'general'),
            retrieval_time_ms=request.get('retrieval_time_ms', 0.0),
            generation_time_ms=request.get('generation_time_ms', 0.0),
            chunk_count=request.get('chunk_count', 0),
            relevance_score=request.get('relevance_score'),
            tokens_processed=request.get('tokens_processed'),
            model_used=request.get('model_used'),
            success=request.get('success', True)
        )
        
        # Record as general metric for dashboard
        await service.record_metric(
            MetricType.RAG_PERFORMANCE,
            request.get('retrieval_time_ms', 0.0) + request.get('generation_time_ms', 0.0),
            tags={'query_type': request.get('query_type', 'general')},
            metadata=request
        )
        
        return {"status": "tracked", "integration": "rag_pipeline"}
        
    except Exception as e:
        logger.error(f"Error tracking RAG integration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/integrations/fpga/track")
async def track_fpga_integration(
    request: dict,
    current_user: dict = Depends(get_current_user),
    service: TimeSeriesAnalyticsService = Depends(get_analytics_service)
):
    """Track FPGA workflow engine integration metrics"""
    try:
        # Record FPGA utilization metrics
        await service.record_metric(
            MetricType.FPGA_UTILIZATION,
            request.get('utilization_percent', 0.0),
            tags={
                'workflow_type': request.get('workflow_type', 'general'),
                'acceleration_enabled': str(request.get('acceleration_enabled', False))
            },
            metadata=request
        )
        
        # Record performance metrics
        if 'processing_time_ms' in request:
            await service.record_performance_metric(
                component="fpga_workflow",
                operation=request.get('workflow_type', 'general'),
                latency_ms=request['processing_time_ms'],
                throughput_ops_sec=request.get('throughput_ops_sec'),
                metadata=request
            )
        
        return {"status": "tracked", "integration": "fpga_workflow"}
        
    except Exception as e:
        logger.error(f"Error tracking FPGA integration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
