"""
Tracing Management API Routes

Provides endpoints for managing distributed tracing, performance analysis,
and trace data queries for debugging and optimization.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.opentelemetry_tracing import otel_tracing
from ..services.prometheus_metrics import prometheus_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tracing", tags=["tracing_management"])

# Request/Response Models

class TraceQuery(BaseModel):
    """Trace query parameters"""
    service_name: Optional[str] = None
    operation_name: Optional[str] = None
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    status: Optional[str] = Field(None, pattern="^(success|error)$")
    time_range_hours: int = Field(default=1, ge=1, le=24)

class PerformanceAnalysis(BaseModel):
    """Performance analysis request"""
    operation_pattern: Optional[str] = None
    include_percentiles: bool = True
    include_trends: bool = True
    time_window_hours: int = Field(default=1, ge=1, le=168)

class TracingConfig(BaseModel):
    """Tracing configuration"""
    sampling_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    slow_request_threshold_ms: float = Field(default=1000, ge=0)
    slow_query_threshold_ms: float = Field(default=100, ge=0)
    critical_path_threshold_ms: float = Field(default=50, ge=0)
    enabled: bool = True

# Tracing Status and Configuration

@router.get("/status")
async def get_tracing_status() -> Dict[str, Any]:
    """Get current tracing system status"""
    
    try:
        # Get basic tracing status
        tracing_enabled = otel_tracing.enabled
        current_trace_id = otel_tracing.get_trace_id()
        current_span_id = otel_tracing.get_span_id()
        
        # Get performance summary
        performance_summary = otel_tracing.get_performance_summary()
        
        return {
            "tracing_enabled": tracing_enabled,
            "service_name": otel_tracing.service_name if tracing_enabled else None,
            "service_version": otel_tracing.service_version if tracing_enabled else None,
            "current_trace_id": current_trace_id,
            "current_span_id": current_span_id,
            "active_spans": len(otel_tracing.active_spans),
            "jaeger_endpoint": otel_tracing.jaeger_endpoint if tracing_enabled else None,
            "otlp_endpoint": otel_tracing.otlp_endpoint if tracing_enabled else None,
            "performance_summary": performance_summary,
            "status": "healthy" if tracing_enabled else "disabled",
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get tracing status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tracing status: {e}")

@router.get("/performance/summary")
async def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary from traces"""
    
    try:
        performance_data = otel_tracing.get_performance_summary()
        
        # Enhance with additional analysis
        enhanced_summary = {}
        for operation, metrics in performance_data.items():
            enhanced_summary[operation] = {
                **metrics,
                "performance_grade": _calculate_performance_grade(operation, metrics),
                "recommendations": _get_performance_recommendations(operation, metrics)
            }
        
        return {
            "operations": enhanced_summary,
            "total_operations_tracked": len(performance_data),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_targets": {
                "memory_search_ms": 50,
                "api_request_ms": 200,
                "database_query_ms": 100,
                "ai_processing_ms": 2000
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {e}")

@router.get("/performance/bottlenecks")
async def identify_performance_bottlenecks(
    threshold_ms: float = Query(default=100, description="Threshold for slow operations in ms")
) -> Dict[str, Any]:
    """Identify performance bottlenecks from tracing data"""
    
    try:
        performance_data = otel_tracing.get_performance_summary()
        
        bottlenecks = []
        for operation, metrics in performance_data.items():
            if metrics.get("p95_duration_ms", 0) > threshold_ms:
                bottleneck = {
                    "operation": operation,
                    "p95_duration_ms": metrics["p95_duration_ms"],
                    "avg_duration_ms": metrics["avg_duration_ms"],
                    "max_duration_ms": metrics["max_duration_ms"],
                    "slow_operations_count": metrics["slow_operations"],
                    "total_operations": metrics["count"],
                    "severity": _calculate_bottleneck_severity(metrics["p95_duration_ms"], threshold_ms),
                    "impact_score": _calculate_impact_score(metrics)
                }
                bottlenecks.append(bottleneck)
        
        # Sort by impact score
        bottlenecks.sort(key=lambda x: x["impact_score"], reverse=True)
        
        return {
            "bottlenecks": bottlenecks,
            "total_bottlenecks": len(bottlenecks),
            "threshold_ms": threshold_ms,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": _get_bottleneck_recommendations(bottlenecks[:5])  # Top 5
        }
        
    except Exception as e:
        logger.error(f"Failed to identify bottlenecks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to identify bottlenecks: {e}")

@router.get("/operations/{operation_name}/analysis")
async def analyze_operation_performance(
    operation_name: str,
    time_window_hours: int = Query(default=1, ge=1, le=24)
) -> Dict[str, Any]:
    """Get detailed performance analysis for a specific operation"""
    
    try:
        performance_data = otel_tracing.get_performance_summary()
        
        if operation_name not in performance_data:
            raise HTTPException(status_code=404, detail=f"Operation '{operation_name}' not found")
        
        metrics = performance_data[operation_name]
        
        # Detailed analysis
        analysis = {
            "operation_name": operation_name,
            "time_window_hours": time_window_hours,
            "performance_metrics": metrics,
            "performance_grade": _calculate_performance_grade(operation_name, metrics),
            "trend_analysis": _analyze_performance_trend(operation_name, metrics),
            "recommendations": _get_detailed_recommendations(operation_name, metrics),
            "comparative_analysis": _compare_with_targets(operation_name, metrics),
            "optimization_opportunities": _identify_optimization_opportunities(operation_name, metrics)
        }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze operation {operation_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze operation: {e}")

@router.get("/traces/search")
async def search_traces(
    service_name: Optional[str] = Query(None, description="Filter by service name"),
    operation_name: Optional[str] = Query(None, description="Filter by operation name"),
    min_duration_ms: Optional[float] = Query(None, description="Minimum duration in ms"),
    max_duration_ms: Optional[float] = Query(None, description="Maximum duration in ms"),
    status: Optional[str] = Query(None, description="Filter by status (success/error)"),
    limit: int = Query(default=100, le=1000, description="Maximum number of traces")
) -> Dict[str, Any]:
    """Search for traces with filtering criteria"""
    
    try:
        # This would typically query the tracing backend (Jaeger/Tempo)
        # For now, return mock data based on current performance metrics
        
        performance_data = otel_tracing.get_performance_summary()
        mock_traces = []
        
        for operation, metrics in performance_data.items():
            if operation_name and operation_name not in operation:
                continue
                
            # Generate mock trace data
            trace = {
                "trace_id": f"trace_{operation}_{int(datetime.now().timestamp())}",
                "operation_name": operation,
                "service_name": service_name or "knowledgehub-api",
                "duration_ms": metrics.get("avg_duration_ms", 0),
                "status": "success" if metrics.get("slow_operations", 0) == 0 else "mixed",
                "span_count": 5,  # Mock span count
                "start_time": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
                "tags": {
                    "performance_grade": _calculate_performance_grade(operation, metrics),
                    "has_errors": metrics.get("slow_operations", 0) > 0
                }
            }
            
            # Apply filters
            if min_duration_ms and trace["duration_ms"] < min_duration_ms:
                continue
            if max_duration_ms and trace["duration_ms"] > max_duration_ms:
                continue
            if status and trace["status"] != status:
                continue
                
            mock_traces.append(trace)
            
            if len(mock_traces) >= limit:
                break
        
        return {
            "traces": mock_traces[:limit],
            "total_found": len(mock_traces),
            "query_duration_ms": 15,  # Mock query time
            "filters_applied": {
                "service_name": service_name,
                "operation_name": operation_name,
                "min_duration_ms": min_duration_ms,
                "max_duration_ms": max_duration_ms,
                "status": status
            },
            "search_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to search traces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search traces: {e}")

@router.get("/health")
async def get_tracing_health() -> Dict[str, Any]:
    """Get tracing system health status"""
    
    try:
        return {
            "status": "healthy" if otel_tracing.enabled else "disabled",
            "tracing_enabled": otel_tracing.enabled,
            "active_spans": len(otel_tracing.active_spans),
            "performance_tracking": len(otel_tracing.performance_metrics),
            "exporters": {
                "jaeger": "configured" if otel_tracing.jaeger_endpoint else "not_configured",
                "otlp": "configured" if otel_tracing.otlp_endpoint else "not_configured"
            },
            "auto_instrumentation": "enabled" if otel_tracing.enabled else "disabled",
            "last_check": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get tracing health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }

# Helper Functions

def _calculate_performance_grade(operation: str, metrics: Dict[str, Any]) -> str:
    """Calculate performance grade for an operation"""
    p95_duration = metrics.get("p95_duration_ms", 0)
    
    # Define thresholds based on operation type
    if "memory" in operation.lower():
        thresholds = {"A": 50, "B": 100, "C": 200, "D": 500}
    elif "db" in operation.lower():
        thresholds = {"A": 100, "B": 250, "C": 500, "D": 1000}
    elif "ai" in operation.lower():
        thresholds = {"A": 2000, "B": 5000, "C": 10000, "D": 20000}
    else:
        thresholds = {"A": 200, "B": 500, "C": 1000, "D": 2000}
    
    for grade, threshold in thresholds.items():
        if p95_duration <= threshold:
            return grade
    return "F"

def _get_performance_recommendations(operation: str, metrics: Dict[str, Any]) -> List[str]:
    """Get performance recommendations for an operation"""
    recommendations = []
    
    p95_duration = metrics.get("p95_duration_ms", 0)
    slow_operations = metrics.get("slow_operations", 0)
    
    if "memory" in operation.lower() and p95_duration > 50:
        recommendations.append("Consider optimizing memory search indices")
        recommendations.append("Review semantic search algorithm efficiency")
    
    if "db" in operation.lower() and p95_duration > 100:
        recommendations.append("Optimize database queries and add appropriate indices")
        recommendations.append("Consider database connection pooling optimization")
    
    if "ai" in operation.lower() and p95_duration > 5000:
        recommendations.append("Consider model optimization or caching")
        recommendations.append("Review AI model input preprocessing")
    
    if slow_operations > metrics.get("count", 0) * 0.1:
        recommendations.append("High percentage of slow operations detected")
        recommendations.append("Consider implementing caching strategies")
    
    return recommendations

def _calculate_bottleneck_severity(duration_ms: float, threshold_ms: float) -> str:
    """Calculate bottleneck severity"""
    ratio = duration_ms / threshold_ms
    
    if ratio >= 5.0:
        return "critical"
    elif ratio >= 3.0:
        return "high"
    elif ratio >= 2.0:
        return "medium"
    else:
        return "low"

def _calculate_impact_score(metrics: Dict[str, Any]) -> float:
    """Calculate impact score for bottleneck prioritization"""
    p95_duration = metrics.get("p95_duration_ms", 0)
    operation_count = metrics.get("count", 0)
    slow_operations = metrics.get("slow_operations", 0)
    
    # Impact = (duration impact) * (frequency impact) * (reliability impact)
    duration_impact = min(p95_duration / 1000, 10)  # Cap at 10
    frequency_impact = min(operation_count / 100, 5)  # Cap at 5
    reliability_impact = (slow_operations / max(operation_count, 1)) * 3  # Cap at 3
    
    return duration_impact * frequency_impact * reliability_impact

def _get_bottleneck_recommendations(bottlenecks: List[Dict[str, Any]]) -> List[str]:
    """Get recommendations for addressing bottlenecks"""
    if not bottlenecks:
        return ["No significant bottlenecks detected"]
    
    recommendations = [
        f"Priority 1: Optimize '{bottlenecks[0]['operation']}' operation",
        "Consider implementing performance monitoring alerts",
        "Review and optimize critical path operations",
        "Implement caching for frequently accessed data"
    ]
    
    if len(bottlenecks) > 3:
        recommendations.append("Multiple bottlenecks detected - consider systematic performance review")
    
    return recommendations

def _analyze_performance_trend(operation: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance trends for an operation"""
    # Mock trend analysis - in production would analyze historical data
    return {
        "trend": "stable",
        "trend_confidence": 0.85,
        "recent_change_percent": 2.3,
        "prediction": "Performance expected to remain stable"
    }

def _get_detailed_recommendations(operation: str, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """Get detailed recommendations with priorities"""
    recommendations = []
    
    p95_duration = metrics.get("p95_duration_ms", 0)
    
    if "memory" in operation.lower():
        if p95_duration > 50:
            recommendations.append({
                "priority": "high",
                "category": "performance",
                "recommendation": "Memory search exceeds 50ms target - optimize search indices",
                "expected_impact": "30-50% improvement"
            })
    
    return recommendations

def _compare_with_targets(operation: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Compare operation performance with targets"""
    targets = {
        "memory": 50,
        "db": 100,
        "ai": 2000,
        "api": 200
    }
    
    # Determine operation type
    operation_type = "api"  # default
    for op_type in targets.keys():
        if op_type in operation.lower():
            operation_type = op_type
            break
    
    target_ms = targets[operation_type]
    p95_duration = metrics.get("p95_duration_ms", 0)
    
    return {
        "target_ms": target_ms,
        "current_p95_ms": p95_duration,
        "meets_target": p95_duration <= target_ms,
        "variance_percent": ((p95_duration - target_ms) / target_ms) * 100,
        "grade": _calculate_performance_grade(operation, metrics)
    }

def _identify_optimization_opportunities(operation: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify specific optimization opportunities"""
    opportunities = []
    
    slow_operations = metrics.get("slow_operations", 0)
    total_operations = metrics.get("count", 0)
    
    if slow_operations > 0:
        opportunities.append({
            "type": "performance",
            "description": f"{slow_operations} slow operations out of {total_operations}",
            "potential_improvement": "20-40% latency reduction",
            "effort": "medium"
        })
    
    return opportunities