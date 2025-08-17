"""
Performance Metrics API - Track command execution, success rates, and optimize performance
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import time

from ..models import get_db
from ..services.performance_metrics_tracker import PerformanceMetricsTracker

router = APIRouter(prefix="/api/performance", tags=["performance"])

# Global performance tracker
performance_tracker = PerformanceMetricsTracker()


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "performance-metrics",
        "description": "Track command execution patterns, success rates, and optimize performance"
    }


@router.post("/track")
def track_performance(
    data: Dict[str, Any] = Body(..., description="Performance tracking data"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Track execution of a command with performance metrics
    
    Expected data format:
    {
        "operation": "test_operation",
        "duration_ms": 150,
        "metadata": {"test": true}
    }
    
    OR the full format:
    {
        "command_type": "file_read",
        "execution_time": 0.5,
        "success": true,
        "output_size": 1024,
        "error_message": null,
        "project_id": "project123",
        "session_id": "session123",
        "command_details": {"file_path": "/path/to/file"},
        "context": {"working_directory": "/project"}
    }
    """
    try:
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Received performance tracking data: {data}")
        logger.info(f"Data types: {[(k, type(v).__name__) for k, v in data.items()]}")
        
        # Support both formats
        if "operation" in data:
            # Simple format
            command_type = data.get("operation", "unknown")
            execution_time = data.get("duration_ms", 0) / 1000.0  # Convert ms to seconds
            success = True
            command_details = data.get("metadata", {})
            context = {}
            output_size = None
            error_message = None
            project_id = None
            session_id = None
        else:
            # Full format
            command_type = data.get("command_type") or data.get("command", "unknown")
            execution_time = float(data.get("execution_time", 0))  # Ensure it's a float
            success = bool(data.get("success", True))  # Ensure it's a bool
            command_details = data.get("command_details", {})
            context = data.get("context", {})
            output_size = data.get("output_size")
            error_message = data.get("error_message")
            project_id = data.get("project_id")
            session_id = data.get("session_id")
        
        # Add user_id if provided
        user_id = data.get("user_id", "default-user")
        
        logger.info(f"Calling track_command_execution with:")
        logger.info(f"  command_type: {command_type} (type: {type(command_type).__name__})")
        logger.info(f"  execution_time: {execution_time} (type: {type(execution_time).__name__})")
        logger.info(f"  success: {success} (type: {type(success).__name__})")
        
        result = performance_tracker.track_command_execution(
            db, command_type, command_details, execution_time,
            success, output_size, error_message, context,
            project_id, session_id
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in track_performance: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-detailed")
def track_performance_detailed(
    command_type: str = Query(..., description="Type of command executed"),
    execution_time: float = Query(..., description="Execution time in seconds"),
    success: bool = Query(..., description="Whether command succeeded"),
    output_size: Optional[int] = Query(None, description="Size of output in bytes"),
    error_message: Optional[str] = Query(None, description="Error message if failed"),
    project_id: Optional[str] = Query(None, description="Project identifier"),
    session_id: Optional[str] = Query(None, description="Session identifier"),
    execution_data: Dict[str, Any] = Body(..., description="Command details and context"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Track execution of a command with performance metrics
    
    execution_data format:
    {
        "command_details": {
            "file_path": "/path/to/file",
            "parameters": {...}
        },
        "context": {
            "working_directory": "/project",
            "environment": "development"
        }
    }
    """
    try:
        command_details = execution_data.get("command_details", {})
        context = execution_data.get("context", {})
        
        result = performance_tracker.track_command_execution(
            db, command_type, command_details, execution_time,
            success, output_size, error_message, context,
            project_id, session_id
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-batch")
def track_batch_performance(
    metrics_batch: List[Dict[str, Any]] = Body(..., description="Batch of metrics to track"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Track multiple command executions in batch
    
    metrics_batch format:
    [
        {
            "command_type": "file_read",
            "execution_time": 0.5,
            "success": true,
            "command_details": {...},
            "context": {...}
        },
        ...
    ]
    """
    try:
        results = []
        for metrics in metrics_batch:
            result = performance_tracker.track_command_execution(
                db,
                command_type=metrics["command_type"],
                command_details=metrics.get("command_details", {}),
                execution_time=metrics["execution_time"],
                success=metrics["success"],
                output_size=metrics.get("output_size"),
                error_message=metrics.get("error_message"),
                context=metrics.get("context"),
                project_id=metrics.get("project_id"),
                session_id=metrics.get("session_id")
            )
            results.append(result)
        
        return {
            "tracked_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
def get_performance_report(
    category: Optional[str] = Query(None, description="Filter by command category"),
    time_range: Optional[int] = Query(7, description="Time range in days"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive performance report with insights
    """
    try:
        report = performance_tracker.get_performance_report(
            db, category, time_range, project_id
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
def predict_performance(
    command_type: str = Query(..., description="Type of command to predict"),
    prediction_data: Dict[str, Any] = Body(..., description="Command details and context"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Predict performance for a command before execution
    
    prediction_data format:
    {
        "command_details": {
            "file_path": "/path/to/file",
            "parameters": {...}
        },
        "context": {
            "working_directory": "/project"
        }
    }
    """
    try:
        command_details = prediction_data.get("command_details", {})
        context = prediction_data.get("context", {})
        
        prediction = performance_tracker.predict_performance(
            db, command_type, command_details, context
        )
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
def analyze_patterns(
    time_range: int = Query(7, description="Time range in days"),
    min_frequency: int = Query(3, description="Minimum frequency to consider"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Analyze command execution patterns over time
    """
    try:
        patterns = performance_tracker.analyze_command_patterns(
            db, time_range, min_frequency
        )
        return patterns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization-history")
def get_optimization_history(
    strategy: Optional[str] = Query(None, description="Filter by optimization strategy"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get history of optimization attempts and their results
    """
    try:
        history = performance_tracker.get_optimization_history(
            db, strategy, project_id
        )
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
def get_command_categories() -> Dict[str, List[str]]:
    """
    Get all command categories and their keywords
    """
    return performance_tracker.command_categories


@router.get("/thresholds")
def get_performance_thresholds() -> Dict[str, float]:
    """
    Get performance threshold definitions
    """
    return performance_tracker.performance_thresholds


@router.get("/optimization-strategies")
def get_optimization_strategies() -> Dict[str, Any]:
    """
    Get available optimization strategies
    """
    return performance_tracker.optimization_strategies


@router.post("/benchmark")
async def benchmark_command(
    command_type: str = Query(..., description="Command type to benchmark"),
    iterations: int = Query(5, ge=1, le=20, description="Number of iterations"),
    benchmark_data: Dict[str, Any] = Body(..., description="Command details for benchmarking"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Benchmark a command by running it multiple times
    
    benchmark_data format:
    {
        "command_details": {...},
        "context": {...}
    }
    """
    try:
        command_details = benchmark_data.get("command_details", {})
        context = benchmark_data.get("context", {})
        
        execution_times = []
        successes = 0
        
        for i in range(iterations):
            start_time = time.time()
            
            # Simulate command execution (in real implementation, would execute actual command)
            # For now, we'll use a simple sleep to simulate work
            import random
            await asyncio.sleep(random.uniform(0.1, 0.5))
            success = random.random() > 0.1  # 90% success rate simulation
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            if success:
                successes += 1
            
            # Track each execution
            performance_tracker.track_command_execution(
                db, command_type, command_details, execution_time,
                success, None, None, context, None, None
            )
        
        # Calculate statistics
        import statistics
        
        benchmark_result = {
            "command_type": command_type,
            "iterations": iterations,
            "success_rate": successes / iterations,
            "execution_times": {
                "min": min(execution_times),
                "max": max(execution_times),
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "stdev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            "performance_rating": "fast" if statistics.mean(execution_times) < 1.0 else "normal"
        }
        
        return benchmark_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
def get_performance_recommendations(
    limit: int = Query(10, le=50, description="Maximum recommendations"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get personalized performance recommendations based on usage patterns
    """
    try:
        # Get recent performance report
        report = performance_tracker.get_performance_report(db, time_range=7)
        
        recommendations = []
        
        # Analyze slow commands
        for category, data in report["performance_breakdown"].items():
            if data["average_time"] > 5.0:
                recommendations.append({
                    "type": "slow_command",
                    "category": category,
                    "recommendation": f"Commands in {category} category are running slowly (avg {data['average_time']:.1f}s)",
                    "action": "Consider caching results or optimizing algorithms",
                    "priority": "high",
                    "potential_time_saved": data["average_time"] * 0.5
                })
        
        # Analyze failure rates
        for category, data in report["performance_breakdown"].items():
            if data["success_rate"] < 0.8:
                recommendations.append({
                    "type": "high_failure_rate",
                    "category": category,
                    "recommendation": f"{category} commands have {(1-data['success_rate'])*100:.0f}% failure rate",
                    "action": "Review error handling and prerequisites",
                    "priority": "high",
                    "potential_improvement": "Reduce failures by 50%"
                })
        
        # Pattern-based recommendations
        for pattern, count in report["common_patterns"].items():
            if pattern == "repeated_command" and count > 5:
                recommendations.append({
                    "type": "repeated_execution",
                    "pattern": pattern,
                    "recommendation": "Same commands are being executed repeatedly",
                    "action": "Implement result caching to avoid redundant work",
                    "priority": "medium",
                    "occurrences": count
                })
            elif pattern == "sequential_operations" and count > 3:
                recommendations.append({
                    "type": "sequential_execution",
                    "pattern": pattern,
                    "recommendation": "Independent operations running sequentially",
                    "action": "Execute operations in parallel where possible",
                    "priority": "medium",
                    "potential_speedup": "2-3x"
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
def get_performance_trends(
    metric: str = Query("execution_time", description="Metric to analyze"),
    time_range: int = Query(30, description="Time range in days"),
    interval: str = Query("daily", description="Aggregation interval: hourly, daily, weekly"),
    project_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get performance trends over time
    """
    try:
        # Get performance data
        report = performance_tracker.get_performance_report(
            db, time_range=time_range, project_id=project_id
        )
        
        trends = {
            "metric": metric,
            "time_range": time_range,
            "interval": interval,
            "data_points": [],
            "summary": {
                "direction": "stable",
                "change_percentage": 0,
                "current_value": 0,
                "historical_average": 0
            }
        }
        
        # Calculate trend direction from report
        if report.get("trends"):
            for trend in report["trends"]:
                if trend["metric"] == metric:
                    trends["summary"]["direction"] = trend["trend"]
                    trends["summary"]["change_percentage"] = trend["change_percentage"]
        
        # Add current metrics
        if metric == "execution_time":
            trends["summary"]["current_value"] = report["summary"]["average_execution_time"]
        elif metric == "success_rate":
            trends["summary"]["current_value"] = report["summary"].get("success_rate", 0)
        elif metric == "quality_score":
            trends["summary"]["current_value"] = report["summary"]["average_quality_score"]
        
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/hourly")
def get_hourly_metrics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get hourly performance metrics for dashboard
    """
    from datetime import datetime, timedelta
    import random
    
    # Generate mock hourly data for now
    hours = []
    now = datetime.now()
    for i in range(24):
        hour_time = now - timedelta(hours=23-i)
        hours.append({
            "timestamp": hour_time.isoformat(),
            "cpu": random.randint(20, 80),
            "memory": random.randint(30, 70),
            "requests": random.randint(100, 1000),
            "response_time": random.uniform(50, 500),
            "error_rate": random.uniform(0, 5),
            "throughput": random.randint(50, 200)
        })
    
    return {
        "metrics": hours,
        "summary": {
            "avg_cpu": sum(h["cpu"] for h in hours) / len(hours),
            "avg_memory": sum(h["memory"] for h in hours) / len(hours),
            "total_requests": sum(h["requests"] for h in hours),
            "avg_response_time": sum(h["response_time"] for h in hours) / len(hours),
            "avg_error_rate": sum(h["error_rate"] for h in hours) / len(hours),
            "peak_throughput": max(h["throughput"] for h in hours)
        }
    }


@router.get("/radar")
def get_performance_radar(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get radar chart data for performance visualization
    """
    return {
        "categories": [
            "CPU Usage",
            "Memory Usage",
            "Response Time",
            "Throughput",
            "Error Rate",
            "Availability"
        ],
        "datasets": [
            {
                "label": "Current",
                "data": [65, 45, 85, 75, 95, 99],
                "borderColor": "rgb(54, 162, 235)",
                "backgroundColor": "rgba(54, 162, 235, 0.2)"
            },
            {
                "label": "Baseline",
                "data": [70, 50, 80, 70, 90, 95],
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)"
            },
            {
                "label": "Target",
                "data": [80, 60, 90, 85, 98, 99.9],
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)"
            }
        ]
    }


# Import asyncio for async operations
import asyncio