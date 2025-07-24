"""
Performance Monitoring System for KnowledgeHub MCP Server.

This module provides comprehensive performance monitoring, metrics collection,
and health checks for all MCP tools and operations.

Features:
- Real-time tool execution monitoring
- Performance metrics collection
- Health status tracking
- Alert generation for performance issues
- Integration with KnowledgeHub analytics
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import statistics
import threading

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionMetric:
    """Metric for a single tool execution."""
    tool_name: str
    execution_time: float
    success: bool
    timestamp: datetime
    user_context: Optional[str] = None
    error_message: Optional[str] = None
    memory_usage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "execution_time": self.execution_time,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "user_context": self.user_context,
            "error_message": self.error_message,
            "memory_usage": self.memory_usage
        }


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    tool_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    error_rate: float = 0.0
    recent_errors: List[str] = field(default_factory=list)
    
    def update(self, metric: ToolExecutionMetric):
        """Update stats with new metric."""
        self.total_executions += 1
        self.last_execution = metric.timestamp
        
        if metric.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            if metric.error_message:
                self.recent_errors.append(metric.error_message)
                # Keep only last 5 errors
                self.recent_errors = self.recent_errors[-5:]
        
        # Update execution time stats
        self.min_execution_time = min(self.min_execution_time, metric.execution_time)
        self.max_execution_time = max(self.max_execution_time, metric.execution_time)
        
        # Calculate average execution time
        if self.total_executions > 0:
            # Simple moving average for performance
            weight = 0.1  # Weight for new value
            if self.avg_execution_time == 0:
                self.avg_execution_time = metric.execution_time
            else:
                self.avg_execution_time = (
                    (1 - weight) * self.avg_execution_time + 
                    weight * metric.execution_time
                )
        
        # Calculate error rate
        self.error_rate = self.failed_executions / self.total_executions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "avg_execution_time": round(self.avg_execution_time, 3),
            "min_execution_time": round(self.min_execution_time, 3) if self.min_execution_time != float('inf') else None,
            "max_execution_time": round(self.max_execution_time, 3),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "error_rate": round(self.error_rate, 3),
            "recent_errors": self.recent_errors
        }


@dataclass
class HealthStatus:
    """System health status."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "checks": self.checks,
            "alerts": self.alerts
        }


class MCPPerformanceMonitor:
    """Comprehensive performance monitoring for MCP operations."""
    
    def __init__(self, max_metrics_history: int = 1000):
        self.max_metrics_history = max_metrics_history
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.tool_stats: Dict[str, PerformanceStats] = {}
        self.system_stats = {
            "start_time": datetime.utcnow(),
            "total_tools_executed": 0,
            "active_connections": 0,
            "total_errors": 0
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "execution_time_warning": 1000.0,  # 1 second
            "execution_time_critical": 5000.0,  # 5 seconds
            "error_rate_warning": 0.05,  # 5%
            "error_rate_critical": 0.15,  # 15%
            "response_time_p95": 2000.0,  # 2 seconds for 95th percentile
        }
        
        # Health monitoring
        self.health_status = HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow()
        )
        
        # Real-time monitoring
        self.monitoring_active = True
        self.monitor_task = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("MCP Performance Monitor initialized")
    
    def record_tool_execution(
        self,
        tool_name: str,
        execution_time: float,
        success: bool,
        user_context: Optional[str] = None,
        error_message: Optional[str] = None,
        memory_usage: Optional[float] = None
    ):
        """Record a tool execution metric."""
        with self.lock:
            # Create metric
            metric = ToolExecutionMetric(
                tool_name=tool_name,
                execution_time=execution_time,
                success=success,
                timestamp=datetime.utcnow(),
                user_context=user_context,
                error_message=error_message,
                memory_usage=memory_usage
            )
            
            # Add to history
            self.metrics_history.append(metric)
            
            # Update tool-specific stats
            if tool_name not in self.tool_stats:
                self.tool_stats[tool_name] = PerformanceStats(tool_name=tool_name)
            
            self.tool_stats[tool_name].update(metric)
            
            # Update system stats
            self.system_stats["total_tools_executed"] += 1
            if not success:
                self.system_stats["total_errors"] += 1
            
            # Check for performance issues
            self._check_performance_alerts(metric)
            
            logger.debug(f"Recorded execution: {tool_name} in {execution_time:.1f}ms ({'success' if success else 'failed'})")
    
    def _check_performance_alerts(self, metric: ToolExecutionMetric):
        """Check for performance alerts based on new metric."""
        alerts = []
        
        # Check execution time
        if metric.execution_time > self.performance_thresholds["execution_time_critical"]:
            alerts.append({
                "type": "critical",
                "category": "performance",
                "message": f"Tool {metric.tool_name} execution time critical: {metric.execution_time:.1f}ms",
                "threshold": self.performance_thresholds["execution_time_critical"],
                "actual_value": metric.execution_time,
                "timestamp": metric.timestamp.isoformat()
            })
        elif metric.execution_time > self.performance_thresholds["execution_time_warning"]:
            alerts.append({
                "type": "warning",
                "category": "performance",
                "message": f"Tool {metric.tool_name} execution time warning: {metric.execution_time:.1f}ms",
                "threshold": self.performance_thresholds["execution_time_warning"],
                "actual_value": metric.execution_time,
                "timestamp": metric.timestamp.isoformat()
            })
        
        # Check error rate for tool
        tool_stats = self.tool_stats.get(metric.tool_name)
        if tool_stats and tool_stats.total_executions >= 5:  # Only check after 5+ executions
            if tool_stats.error_rate > self.performance_thresholds["error_rate_critical"]:
                alerts.append({
                    "type": "critical",
                    "category": "reliability",
                    "message": f"Tool {metric.tool_name} error rate critical: {tool_stats.error_rate:.1%}",
                    "threshold": self.performance_thresholds["error_rate_critical"],
                    "actual_value": tool_stats.error_rate,
                    "timestamp": metric.timestamp.isoformat()
                })
            elif tool_stats.error_rate > self.performance_thresholds["error_rate_warning"]:
                alerts.append({
                    "type": "warning",
                    "category": "reliability",
                    "message": f"Tool {metric.tool_name} error rate warning: {tool_stats.error_rate:.1%}",
                    "threshold": self.performance_thresholds["error_rate_warning"],
                    "actual_value": tool_stats.error_rate,
                    "timestamp": metric.timestamp.isoformat()
                })
        
        # Add alerts to health status
        if alerts:
            self.health_status.alerts.extend(alerts)
            # Keep only recent alerts (last 50)
            self.health_status.alerts = self.health_status.alerts[-50:]
            
            # Update health status based on alert severity
            has_critical = any(alert["type"] == "critical" for alert in alerts)
            has_warning = any(alert["type"] == "warning" for alert in alerts)
            
            if has_critical:
                self.health_status.status = "unhealthy"
            elif has_warning and self.health_status.status == "healthy":
                self.health_status.status = "degraded"
    
    def get_tool_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for tools."""
        with self.lock:
            if tool_name:
                stats = self.tool_stats.get(tool_name)
                return stats.to_dict() if stats else {}
            else:
                return {
                    name: stats.to_dict() 
                    for name, stats in self.tool_stats.items()
                }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        with self.lock:
            current_time = datetime.utcnow()
            uptime = current_time - self.system_stats["start_time"]
            
            # Calculate recent metrics (last hour)
            recent_cutoff = current_time - timedelta(hours=1)
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp > recent_cutoff
            ]
            
            # Calculate percentiles for recent executions
            recent_times = [m.execution_time for m in recent_metrics]
            percentiles = {}
            if recent_times:
                percentiles = {
                    "p50": statistics.median(recent_times),
                    "p95": statistics.quantiles(recent_times, n=20)[18] if len(recent_times) >= 20 else max(recent_times),
                    "p99": statistics.quantiles(recent_times, n=100)[98] if len(recent_times) >= 100 else max(recent_times)
                }
            
            return {
                "uptime_seconds": uptime.total_seconds(),
                "total_tools_executed": self.system_stats["total_tools_executed"],
                "active_connections": self.system_stats["active_connections"],
                "total_errors": self.system_stats["total_errors"],
                "error_rate": (self.system_stats["total_errors"] / max(self.system_stats["total_tools_executed"], 1)),
                "recent_executions": len(recent_metrics),
                "execution_time_percentiles": percentiles,
                "metrics_history_size": len(self.metrics_history),
                "unique_tools_used": len(self.tool_stats),
                "timestamp": current_time.isoformat()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self.lock:
            # Update health checks
            self._update_health_checks()
            return self.health_status.to_dict()
    
    def _update_health_checks(self):
        """Update health status checks."""
        current_time = datetime.utcnow()
        
        # Check overall system health
        system_metrics = self.get_system_metrics()
        
        checks = {}
        
        # Check error rate
        error_rate = system_metrics.get("error_rate", 0)
        if error_rate > self.performance_thresholds["error_rate_critical"]:
            checks["error_rate"] = {
                "status": "unhealthy",
                "message": f"System error rate critical: {error_rate:.1%}",
                "value": error_rate
            }
        elif error_rate > self.performance_thresholds["error_rate_warning"]:
            checks["error_rate"] = {
                "status": "degraded",
                "message": f"System error rate elevated: {error_rate:.1%}",
                "value": error_rate
            }
        else:
            checks["error_rate"] = {
                "status": "healthy",
                "message": f"Error rate normal: {error_rate:.1%}",
                "value": error_rate
            }
        
        # Check response time percentiles
        percentiles = system_metrics.get("execution_time_percentiles", {})
        p95_time = percentiles.get("p95", 0)
        
        if p95_time > self.performance_thresholds["response_time_p95"]:
            checks["response_time"] = {
                "status": "degraded",
                "message": f"95th percentile response time elevated: {p95_time:.1f}ms",
                "value": p95_time
            }
        else:
            checks["response_time"] = {
                "status": "healthy",
                "message": f"Response times normal (p95: {p95_time:.1f}ms)",
                "value": p95_time
            }
        
        # Check for recent activity
        recent_executions = system_metrics.get("recent_executions", 0)
        checks["activity"] = {
            "status": "healthy",
            "message": f"Recent activity: {recent_executions} executions in last hour",
            "value": recent_executions
        }
        
        # Check monitoring system itself
        checks["monitoring"] = {
            "status": "healthy" if self.monitoring_active else "unhealthy",
            "message": f"Performance monitoring {'active' if self.monitoring_active else 'inactive'}",
            "value": self.monitoring_active
        }
        
        # Update health status
        self.health_status.checks = checks
        self.health_status.timestamp = current_time
        
        # Determine overall status
        statuses = [check["status"] for check in checks.values()]
        if "unhealthy" in statuses:
            self.health_status.status = "unhealthy"
        elif "degraded" in statuses:
            self.health_status.status = "degraded"
        else:
            self.health_status.status = "healthy"
    
    def get_performance_report(self, time_window: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self.lock:
            # Determine time window
            if time_window == "1h":
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
            elif time_window == "24h":
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
            elif time_window == "7d":
                cutoff_time = datetime.utcnow() - timedelta(days=7)
            else:
                cutoff_time = None  # All time
            
            # Filter metrics by time window
            if cutoff_time:
                filtered_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            else:
                filtered_metrics = list(self.metrics_history)
            
            # Calculate report metrics
            total_executions = len(filtered_metrics)
            successful_executions = sum(1 for m in filtered_metrics if m.success)
            failed_executions = total_executions - successful_executions
            
            execution_times = [m.execution_time for m in filtered_metrics]
            
            # Tool usage breakdown
            tool_usage = defaultdict(int)
            tool_errors = defaultdict(int)
            
            for metric in filtered_metrics:
                tool_usage[metric.tool_name] += 1
                if not metric.success:
                    tool_errors[metric.tool_name] += 1
            
            # Most used tools
            most_used_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Tools with highest error rates
            problematic_tools = []
            for tool, usage_count in tool_usage.items():
                error_count = tool_errors.get(tool, 0)
                error_rate = error_count / usage_count if usage_count > 0 else 0
                if error_rate > 0:
                    problematic_tools.append({
                        "tool": tool,
                        "usage_count": usage_count,
                        "error_count": error_count,
                        "error_rate": error_rate
                    })
            
            problematic_tools.sort(key=lambda x: x["error_rate"], reverse=True)
            
            # Performance summary
            performance_summary = {}
            if execution_times:
                performance_summary = {
                    "avg_execution_time": statistics.mean(execution_times),
                    "median_execution_time": statistics.median(execution_times),
                    "min_execution_time": min(execution_times),
                    "max_execution_time": max(execution_times),
                    "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                }
            
            return {
                "report_period": time_window or "all_time",
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {
                    "total_executions": total_executions,
                    "successful_executions": successful_executions,
                    "failed_executions": failed_executions,
                    "success_rate": successful_executions / max(total_executions, 1),
                    "unique_tools_used": len(tool_usage)
                },
                "performance": performance_summary,
                "tool_usage": {
                    "most_used": most_used_tools[:5],
                    "problematic_tools": problematic_tools[:5]
                },
                "health": self.get_health_status(),
                "recommendations": self._generate_recommendations(filtered_metrics)
            }
    
    def _generate_recommendations(self, metrics: List[ToolExecutionMetric]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        if not metrics:
            return ["No recent activity to analyze"]
        
        # Analyze execution times
        execution_times = [m.execution_time for m in metrics]
        avg_time = statistics.mean(execution_times)
        
        if avg_time > 1000:
            recommendations.append("Consider optimizing tool execution - average response time is high")
        
        # Analyze error patterns
        errors = [m for m in metrics if not m.success]
        if len(errors) > len(metrics) * 0.1:  # More than 10% error rate
            recommendations.append("Investigate error patterns - error rate is elevated")
        
        # Analyze tool usage patterns
        tool_usage = defaultdict(int)
        for metric in metrics:
            tool_usage[metric.tool_name] += 1
        
        # Check for imbalanced tool usage
        if tool_usage:
            most_used = max(tool_usage.values())
            least_used = min(tool_usage.values())
            if most_used > least_used * 10:  # Imbalanced usage
                recommendations.append("Consider load balancing - some tools are heavily used")
        
        # Memory usage recommendations
        memory_metrics = [m.memory_usage for m in metrics if m.memory_usage is not None]
        if memory_metrics and statistics.mean(memory_metrics) > 100:  # MB
            recommendations.append("Monitor memory usage - some operations are memory intensive")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations
    
    def set_connection_count(self, count: int):
        """Update active connection count."""
        with self.lock:
            self.system_stats["active_connections"] = count
    
    def start_monitoring(self):
        """Start background monitoring tasks."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_task = asyncio.create_task(self._background_monitor())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _background_monitor(self):
        """Background monitoring task."""
        while self.monitoring_active:
            try:
                # Update health status periodically
                with self.lock:
                    self._update_health_checks()
                
                # Clean up old alerts (older than 1 hour)
                cutoff = datetime.utcnow() - timedelta(hours=1)
                with self.lock:
                    self.health_status.alerts = [
                        alert for alert in self.health_status.alerts
                        if datetime.fromisoformat(alert["timestamp"]) > cutoff
                    ]
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background monitor: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        with self.lock:
            if format_type == "json":
                export_data = {
                    "system_metrics": self.get_system_metrics(),
                    "tool_stats": self.get_tool_stats(),
                    "health_status": self.get_health_status(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(export_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")


# Global performance monitor instance
performance_monitor = MCPPerformanceMonitor()


# Decorator for automatic performance monitoring
def monitor_tool_execution(func: Callable) -> Callable:
    """Decorator to automatically monitor tool execution performance."""
    
    async def async_wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            result = await func(*args, **kwargs)
            success = result.get("success", False) if isinstance(result, dict) else True
            return result
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            performance_monitor.record_tool_execution(
                tool_name=tool_name,
                execution_time=execution_time,
                success=success,
                error_message=error_message
            )
    
    def sync_wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            result = func(*args, **kwargs)
            success = result.get("success", False) if isinstance(result, dict) else True
            return result
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            performance_monitor.record_tool_execution(
                tool_name=tool_name,
                execution_time=execution_time,
                success=success,
                error_message=error_message
            )
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper