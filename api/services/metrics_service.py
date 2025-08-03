"""
Real-time Metrics Collection and Analysis Service.

This service provides:
- Real-time metric collection
- Metric aggregation and processing
- Performance monitoring
- Alert generation
- Trend analysis
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, and_, or_, desc

from ..models.base import get_db_context
from ..models.memory import MemoryItem
from ..models.session import Session
from ..models.error_pattern import ErrorOccurrence
from ..models.workflow import WorkflowExecution
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("metrics_service")


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MetricAggregation:
    """Aggregated metric data."""
    name: str
    metric_type: MetricType
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    percentiles: Dict[str, float]
    time_window: str
    start_time: datetime
    end_time: datetime
    tags: Dict[str, str] = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # e.g., "gt", "lt", "eq"
    threshold: float
    severity: AlertSeverity
    duration_minutes: int = 5
    cooldown_minutes: int = 30
    tags_filter: Dict[str, str] = None
    is_active: bool = True


@dataclass
class Alert:
    """Active alert instance."""
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    triggered_at: datetime
    description: str
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None


class MetricsService:
    """
    Comprehensive metrics collection and analysis service.
    
    Features:
    - Real-time metric collection
    - Time-series aggregation
    - Alert management
    - Performance analytics
    - Custom dashboard support
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.analytics_service = TimeSeriesAnalyticsService()
        
        # Service state
        self._running = False
        self._metric_buffer = []
        self._alert_rules = {}
        self._active_alerts = {}
        self._metric_cache = {}
        
        # Configuration
        self.buffer_size = 1000
        self.flush_interval = 10  # seconds
        self.retention_days = 30
        self.aggregation_intervals = ["1m", "5m", "15m", "1h", "1d"]
        
        # Performance tracking
        self._performance_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "avg_processing_time": 0.0
        }
        
        logger.info("Initialized MetricsService")
    
    async def initialize(self):
        """Initialize the metrics service."""
        try:
            await redis_client.initialize()
            await self.analytics_service.initialize()
            
            # Load alert rules
            await self._load_alert_rules()
            
            # Start background tasks
            asyncio.create_task(self._metric_flush_loop())
            asyncio.create_task(self._aggregation_loop())
            asyncio.create_task(self._alert_evaluation_loop())
            asyncio.create_task(self._cleanup_loop())
            
            self._running = True
            logger.info("MetricsService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MetricsService: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup service resources."""
        self._running = False
        
        # Flush remaining metrics
        if self._metric_buffer:
            await self._flush_metrics()
        
        await self.analytics_service.cleanup()
        logger.info("MetricsService cleaned up")
    
    async def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a single metric point."""
        try:
            metric_point = MetricPoint(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            # Add to buffer
            self._metric_buffer.append(metric_point)
            
            # Update performance stats
            self._performance_stats["metrics_collected"] += 1
            
            # Immediate flush if buffer is full
            if len(self._metric_buffer) >= self.buffer_size:
                await self._flush_metrics()
            
            # Cache recent metrics for real-time queries
            await self._cache_metric(metric_point)
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    async def record_counter(
        self,
        name: str,
        increment: Union[int, float] = 1,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a counter metric."""
        await self.record_metric(
            name=name,
            value=increment,
            metric_type=MetricType.COUNTER,
            tags=tags
        )
    
    async def record_timer(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a timer metric."""
        await self.record_metric(
            name=name,
            value=duration_ms,
            metric_type=MetricType.TIMER,
            tags=tags
        )
    
    async def record_histogram(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a histogram metric."""
        await self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            tags=tags
        )
    
    async def get_metric_aggregation(
        self,
        metric_name: str,
        time_window: str = "1h",
        tags_filter: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[MetricAggregation]:
        """Get aggregated metric data for a time window."""
        try:
            # Parse time window
            window_seconds = self._parse_time_window(time_window)
            
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(seconds=window_seconds)
            
            # Get raw metrics from cache/storage
            metrics = await self._get_metrics_in_range(
                metric_name, start_time, end_time, tags_filter
            )
            
            if not metrics:
                return None
            
            # Calculate aggregations
            values = [m.value for m in metrics]
            
            aggregation = MetricAggregation(
                name=metric_name,
                metric_type=metrics[0].metric_type,
                count=len(values),
                sum_value=sum(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=np.mean(values),
                percentiles={
                    "p50": np.percentile(values, 50),
                    "p90": np.percentile(values, 90),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                },
                time_window=time_window,
                start_time=start_time,
                end_time=end_time,
                tags=tags_filter or {}
            )
            
            return aggregation
            
        except Exception as e:
            logger.error(f"Failed to get metric aggregation: {e}")
            return None
    
    async def get_metrics_dashboard_data(
        self,
        time_window: str = "1h",
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            dashboard_data = {
                "summary": {},
                "performance": {},
                "system": {},
                "alerts": [],
                "trends": {},
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # System metrics
            system_metrics = await self._get_system_metrics(time_window)
            dashboard_data["system"] = system_metrics
            
            # Performance metrics
            performance_metrics = await self._get_performance_metrics(
                time_window, user_id, project_id
            )
            dashboard_data["performance"] = performance_metrics
            
            # Active alerts
            alerts = await self._get_active_alerts()
            dashboard_data["alerts"] = [asdict(alert) for alert in alerts]
            
            # Summary statistics
            summary = await self._generate_summary_stats(time_window)
            dashboard_data["summary"] = summary
            
            # Trend analysis
            trends = await self._generate_trend_analysis(time_window)
            dashboard_data["trends"] = trends
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}
    
    async def create_alert_rule(self, rule: AlertRule):
        """Create a new alert rule."""
        try:
            self._alert_rules[rule.name] = rule
            
            # Store in Redis for persistence
            await redis_client.set(
                f"alert_rule:{rule.name}",
                json.dumps(asdict(rule)),
                ex=86400 * 7  # 7 days
            )
            
            logger.info(f"Created alert rule: {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
    
    async def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return list(self._alert_rules.values())
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return await self._get_active_alerts()
    
    async def resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        try:
            if rule_name in self._active_alerts:
                del self._active_alerts[rule_name]
                
                # Remove from Redis
                await redis_client.delete(f"active_alert:{rule_name}")
                
                logger.info(f"Resolved alert: {rule_name}")
                
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
    
    async def get_metric_trends(
        self,
        metric_names: List[str],
        time_window: str = "24h",
        tags_filter: Optional[Dict[str, str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get trend analysis for multiple metrics."""
        try:
            trends = {}
            
            for metric_name in metric_names:
                trend_data = await self._analyze_metric_trend(
                    metric_name, time_window, tags_filter
                )
                trends[metric_name] = trend_data
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get metric trends: {e}")
            return {}
    
    async def export_metrics(
        self,
        format_type: str = "json",
        time_window: str = "1h",
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Export metrics in various formats."""
        try:
            # Get metrics data
            end_time = datetime.utcnow()
            window_seconds = self._parse_time_window(time_window)
            start_time = end_time - timedelta(seconds=window_seconds)
            
            metrics_data = []
            
            if metric_names:
                for metric_name in metric_names:
                    metrics = await self._get_metrics_in_range(
                        metric_name, start_time, end_time
                    )
                    metrics_data.extend(metrics)
            else:
                # Get all metrics
                metrics_data = await self._get_all_metrics_in_range(
                    start_time, end_time
                )
            
            # Format data
            if format_type == "json":
                return {
                    "metrics": [asdict(m) for m in metrics_data],
                    "count": len(metrics_data),
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    }
                }
            elif format_type == "csv":
                # Convert to CSV format
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Header
                writer.writerow(["timestamp", "name", "value", "type", "tags"])
                
                # Data
                for metric in metrics_data:
                    writer.writerow([
                        metric.timestamp.isoformat(),
                        metric.name,
                        metric.value,
                        metric.metric_type.value,
                        json.dumps(metric.tags)
                    ])
                
                return {"csv_data": output.getvalue()}
            
            return {"error": "Unsupported format"}
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return {"error": str(e)}
    
    # Internal methods
    
    async def _flush_metrics(self):
        """Flush buffered metrics to storage."""
        if not self._metric_buffer:
            return
        
        try:
            start_time = time.time()
            
            # Store in TimescaleDB via analytics service
            for metric in self._metric_buffer:
                await self.analytics_service.record_metric(
                    metric_type=metric.name,
                    value=metric.value,
                    tags={
                        **metric.tags,
                        "metric_type": metric.metric_type.value
                    }
                )
            
            # Clear buffer
            flushed_count = len(self._metric_buffer)
            self._metric_buffer.clear()
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._performance_stats["avg_processing_time"] = (
                (self._performance_stats["avg_processing_time"] * 0.9) +
                (processing_time * 0.1)
            )
            
            logger.debug(f"Flushed {flushed_count} metrics in {processing_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    async def _cache_metric(self, metric: MetricPoint):
        """Cache metric for real-time queries."""
        try:
            cache_key = f"metric:{metric.name}"
            
            # Store in Redis with TTL
            metric_data = {
                "value": metric.value,
                "type": metric.metric_type.value,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags
            }
            
            await redis_client.set(
                cache_key,
                json.dumps(metric_data),
                ex=300  # 5 minutes
            )
            
            # Also add to time series for recent data
            ts_key = f"metrics_ts:{metric.name}"
            await redis_client.zadd(
                ts_key,
                {json.dumps(metric_data): metric.timestamp.timestamp()}
            )
            
            # Keep only recent data (1 hour)
            cutoff = (datetime.utcnow() - timedelta(hours=1)).timestamp()
            await redis_client.zremrangebyscore(ts_key, 0, cutoff)
            
        except Exception as e:
            logger.error(f"Failed to cache metric: {e}")
    
    async def _load_alert_rules(self):
        """Load alert rules from storage."""
        try:
            # Load from Redis
            pattern = "alert_rule:*"
            keys = await redis_client.keys(pattern)
            
            for key in keys:
                rule_data = await redis_client.get(key)
                if rule_data:
                    rule_dict = json.loads(rule_data)
                    rule = AlertRule(**rule_dict)
                    self._alert_rules[rule.name] = rule
            
            # Add default alert rules if none exist
            if not self._alert_rules:
                await self._create_default_alert_rules()
            
            logger.info(f"Loaded {len(self._alert_rules)} alert rules")
            
        except Exception as e:
            logger.error(f"Failed to load alert rules: {e}")
    
    async def _create_default_alert_rules(self):
        """Create default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                metric_name="error_count",
                condition="gt",
                threshold=10,
                severity=AlertSeverity.WARNING,
                duration_minutes=5
            ),
            AlertRule(
                name="high_response_time",
                metric_name="response_time",
                condition="gt",
                threshold=1000,
                severity=AlertSeverity.WARNING,
                duration_minutes=3
            ),
            AlertRule(
                name="low_memory_available",
                metric_name="memory_usage",
                condition="gt",
                threshold=0.9,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=2
            )
        ]
        
        for rule in default_rules:
            await self.create_alert_rule(rule)
    
    async def _metric_flush_loop(self):
        """Background loop to flush metrics."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
                
            except Exception as e:
                logger.error(f"Error in metric flush loop: {e}")
    
    async def _aggregation_loop(self):
        """Background loop for metric aggregation."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._perform_aggregations()
                
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
    
    async def _alert_evaluation_loop(self):
        """Background loop for alert evaluation."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._evaluate_alert_rules()
                
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
    
    async def _cleanup_loop(self):
        """Background loop for data cleanup."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _perform_aggregations(self):
        """Perform metric aggregations."""
        try:
            # Get unique metric names from recent data
            metric_names = await self._get_recent_metric_names()
            
            for metric_name in metric_names:
                for interval in self.aggregation_intervals:
                    aggregation = await self.get_metric_aggregation(
                        metric_name, interval
                    )
                    
                    if aggregation:
                        # Store aggregation
                        await self._store_aggregation(aggregation)
            
        except Exception as e:
            logger.error(f"Failed to perform aggregations: {e}")
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules."""
        try:
            for rule in self._alert_rules.values():
                if not rule.is_active:
                    continue
                
                await self._evaluate_single_alert_rule(rule)
                
        except Exception as e:
            logger.error(f"Failed to evaluate alert rules: {e}")
    
    async def _evaluate_single_alert_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        try:
            # Get recent metric data
            aggregation = await self.get_metric_aggregation(
                rule.metric_name,
                f"{rule.duration_minutes}m",
                rule.tags_filter
            )
            
            if not aggregation:
                return
            
            # Evaluate condition
            current_value = aggregation.avg_value
            should_trigger = self._evaluate_condition(
                current_value, rule.condition, rule.threshold
            )
            
            if should_trigger:
                # Check if already alerting and in cooldown
                if rule.name in self._active_alerts:
                    last_alert = self._active_alerts[rule.name]
                    cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
                    
                    if datetime.utcnow() - last_alert.triggered_at < cooldown_delta:
                        return  # Still in cooldown
                
                # Trigger alert
                alert = Alert(
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    triggered_at=datetime.utcnow(),
                    description=f"{rule.metric_name} {rule.condition} {rule.threshold} (current: {current_value})",
                    tags=rule.tags_filter or {}
                )
                
                self._active_alerts[rule.name] = alert
                
                # Store in Redis
                await redis_client.set(
                    f"active_alert:{rule.name}",
                    json.dumps(asdict(alert)),
                    ex=86400  # 24 hours
                )
                
                # Update performance stats
                self._performance_stats["alerts_triggered"] += 1
                
                logger.warning(f"Alert triggered: {rule.name} - {alert.description}")
                
            else:
                # Clear alert if it was active
                if rule.name in self._active_alerts:
                    await self.resolve_alert(rule.name)
                    
        except Exception as e:
            logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")
    
    def _evaluate_condition(
        self,
        value: float,
        condition: str,
        threshold: float
    ) -> bool:
        """Evaluate alert condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        else:
            return False
    
    def _parse_time_window(self, time_window: str) -> int:
        """Parse time window string to seconds."""
        if time_window.endswith("s"):
            return int(time_window[:-1])
        elif time_window.endswith("m"):
            return int(time_window[:-1]) * 60
        elif time_window.endswith("h"):
            return int(time_window[:-1]) * 3600
        elif time_window.endswith("d"):
            return int(time_window[:-1]) * 86400
        else:
            return int(time_window)  # Assume seconds
    
    async def _get_metrics_in_range(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> List[MetricPoint]:
        """Get metrics from time range."""
        try:
            # Try Redis time series first for recent data
            ts_key = f"metrics_ts:{metric_name}"
            start_ts = start_time.timestamp()
            end_ts = end_time.timestamp()
            
            redis_data = await redis_client.zrangebyscore(
                ts_key, start_ts, end_ts, withscores=True
            )
            
            metrics = []
            for data, timestamp in redis_data:
                metric_data = json.loads(data)
                
                # Apply tags filter
                if tags_filter:
                    if not all(
                        metric_data.get("tags", {}).get(k) == v
                        for k, v in tags_filter.items()
                    ):
                        continue
                
                metric = MetricPoint(
                    name=metric_name,
                    value=metric_data["value"],
                    metric_type=MetricType(metric_data["type"]),
                    timestamp=datetime.fromtimestamp(timestamp),
                    tags=metric_data.get("tags", {})
                )
                metrics.append(metric)
            
            # If not enough data in Redis, query TimescaleDB
            if len(metrics) < 10:
                # Query analytics service for more data
                ts_data = await self.analytics_service.query_metrics(
                    metric_types=[metric_name],
                    start_time=start_time,
                    end_time=end_time,
                    tags=tags_filter
                )
                
                # Convert to MetricPoint objects
                for point in ts_data:
                    metric = MetricPoint(
                        name=metric_name,
                        value=point.get("value", 0),
                        metric_type=MetricType.GAUGE,  # Default
                        timestamp=point.get("timestamp"),
                        tags=point.get("tags", {})
                    )
                    metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics in range: {e}")
            return []
    
    async def _get_system_metrics(self, time_window: str) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            system_metrics = {}
            
            # Memory usage
            memory_agg = await self.get_metric_aggregation("memory_usage", time_window)
            if memory_agg:
                system_metrics["memory"] = {
                    "current": memory_agg.avg_value,
                    "max": memory_agg.max_value,
                    "trend": "stable"  # Would calculate actual trend
                }
            
            # CPU usage
            cpu_agg = await self.get_metric_aggregation("cpu_usage", time_window)
            if cpu_agg:
                system_metrics["cpu"] = {
                    "current": cpu_agg.avg_value,
                    "max": cpu_agg.max_value,
                    "trend": "stable"
                }
            
            # Request count
            request_agg = await self.get_metric_aggregation("request_count", time_window)
            if request_agg:
                system_metrics["requests"] = {
                    "total": request_agg.sum_value,
                    "rate": request_agg.sum_value / self._parse_time_window(time_window),
                    "trend": "increasing"
                }
            
            return system_metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    async def _get_performance_metrics(
        self,
        time_window: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            performance_metrics = {}
            
            # Response time
            response_agg = await self.get_metric_aggregation("response_time", time_window)
            if response_agg:
                performance_metrics["response_time"] = {
                    "avg": response_agg.avg_value,
                    "p95": response_agg.percentiles.get("p95", 0),
                    "p99": response_agg.percentiles.get("p99", 0)
                }
            
            # Error rate
            error_agg = await self.get_metric_aggregation("error_count", time_window)
            if error_agg:
                performance_metrics["error_rate"] = {
                    "count": error_agg.sum_value,
                    "rate": error_agg.sum_value / self._parse_time_window(time_window)
                }
            
            # Database query time
            db_agg = await self.get_metric_aggregation("db_query_time", time_window)
            if db_agg:
                performance_metrics["database"] = {
                    "avg_query_time": db_agg.avg_value,
                    "slow_queries": len([v for v in [db_agg.max_value] if v > 1000])
                }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    async def _get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        try:
            alerts = []
            
            # Get from Redis
            pattern = "active_alert:*"
            keys = await redis_client.keys(pattern)
            
            for key in keys:
                alert_data = await redis_client.get(key)
                if alert_data:
                    alert_dict = json.loads(alert_data)
                    # Convert timestamp strings back to datetime
                    alert_dict["triggered_at"] = datetime.fromisoformat(
                        alert_dict["triggered_at"]
                    )
                    alert = Alert(**alert_dict)
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    async def _generate_summary_stats(self, time_window: str) -> Dict[str, Any]:
        """Generate summary statistics."""
        try:
            summary = {
                "total_metrics": self._performance_stats["metrics_collected"],
                "active_alerts": len(self._active_alerts),
                "avg_processing_time": self._performance_stats["avg_processing_time"]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary stats: {e}")
            return {}
    
    async def _generate_trend_analysis(self, time_window: str) -> Dict[str, Any]:
        """Generate trend analysis."""
        try:
            trends = {}
            
            # Get trends for key metrics
            key_metrics = ["response_time", "error_count", "memory_usage", "cpu_usage"]
            
            for metric_name in key_metrics:
                trend_data = await self._analyze_metric_trend(metric_name, time_window)
                trends[metric_name] = trend_data
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to generate trend analysis: {e}")
            return {}
    
    async def _analyze_metric_trend(
        self,
        metric_name: str,
        time_window: str,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Analyze trend for a single metric."""
        try:
            # Get metric data
            end_time = datetime.utcnow()
            window_seconds = self._parse_time_window(time_window)
            start_time = end_time - timedelta(seconds=window_seconds)
            
            metrics = await self._get_metrics_in_range(
                metric_name, start_time, end_time, tags_filter
            )
            
            if len(metrics) < 3:
                return {"trend": "insufficient_data"}
            
            # Calculate trend
            values = [m.value for m in sorted(metrics, key=lambda x: x.timestamp)]
            x = np.arange(len(values))
            
            # Linear regression
            slope, intercept = np.polyfit(x, values, 1)
            
            # Determine trend direction
            if abs(slope) < 0.01:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            # Calculate R-squared
            predicted = slope * x + intercept
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                "trend": trend_direction,
                "slope": slope,
                "confidence": r_squared,
                "current_value": values[-1],
                "change_rate": slope / np.mean(values) if np.mean(values) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze metric trend: {e}")
            return {"trend": "error", "error": str(e)}
    
    async def _get_recent_metric_names(self) -> List[str]:
        """Get list of recent metric names."""
        try:
            pattern = "metrics_ts:*"
            keys = await redis_client.keys(pattern)
            
            metric_names = []
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()
                metric_name = key.replace("metrics_ts:", "")
                metric_names.append(metric_name)
            
            return metric_names
            
        except Exception as e:
            logger.error(f"Failed to get recent metric names: {e}")
            return []
    
    async def _store_aggregation(self, aggregation: MetricAggregation):
        """Store metric aggregation."""
        try:
            # Store in Redis for fast access
            agg_key = f"agg:{aggregation.name}:{aggregation.time_window}"
            
            agg_data = asdict(aggregation)
            # Convert datetime objects to strings
            agg_data["start_time"] = aggregation.start_time.isoformat()
            agg_data["end_time"] = aggregation.end_time.isoformat()
            
            await redis_client.set(
                agg_key,
                json.dumps(agg_data),
                ex=86400 * 7  # 7 days
            )
            
        except Exception as e:
            logger.error(f"Failed to store aggregation: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old metric data."""
        try:
            # Clean old Redis time series data
            pattern = "metrics_ts:*"
            keys = await redis_client.keys(pattern)
            
            cutoff = (datetime.utcnow() - timedelta(days=1)).timestamp()
            
            for key in keys:
                await redis_client.zremrangebyscore(key, 0, cutoff)
            
            logger.debug("Cleaned up old metric data")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    async def _get_all_metrics_in_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[MetricPoint]:
        """Get all metrics in time range."""
        try:
            all_metrics = []
            
            # Get all metric names
            metric_names = await self._get_recent_metric_names()
            
            for metric_name in metric_names:
                metrics = await self._get_metrics_in_range(
                    metric_name, start_time, end_time
                )
                all_metrics.extend(metrics)
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Failed to get all metrics in range: {e}")
            return []


# Global metrics service instance
metrics_service = MetricsService()