"""
Performance Monitoring System

Provides comprehensive performance monitoring including:
- Real-time performance metrics
- System resource monitoring
- Performance alerting
- Automated optimization recommendations
- Performance dashboards
"""

import asyncio
import time
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Mock psutil functions if not available
    class MockPsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 0.0
        @staticmethod
        def virtual_memory():
            class MockMemory:
                percent = 0.0
                used = 0
                available = 1024 * 1024 * 1024
            return MockMemory()
        @staticmethod
        def disk_usage(path):
            class MockDisk:
                percent = 0.0
                free = 1024 * 1024 * 1024
            return MockDisk()
        @staticmethod
        def net_io_counters():
            class MockNetwork:
                bytes_sent = 0
                bytes_recv = 0
            return MockNetwork()
        @staticmethod
        def getloadavg():
            return [0.0, 0.0, 0.0]
    psutil = MockPsutil()
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class SystemResources:
    """System resource usage snapshot"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    timestamp: float


@dataclass
class PerformanceAlert:
    """Performance alert"""
    name: str
    level: AlertLevel
    message: str
    threshold: float
    current_value: float
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MetricCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
    def record_counter(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """Record counter metric"""
        self.counters[name] += value
        metric = PerformanceMetric(
            name=name,
            value=self.counters[name],
            metric_type=MetricType.COUNTER,
            timestamp=time.time(),
            tags=tags
        )
        self.metrics[name].append(metric)
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record gauge metric"""
        self.gauges[name] = value
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            tags=tags
        )
        self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record histogram metric"""
        self.histograms[name].append(value)
        
        # Keep only recent values
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=time.time(),
            tags=tags
        )
        self.metrics[name].append(metric)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record timer metric"""
        self.timers[name].append(duration)
        
        # Keep only recent values
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-1000:]
        
        metric = PerformanceMetric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            timestamp=time.time(),
            tags=tags
        )
        self.metrics[name].append(metric)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        if name not in self.metrics:
            return {}
        
        recent_metrics = list(self.metrics[name])
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        metric_type = recent_metrics[0].metric_type
        
        summary = {
            'name': name,
            'type': metric_type.value,
            'count': len(values),
            'latest_value': values[-1] if values else 0,
            'latest_timestamp': recent_metrics[-1].timestamp if recent_metrics else 0
        }
        
        if metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
            if values:
                sorted_values = sorted(values)
                summary.update({
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'median': sorted_values[len(sorted_values) // 2],
                    'p95': sorted_values[int(len(sorted_values) * 0.95)],
                    'p99': sorted_values[int(len(sorted_values) * 0.99)]
                })
        
        return summary
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metric summaries"""
        return {name: self.get_metric_summary(name) for name in self.metrics.keys()}


class SystemMonitor:
    """Monitors system resources"""
    
    def __init__(self):
        self.history: deque = deque(maxlen=1000)
        self.network_io_start = psutil.net_io_counters()
        
    def get_current_resources(self) -> SystemResources:
        """Get current system resource usage"""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        
        # Load average (Unix only)
        try:
            load_avg = list(psutil.getloadavg())
        except AttributeError:
            load_avg = [0.0, 0.0, 0.0]
        
        resources = SystemResources(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            load_average=load_avg,
            timestamp=time.time()
        )
        
        self.history.append(resources)
        return resources
    
    def get_resource_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get resource usage trends"""
        cutoff_time = time.time() - (hours * 3600)
        recent_resources = [r for r in self.history if r.timestamp > cutoff_time]
        
        if not recent_resources:
            return {}
        
        # Calculate trends
        cpu_values = [r.cpu_percent for r in recent_resources]
        memory_values = [r.memory_percent for r in recent_resources]
        disk_values = [r.disk_usage_percent for r in recent_resources]
        
        return {
            'cpu': {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'min': min(cpu_values) if cpu_values else 0
            },
            'memory': {
                'current': memory_values[-1] if memory_values else 0,
                'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'min': min(memory_values) if memory_values else 0
            },
            'disk': {
                'current': disk_values[-1] if disk_values else 0,
                'average': sum(disk_values) / len(disk_values) if disk_values else 0,
                'max': max(disk_values) if disk_values else 0,
                'min': min(disk_values) if disk_values else 0
            },
            'sample_count': len(recent_resources),
            'time_period_hours': hours
        }


class AlertManager:
    """Manages performance alerts"""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
    def add_alert_rule(self, 
                      name: str, 
                      metric_name: str, 
                      threshold: float, 
                      comparison: str = "gt",
                      level: AlertLevel = AlertLevel.WARNING,
                      cooldown: float = 300):
        """Add performance alert rule"""
        self.alert_rules[name] = {
            'metric_name': metric_name,
            'threshold': threshold,
            'comparison': comparison,  # gt, lt, eq
            'level': level,
            'cooldown': cooldown,
            'last_triggered': 0
        }
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        current_time = time.time()
        
        for alert_name, rule in self.alert_rules.items():
            metric_name = rule['metric_name']
            
            if metric_name not in metrics:
                continue
            
            metric_summary = metrics[metric_name]
            current_value = metric_summary.get('latest_value', 0)
            threshold = rule['threshold']
            comparison = rule['comparison']
            
            # Check if alert condition is met
            triggered = False
            if comparison == "gt" and current_value > threshold:
                triggered = True
            elif comparison == "lt" and current_value < threshold:
                triggered = True
            elif comparison == "eq" and abs(current_value - threshold) < 0.001:
                triggered = True
            
            # Check cooldown
            if triggered and (current_time - rule['last_triggered']) > rule['cooldown']:
                self._trigger_alert(alert_name, rule, current_value)
                rule['last_triggered'] = current_time
            elif not triggered and alert_name in self.active_alerts:
                self._resolve_alert(alert_name)
    
    def _trigger_alert(self, alert_name: str, rule: Dict[str, Any], current_value: float):
        """Trigger an alert"""
        alert = PerformanceAlert(
            name=alert_name,
            level=rule['level'],
            message=f"Metric {rule['metric_name']} is {current_value:.2f}, threshold is {rule['threshold']:.2f}",
            threshold=rule['threshold'],
            current_value=current_value,
            timestamp=time.time()
        )
        
        self.active_alerts[alert_name] = alert
        self.alert_history.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"Performance alert triggered: {alert.message}")
    
    def _resolve_alert(self, alert_name: str):
        """Resolve an active alert"""
        if alert_name in self.active_alerts:
            del self.active_alerts[alert_name]
            logger.info(f"Performance alert resolved: {alert_name}")
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alert history"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]


class PerformanceOptimizer:
    """Provides automated performance optimization recommendations"""
    
    def __init__(self):
        self.recommendations: List[Dict[str, Any]] = []
        
    def analyze_performance(self, 
                          metrics: Dict[str, Any], 
                          system_resources: SystemResources) -> List[Dict[str, Any]]:
        """Analyze performance and provide recommendations"""
        
        recommendations = []
        
        # CPU recommendations
        if system_resources.cpu_percent > 80:
            recommendations.append({
                'category': 'cpu',
                'priority': 'high',
                'issue': 'High CPU usage detected',
                'current_value': system_resources.cpu_percent,
                'recommendations': [
                    'Consider scaling horizontally by adding more instances',
                    'Review CPU-intensive operations and optimize algorithms',
                    'Implement caching to reduce computational overhead',
                    'Consider using async processing for I/O operations'
                ]
            })
        
        # Memory recommendations
        if system_resources.memory_percent > 85:
            recommendations.append({
                'category': 'memory',
                'priority': 'high',
                'issue': 'High memory usage detected',
                'current_value': system_resources.memory_percent,
                'recommendations': [
                    'Implement memory caching strategies with TTL',
                    'Review object lifecycle and garbage collection',
                    'Consider database connection pooling optimization',
                    'Implement data streaming for large datasets'
                ]
            })
        
        # Database performance recommendations
        db_metrics = {k: v for k, v in metrics.items() if 'database' in k or 'query' in k}
        if db_metrics:
            slow_queries = []
            for metric_name, metric_data in db_metrics.items():
                if 'response_time' in metric_name or 'duration' in metric_name:
                    avg_time = metric_data.get('mean', 0)
                    if avg_time > 1.0:  # Slower than 1 second
                        slow_queries.append(metric_name)
            
            if slow_queries:
                recommendations.append({
                    'category': 'database',
                    'priority': 'medium',
                    'issue': 'Slow database queries detected',
                    'affected_queries': slow_queries,
                    'recommendations': [
                        'Add database indexes for frequently queried columns',
                        'Implement query result caching',
                        'Consider database connection pooling',
                        'Review and optimize SQL queries',
                        'Implement read replicas for read-heavy workloads'
                    ]
                })
        
        # API response time recommendations
        api_metrics = {k: v for k, v in metrics.items() if 'api' in k or 'response' in k}
        if api_metrics:
            slow_endpoints = []
            for metric_name, metric_data in api_metrics.items():
                if 'response_time' in metric_name:
                    avg_time = metric_data.get('mean', 0)
                    if avg_time > 0.5:  # Slower than 500ms
                        slow_endpoints.append(metric_name)
            
            if slow_endpoints:
                recommendations.append({
                    'category': 'api_performance',
                    'priority': 'medium',
                    'issue': 'Slow API endpoints detected',
                    'affected_endpoints': slow_endpoints,
                    'recommendations': [
                        'Implement response caching with ETags',
                        'Enable response compression',
                        'Optimize database queries in endpoints',
                        'Consider async processing for heavy operations',
                        'Implement pagination for large result sets'
                    ]
                })
        
        # Cache efficiency recommendations
        cache_metrics = {k: v for k, v in metrics.items() if 'cache' in k}
        if cache_metrics:
            for metric_name, metric_data in cache_metrics.items():
                if 'hit_rate' in metric_name:
                    hit_rate = metric_data.get('latest_value', 0)
                    if hit_rate < 0.7:  # Less than 70% hit rate
                        recommendations.append({
                            'category': 'caching',
                            'priority': 'medium',
                            'issue': 'Low cache hit rate detected',
                            'current_hit_rate': hit_rate,
                            'recommendations': [
                                'Review cache TTL settings',
                                'Implement cache warming strategies',
                                'Optimize cache key generation',
                                'Consider increasing cache size',
                                'Review cache eviction policies'
                            ]
                        })
        
        self.recommendations = recommendations
        return recommendations
    
    def get_optimization_priority(self) -> List[Dict[str, Any]]:
        """Get prioritized optimization recommendations"""
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        return sorted(
            self.recommendations,
            key=lambda x: priority_order.get(x.get('priority', 'low'), 3)
        )


class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self):
        self.metric_collector = MetricCollector()
        self.system_monitor = SystemMonitor()
        self.alert_manager = AlertManager()
        self.optimizer = PerformanceOptimizer()
        
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 60  # seconds
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default performance alert rules"""
        
        # System resource alerts
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            "system_cpu_percent",
            threshold=80.0,
            comparison="gt",
            level=AlertLevel.WARNING
        )
        
        self.alert_manager.add_alert_rule(
            "high_memory_usage", 
            "system_memory_percent",
            threshold=85.0,
            comparison="gt",
            level=AlertLevel.ERROR
        )
        
        self.alert_manager.add_alert_rule(
            "low_disk_space",
            "system_disk_usage_percent",
            threshold=90.0,
            comparison="gt",
            level=AlertLevel.CRITICAL
        )
        
        # Application performance alerts
        self.alert_manager.add_alert_rule(
            "slow_api_response",
            "api_response_time_mean",
            threshold=1.0,
            comparison="gt",
            level=AlertLevel.WARNING
        )
        
        self.alert_manager.add_alert_rule(
            "low_cache_hit_rate",
            "cache_hit_rate",
            threshold=0.5,
            comparison="lt",
            level=AlertLevel.WARNING
        )
    
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect system resources
                system_resources = self.system_monitor.get_current_resources()
                
                # Record system metrics
                self.metric_collector.record_gauge(
                    "system_cpu_percent", 
                    system_resources.cpu_percent
                )
                self.metric_collector.record_gauge(
                    "system_memory_percent", 
                    system_resources.memory_percent
                )
                self.metric_collector.record_gauge(
                    "system_disk_usage_percent", 
                    system_resources.disk_usage_percent
                )
                
                # Get all metrics
                all_metrics = self.metric_collector.get_all_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts(all_metrics)
                
                # Generate optimization recommendations
                self.optimizer.analyze_performance(all_metrics, system_resources)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     metric_type: MetricType, 
                     tags: Dict[str, str] = None):
        """Record a performance metric"""
        
        if metric_type == MetricType.COUNTER:
            self.metric_collector.record_counter(name, value, tags)
        elif metric_type == MetricType.GAUGE:
            self.metric_collector.record_gauge(name, value, tags)
        elif metric_type == MetricType.HISTOGRAM:
            self.metric_collector.record_histogram(name, value, tags)
        elif metric_type == MetricType.TIMER:
            self.metric_collector.record_timer(name, value, tags)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        # Current system resources
        current_resources = self.system_monitor.get_current_resources()
        
        # Resource trends
        resource_trends = self.system_monitor.get_resource_trends(hours=1)
        
        # All metrics
        all_metrics = self.metric_collector.get_all_metrics()
        
        # Active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Optimization recommendations
        optimization_recommendations = self.optimizer.get_optimization_priority()
        
        return {
            'timestamp': time.time(),
            'system_resources': asdict(current_resources),
            'resource_trends': resource_trends,
            'metrics': all_metrics,
            'active_alerts': [asdict(alert) for alert in active_alerts],
            'recommendations': optimization_recommendations,
            'monitoring_status': {
                'is_running': self.monitoring_task is not None,
                'interval_seconds': self.monitoring_interval
            }
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get monitoring system health check"""
        current_resources = self.system_monitor.get_current_resources()
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Determine overall health
        health_status = "healthy"
        if any(alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] for alert in active_alerts):
            health_status = "unhealthy"
        elif any(alert.level == AlertLevel.WARNING for alert in active_alerts):
            health_status = "degraded"
        
        return {
            'status': health_status,
            'timestamp': time.time(),
            'system_health': {
                'cpu_percent': current_resources.cpu_percent,
                'memory_percent': current_resources.memory_percent,
                'disk_usage_percent': current_resources.disk_usage_percent
            },
            'active_alerts_count': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            'monitoring_active': self.monitoring_task is not None
        }


# Global performance monitor instance
performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
    return performance_monitor


async def initialize_performance_monitor():
    """Initialize the global performance monitor"""
    global performance_monitor
    performance_monitor = get_performance_monitor()
    await performance_monitor.start_monitoring()


async def shutdown_performance_monitor():
    """Shutdown the global performance monitor"""
    global performance_monitor
    if performance_monitor:
        await performance_monitor.stop_monitoring()