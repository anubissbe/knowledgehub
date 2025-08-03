"""
Prometheus Metrics Collection Service

Real production-grade metrics collection and export for Prometheus monitoring.
Provides comprehensive system and application metrics with proper labels and naming.
"""

import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import psutil

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    
from shared.config import Config
from .real_health_monitor import real_health_monitor, HealthStatus

logger = logging.getLogger(__name__)

@dataclass
class MetricDefinition:
    """Definition for a Prometheus metric"""
    name: str
    description: str
    metric_type: str  # counter, gauge, histogram, summary
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms

class PrometheusMetrics:
    """
    Real Prometheus metrics collection service providing:
    - System resource metrics (CPU, memory, disk, network)
    - Application performance metrics (response times, throughput)
    - Health check metrics (service status, uptime)
    - Business metrics (API calls, user sessions, errors)
    - Custom metrics (memory operations, AI processing)
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.enabled = PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            logger.warning("Prometheus client not available. Metrics collection disabled.")
            return
            
        # Create custom registry for better control
        self.registry = CollectorRegistry()
        
        # System metrics
        self._init_system_metrics()
        
        # Application metrics
        self._init_application_metrics()
        
        # Health metrics
        self._init_health_metrics()
        
        # Business metrics
        self._init_business_metrics()
        
        # Custom KnowledgeHub metrics
        self._init_custom_metrics()
        
        # State tracking
        self.is_collecting = False
        self.collection_interval = 15  # seconds
        self.last_collection = datetime.now(timezone.utc)
        
        logger.info("Prometheus metrics service initialized")

    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics"""
        # CPU metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            ['core'],
            registry=self.registry
        )
        
        self.cpu_load = Gauge(
            'system_cpu_load_average',
            'System load average',
            ['period'],
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.memory_usage_percent = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'Disk usage in bytes',
            ['device', 'mountpoint', 'type'],
            registry=self.registry
        )
        
        self.disk_usage_percent = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            ['device', 'mountpoint'],
            registry=self.registry
        )
        
        self.disk_io = Counter(
            'system_disk_io_bytes_total',
            'Total disk I/O in bytes',
            ['device', 'direction'],
            registry=self.registry
        )
        
        # Network metrics
        self.network_io = Counter(
            'system_network_io_bytes_total',
            'Total network I/O in bytes',
            ['interface', 'direction'],
            registry=self.registry
        )
        
        self.network_connections = Gauge(
            'system_network_connections_total',
            'Total network connections',
            ['state'],
            registry=self.registry
        )

    def _init_application_metrics(self) -> None:
        """Initialize application performance metrics"""
        # Request metrics
        self.http_requests_total = Counter(
            'knowledgehub_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'knowledgehub_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'knowledgehub_database_connections_active',
            'Active database connections',
            ['database'],
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'knowledgehub_database_query_duration_seconds',
            'Database query duration',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        # WebSocket metrics
        self.websocket_connections = Gauge(
            'knowledgehub_websocket_connections_active',
            'Active WebSocket connections',
            registry=self.registry
        )
        
        self.websocket_messages = Counter(
            'knowledgehub_websocket_messages_total',
            'Total WebSocket messages',
            ['direction', 'type'],
            registry=self.registry
        )

    def _init_health_metrics(self) -> None:
        """Initialize health check metrics"""
        self.service_up = Gauge(
            'knowledgehub_service_up',
            'Service health status (1=up, 0=down)',
            ['service', 'type'],
            registry=self.registry
        )
        
        self.service_response_time = Gauge(
            'knowledgehub_service_response_time_seconds',
            'Service response time',
            ['service'],
            registry=self.registry
        )
        
        self.health_checks_total = Counter(
            'knowledgehub_health_checks_total',
            'Total health checks performed',
            ['service', 'status'],
            registry=self.registry
        )
        
        self.uptime_seconds = Gauge(
            'knowledgehub_uptime_seconds',
            'Service uptime in seconds',
            ['service'],
            registry=self.registry
        )

    def _init_business_metrics(self) -> None:
        """Initialize business logic metrics"""
        # Memory system metrics
        self.memory_operations = Counter(
            'knowledgehub_memory_operations_total',
            'Total memory operations',
            ['operation', 'user'],
            registry=self.registry
        )
        
        self.memory_search_duration = Histogram(
            'knowledgehub_memory_search_duration_seconds',
            'Memory search operation duration',
            ['search_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry
        )
        
        # AI processing metrics
        self.ai_processing_duration = Histogram(
            'knowledgehub_ai_processing_duration_seconds',
            'AI processing duration',
            ['operation', 'model'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.ai_operations_total = Counter(
            'knowledgehub_ai_operations_total',
            'Total AI operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        # User metrics
        self.active_users = Gauge(
            'knowledgehub_active_users',
            'Currently active users',
            registry=self.registry
        )
        
        self.user_sessions = Counter(
            'knowledgehub_user_sessions_total',
            'Total user sessions',
            ['action'],
            registry=self.registry
        )

    def _init_custom_metrics(self) -> None:
        """Initialize KnowledgeHub-specific metrics"""
        # Error tracking
        self.errors_total = Counter(
            'knowledgehub_errors_total',
            'Total errors by type',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Performance tracking
        self.performance_score = Gauge(
            'knowledgehub_performance_score',
            'Overall performance score',
            ['component'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations = Counter(
            'knowledgehub_cache_operations_total',
            'Total cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            'knowledgehub_queue_size',
            'Current queue size',
            ['queue'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'knowledgehub_app_info',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': '1.0.0',
            'build': 'production',
            'environment': 'production',
            'python_version': '3.11'
        })

    async def start_collection(self) -> None:
        """Start metrics collection"""
        if not self.enabled:
            logger.warning("Metrics collection not available")
            return
            
        logger.info("Starting Prometheus metrics collection...")
        self.is_collecting = True
        
        # Start collection task
        asyncio.create_task(self._collect_metrics_loop())
        
        # Start HTTP server for metrics endpoint
        try:
            start_http_server(8000, registry=self.registry)
            logger.info("Prometheus metrics server started on port 8000")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    async def stop_collection(self) -> None:
        """Stop metrics collection"""
        logger.info("Stopping metrics collection...")
        self.is_collecting = False

    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format"""
        if not self.enabled:
            return "# Prometheus metrics not available\n"
            
        return generate_latest(self.registry).decode('utf-8')

    # Metric recording methods

    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record HTTP request metrics"""
        if not self.enabled:
            return
            
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_database_query(self, operation: str, duration: float) -> None:
        """Record database query metrics"""
        if not self.enabled:
            return
            
        self.db_query_duration.labels(operation=operation).observe(duration)

    def record_memory_operation(self, operation: str, user_id: str, duration: float = None) -> None:
        """Record memory system operation"""
        if not self.enabled:
            return
            
        self.memory_operations.labels(
            operation=operation,
            user=user_id
        ).inc()
        
        if duration is not None:
            search_type = "semantic" if "search" in operation else "standard"
            self.memory_search_duration.labels(search_type=search_type).observe(duration)

    def record_ai_operation(self, operation: str, model: str, duration: float, success: bool) -> None:
        """Record AI processing operation"""
        if not self.enabled:
            return
            
        self.ai_processing_duration.labels(
            operation=operation,
            model=model
        ).observe(duration)
        
        self.ai_operations_total.labels(
            operation=operation,
            status="success" if success else "error"
        ).inc()

    def record_websocket_message(self, direction: str, message_type: str) -> None:
        """Record WebSocket message"""
        if not self.enabled:
            return
            
        self.websocket_messages.labels(
            direction=direction,
            type=message_type
        ).inc()

    def set_active_websocket_connections(self, count: int) -> None:
        """Set current WebSocket connection count"""
        if not self.enabled:
            return
            
        self.websocket_connections.set(count)

    def set_active_users(self, count: int) -> None:
        """Set current active user count"""
        if not self.enabled:
            return
            
        self.active_users.set(count)

    def record_error(self, error_type: str, component: str) -> None:
        """Record an error"""
        if not self.enabled:
            return
            
        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()

    def set_performance_score(self, component: str, score: float) -> None:
        """Set performance score for component"""
        if not self.enabled:
            return
            
        self.performance_score.labels(component=component).set(score)

    def record_cache_operation(self, operation: str, result: str) -> None:
        """Record cache operation"""
        if not self.enabled:
            return
            
        self.cache_operations.labels(
            operation=operation,
            result=result
        ).inc()

    def set_queue_size(self, queue_name: str, size: int) -> None:
        """Set queue size"""
        if not self.enabled:
            return
            
        self.queue_size.labels(queue=queue_name).set(size)

    # Private methods

    async def _collect_metrics_loop(self) -> None:
        """Main metrics collection loop"""
        while self.is_collecting:
            try:
                await self._collect_system_metrics()
                await self._collect_health_metrics()
                
                self.last_collection = datetime.now(timezone.utc)
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)

    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.labels(core="all").set(cpu_percent)
            
            # Per-core CPU
            cpu_percents = psutil.cpu_percent(percpu=True)
            for i, percent in enumerate(cpu_percents):
                self.cpu_usage.labels(core=str(i)).set(percent)
            
            # Load averages
            load_avg = psutil.getloadavg()
            self.cpu_load.labels(period="1m").set(load_avg[0])
            self.cpu_load.labels(period="5m").set(load_avg[1])
            self.cpu_load.labels(period="15m").set(load_avg[2])
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_usage.labels(type="total").set(memory.total)
            self.memory_usage.labels(type="available").set(memory.available)
            self.memory_usage.labels(type="used").set(memory.used)
            self.memory_usage.labels(type="free").set(memory.free)
            self.memory_usage_percent.set(memory.percent)
            
            # Disk metrics
            disk_partitions = psutil.disk_partitions()
            for partition in disk_partitions:
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    device = partition.device.replace('/', '_')
                    mountpoint = partition.mountpoint.replace('/', '_root')
                    
                    self.disk_usage.labels(
                        device=device,
                        mountpoint=mountpoint,
                        type="total"
                    ).set(disk_usage.total)
                    
                    self.disk_usage.labels(
                        device=device,
                        mountpoint=mountpoint,
                        type="used"
                    ).set(disk_usage.used)
                    
                    self.disk_usage.labels(
                        device=device,
                        mountpoint=mountpoint,
                        type="free"
                    ).set(disk_usage.free)
                    
                    usage_percent = (disk_usage.used / disk_usage.total) * 100
                    self.disk_usage_percent.labels(
                        device=device,
                        mountpoint=mountpoint
                    ).set(usage_percent)
                    
                except (PermissionError, OSError):
                    continue
            
            # Network metrics
            network_counters = psutil.net_io_counters(pernic=True)
            for interface, counters in network_counters.items():
                self.network_io.labels(
                    interface=interface,
                    direction="sent"
                )._value._value = counters.bytes_sent
                
                self.network_io.labels(
                    interface=interface,
                    direction="received"
                )._value._value = counters.bytes_recv
            
            # Network connections
            connections = psutil.net_connections()
            connection_states = defaultdict(int)
            for conn in connections:
                if conn.status:
                    connection_states[conn.status] += 1
            
            for state, count in connection_states.items():
                self.network_connections.labels(state=state).set(count)
                
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")

    async def _collect_health_metrics(self) -> None:
        """Collect health check metrics"""
        try:
            system_health = await real_health_monitor.get_system_health()
            
            # Service health metrics
            for service_name, service_health in system_health.services.items():
                # Service up/down status
                status_value = 1 if service_health.status == HealthStatus.HEALTHY else 0
                self.service_up.labels(
                    service=service_name,
                    type=service_health.service_type.value
                ).set(status_value)
                
                # Response time
                response_time_seconds = service_health.response_time_ms / 1000.0
                self.service_response_time.labels(service=service_name).set(response_time_seconds)
                
                # Health check counter
                self.health_checks_total.labels(
                    service=service_name,
                    status=service_health.status.value
                ).inc()
                
        except Exception as e:
            logger.error(f"Health metrics collection failed: {e}")

# Global metrics instance
prometheus_metrics = PrometheusMetrics()