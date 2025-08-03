"""
Real Production Health Monitoring System

This service provides comprehensive health monitoring for all KnowledgeHub services
with real-time alerting, performance tracking, and automated recovery capabilities.
"""

import asyncio
import logging
import time
import psutil
import socket
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, NamedTuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import aiohttp
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select

from shared.config import Config
from ..database import get_db_session
from ..models.health_check import HealthCheck, ServiceStatus
from .real_websocket_events import RealWebSocketEvents, EventType

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    DOWN = "down"
    UNKNOWN = "unknown"

class ServiceType(str, Enum):
    """Service types for monitoring"""
    DATABASE = "database"
    REDIS = "redis"
    WEAVIATE = "weaviate"
    NEO4J = "neo4j"
    TIMESCALE = "timescale"
    MINIO = "minio"
    API = "api"
    WEBSOCKET = "websocket"
    AI_SERVICE = "ai_service"
    MCP_SERVER = "mcp_server"
    EXTERNAL_API = "external_api"

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: Any
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass  
class ServiceHealth:
    """Health status for a service"""
    service_name: str
    service_type: ServiceType
    status: HealthStatus
    response_time_ms: float
    metrics: List[HealthMetric] = field(default_factory=list)
    error_message: Optional[str] = None
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_percentage: float = 100.0
    consecutive_failures: int = 0

@dataclass
class SystemHealth:
    """Overall system health status"""
    overall_status: HealthStatus
    services: Dict[str, ServiceHealth]
    system_metrics: List[HealthMetric]
    alerts: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class RealHealthMonitor:
    """
    Real production-grade health monitoring system with:
    - Comprehensive service health checks
    - Performance metrics collection
    - Real-time alerting
    - Automated recovery triggers
    - Distributed health coordination
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.websocket_events = RealWebSocketEvents(config)
        
        # Monitoring configuration
        self.check_interval = 30  # seconds
        self.timeout_seconds = 10
        self.failure_threshold = 3
        self.recovery_threshold = 2
        
        # State tracking
        self.is_monitoring = False
        self.health_history: Dict[str, List[ServiceHealth]] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.recovery_callbacks: Dict[str, List[Callable]] = {}
        
        # Performance metrics
        self.metrics_buffer: Dict[str, List[HealthMetric]] = {}
        self.alert_handlers: List[Callable] = []
        
        logger.info("Real Health Monitor initialized")

    async def start_monitoring(self) -> None:
        """Start the health monitoring system"""
        logger.info("Starting comprehensive health monitoring...")
        
        try:
            self.is_monitoring = True
            
            # Start monitoring tasks
            monitoring_tasks = [
                asyncio.create_task(self._monitor_services()),
                asyncio.create_task(self._monitor_system_resources()),
                asyncio.create_task(self._process_alerts()),
                asyncio.create_task(self._cleanup_old_data()),
                asyncio.create_task(self._generate_health_reports())
            ]
            
            logger.info("Health monitoring system started successfully")
            
            # Wait for all monitoring tasks
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Health monitoring failed to start: {e}")
            self.is_monitoring = False
            raise

    async def stop_monitoring(self) -> None:
        """Stop the health monitoring system"""
        logger.info("Stopping health monitoring...")
        self.is_monitoring = False

    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        try:
            # Check all services
            services = {}
            for service_name, service_config in self._get_service_configs().items():
                health = await self._check_service_health(service_name, service_config)
                services[service_name] = health
            
            # Get system metrics
            system_metrics = await self._get_system_metrics()
            
            # Determine overall status
            overall_status = self._calculate_overall_status(services)
            
            # Get active alerts
            alerts = list(self.active_alerts.values())
            
            return SystemHealth(
                overall_status=overall_status,
                services=services,
                system_metrics=system_metrics,
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                services={},
                system_metrics=[],
                alerts=[{"error": str(e)}]
            )

    async def check_service_health(self, service_name: str) -> ServiceHealth:
        """Check health of a specific service"""
        service_config = self._get_service_configs().get(service_name)
        if not service_config:
            return ServiceHealth(
                service_name=service_name,
                service_type=ServiceType.EXTERNAL_API,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0,
                error_message=f"Unknown service: {service_name}"
            )
        
        return await self._check_service_health(service_name, service_config)

    async def add_recovery_callback(self, service_name: str, callback: Callable) -> None:
        """Add a recovery callback for a service"""
        if service_name not in self.recovery_callbacks:
            self.recovery_callbacks[service_name] = []
        self.recovery_callbacks[service_name].append(callback)

    async def trigger_manual_check(self, service_name: Optional[str] = None) -> SystemHealth:
        """Trigger a manual health check"""
        logger.info(f"Manual health check triggered for: {service_name or 'all services'}")
        
        if service_name:
            # Check specific service
            service_config = self._get_service_configs().get(service_name)
            if service_config:
                health = await self._check_service_health(service_name, service_config)
                return SystemHealth(
                    overall_status=health.status,
                    services={service_name: health},
                    system_metrics=[],
                    alerts=[]
                )
        
        # Check all services
        return await self.get_system_health()

    async def get_health_metrics(self, service_name: str, time_window: int = 3600) -> List[HealthMetric]:
        """Get historical health metrics for a service"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=time_window)
        
        if service_name not in self.health_history:
            return []
        
        # Filter metrics by time window
        metrics = []
        for health in self.health_history[service_name]:
            if start_time <= health.last_check <= end_time:
                metrics.extend(health.metrics)
        
        return sorted(metrics, key=lambda m: m.timestamp)

    async def get_uptime_statistics(self, service_name: str, days: int = 7) -> Dict[str, Any]:
        """Get uptime statistics for a service"""
        if service_name not in self.health_history:
            return {"error": f"No history for service {service_name}"}
        
        history = self.health_history[service_name]
        if not history:
            return {"error": "No health history available"}
        
        # Calculate uptime stats
        total_checks = len(history)
        healthy_checks = sum(1 for h in history if h.status == HealthStatus.HEALTHY)
        
        uptime_percentage = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Calculate MTTR (Mean Time To Recovery)
        failure_periods = []
        current_failure_start = None
        
        for health in sorted(history, key=lambda h: h.last_check):
            if health.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
                if current_failure_start is None:
                    current_failure_start = health.last_check
            else:
                if current_failure_start is not None:
                    failure_periods.append(
                        (health.last_check - current_failure_start).total_seconds()
                    )
                    current_failure_start = None
        
        mttr_seconds = sum(failure_periods) / len(failure_periods) if failure_periods else 0
        
        return {
            "service_name": service_name,
            "uptime_percentage": round(uptime_percentage, 2),
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "failed_checks": total_checks - healthy_checks,
            "mttr_minutes": round(mttr_seconds / 60, 2),
            "failure_count": len(failure_periods),
            "period_days": days
        }

    # Private methods

    async def _monitor_services(self) -> None:
        """Continuously monitor all services"""
        while self.is_monitoring:
            try:
                services = self._get_service_configs()
                
                # Check all services in parallel
                tasks = []
                for service_name, service_config in services.items():
                    task = self._check_service_health(service_name, service_config)
                    tasks.append(task)
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for i, (service_name, _) in enumerate(services.items()):
                        if isinstance(results[i], ServiceHealth):
                            await self._process_service_health(service_name, results[i])
                        else:
                            logger.error(f"Health check failed for {service_name}: {results[i]}")
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(5)  # Short delay on error

    async def _monitor_system_resources(self) -> None:
        """Monitor system-level resources"""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 80:
                    await self._create_alert(
                        "system_cpu",
                        f"High CPU usage: {cpu_percent}%",
                        "critical" if cpu_percent > 90 else "warning"
                    )
                
                # Memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    await self._create_alert(
                        "system_memory",
                        f"High memory usage: {memory.percent}%",
                        "critical" if memory.percent > 90 else "warning"
                    )
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                if disk_percent > 80:
                    await self._create_alert(
                        "system_disk",
                        f"High disk usage: {disk_percent:.1f}%",
                        "critical" if disk_percent > 90 else "warning"
                    )
                
                # Network connections
                connections = len(psutil.net_connections())
                if connections > 1000:
                    await self._create_alert(
                        "system_network",
                        f"High connection count: {connections}",
                        "warning"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"System resource monitoring error: {e}")
                await asyncio.sleep(30)

    async def _check_service_health(self, service_name: str, config: Dict[str, Any]) -> ServiceHealth:
        """Check health of a specific service"""
        start_time = time.time()
        service_type = ServiceType(config.get('type', 'external_api'))
        
        try:
            if service_type == ServiceType.DATABASE:
                health = await self._check_database_health(config)
            elif service_type == ServiceType.REDIS:
                health = await self._check_redis_health(config)
            elif service_type == ServiceType.WEAVIATE:
                health = await self._check_weaviate_health(config)
            elif service_type == ServiceType.NEO4J:
                health = await self._check_neo4j_health(config)
            elif service_type == ServiceType.MINIO:
                health = await self._check_minio_health(config)
            elif service_type == ServiceType.API:
                health = await self._check_api_health(config)
            else:
                health = await self._check_generic_service_health(config)
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                service_name=service_name,
                service_type=service_type,
                status=health['status'],
                response_time_ms=response_time,
                metrics=health.get('metrics', []),
                error_message=health.get('error')
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check failed for {service_name}: {e}")
            
            return ServiceHealth(
                service_name=service_name,
                service_type=service_type,
                status=HealthStatus.DOWN,
                response_time_ms=response_time,
                error_message=str(e)
            )

    async def _check_database_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check PostgreSQL database health"""
        try:
            async with get_db_session() as session:
                # Basic connectivity
                await session.execute(text("SELECT 1"))
                
                # Get database stats
                result = await session.execute(text("""
                    SELECT 
                        (SELECT count(*) FROM pg_stat_activity) as active_connections,
                        (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections,
                        (SELECT round(100.0 * count(*) / setting::int, 2) FROM pg_stat_activity, pg_settings WHERE pg_settings.name = 'max_connections') as connection_usage
                """))
                row = result.fetchone()
                
                metrics = [
                    HealthMetric("active_connections", row[0], "count", 50, 80),
                    HealthMetric("max_connections", row[1], "count"),
                    HealthMetric("connection_usage", row[2], "percent", 70, 90)
                ]
                
                status = HealthStatus.HEALTHY
                if row[2] > 90:
                    status = HealthStatus.CRITICAL
                elif row[2] > 70:
                    status = HealthStatus.DEGRADED
                
                return {"status": status, "metrics": metrics}
                
        except Exception as e:
            return {"status": HealthStatus.DOWN, "error": str(e)}

    async def _check_redis_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            redis_url = config.get('url', 'redis://redis:6379')
            redis = aioredis.from_url(redis_url)
            
            # Basic connectivity
            await redis.ping()
            
            # Get Redis info
            info = await redis.info()
            
            metrics = [
                HealthMetric("connected_clients", info.get('connected_clients', 0), "count", 100, 200),
                HealthMetric("used_memory", info.get('used_memory', 0), "bytes"),
                HealthMetric("keyspace_hits", info.get('keyspace_hits', 0), "count"),
                HealthMetric("keyspace_misses", info.get('keyspace_misses', 0), "count")
            ]
            
            # Calculate hit ratio
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            hit_ratio = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 100
            metrics.append(HealthMetric("hit_ratio", hit_ratio, "percent", 80, 60))
            
            await redis.close()
            
            status = HealthStatus.HEALTHY
            if hit_ratio < 60:
                status = HealthStatus.DEGRADED
            
            return {"status": status, "metrics": metrics}
            
        except Exception as e:
            return {"status": HealthStatus.DOWN, "error": str(e)}

    async def _check_weaviate_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check Weaviate health"""
        try:
            url = config.get('url', 'http://localhost:8080')
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                # Check readiness
                async with session.get(f"{url}/v1/meta") as response:
                    if response.status == 200:
                        meta = await response.json()
                        
                        metrics = [
                            HealthMetric("hostname", meta.get('hostname', 'unknown'), "string"),
                            HealthMetric("version", meta.get('version', 'unknown'), "string")
                        ]
                        
                        return {"status": HealthStatus.HEALTHY, "metrics": metrics}
                    else:
                        return {"status": HealthStatus.DOWN, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"status": HealthStatus.DOWN, "error": str(e)}

    async def _check_neo4j_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check Neo4j health"""
        try:
            # This would require neo4j driver
            # For now, just check if port is open
            host = config.get('host', 'localhost')
            port = config.get('port', 7687)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return {"status": HealthStatus.HEALTHY}
            else:
                return {"status": HealthStatus.DOWN, "error": "Connection refused"}
                
        except Exception as e:
            return {"status": HealthStatus.DOWN, "error": str(e)}

    async def _check_minio_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check MinIO health"""
        try:
            url = config.get('url', 'http://localhost:9000')
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                async with session.get(f"{url}/minio/health/live") as response:
                    if response.status == 200:
                        return {"status": HealthStatus.HEALTHY}
                    else:
                        return {"status": HealthStatus.DOWN, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"status": HealthStatus.DOWN, "error": str(e)}

    async def _check_api_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check API endpoint health"""
        try:
            url = config.get('url')
            health_endpoint = config.get('health_endpoint', '/health')
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                async with session.get(f"{url}{health_endpoint}") as response:
                    if response.status == 200:
                        data = await response.json()
                        status = HealthStatus(data.get('status', 'healthy'))
                        return {"status": status}
                    else:
                        return {"status": HealthStatus.DOWN, "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"status": HealthStatus.DOWN, "error": str(e)}

    async def _check_generic_service_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check generic service health via port"""
        try:
            host = config.get('host', 'localhost')
            port = config.get('port')
            
            if not port:
                return {"status": HealthStatus.UNKNOWN, "error": "No port specified"}
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return {"status": HealthStatus.HEALTHY}
            else:
                return {"status": HealthStatus.DOWN, "error": "Connection refused"}
                
        except Exception as e:
            return {"status": HealthStatus.DOWN, "error": str(e)}

    async def _get_system_metrics(self) -> List[HealthMetric]:
        """Get system-level metrics"""
        metrics = []
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            metrics.append(HealthMetric("cpu_usage", cpu_percent, "percent", 80, 90))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(HealthMetric("memory_usage", memory.percent, "percent", 80, 90))
            metrics.append(HealthMetric("memory_available", memory.available, "bytes"))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(HealthMetric("disk_usage", disk_percent, "percent", 80, 90))
            metrics.append(HealthMetric("disk_free", disk.free, "bytes"))
            
            # Network metrics
            net_io = psutil.net_io_counters()
            metrics.append(HealthMetric("network_bytes_sent", net_io.bytes_sent, "bytes"))
            metrics.append(HealthMetric("network_bytes_recv", net_io.bytes_recv, "bytes"))
            
            # Process metrics
            process = psutil.Process()
            metrics.append(HealthMetric("process_memory", process.memory_info().rss, "bytes"))
            metrics.append(HealthMetric("process_cpu", process.cpu_percent(), "percent"))
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
        
        return metrics

    def _get_service_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get service configurations for health checking"""
        return {
            "postgresql": {
                "type": "database",
                "host": "localhost",
                "port": 5432
            },
            "redis": {
                "type": "redis", 
                "url": "redis://redis:6379"
            },
            "weaviate": {
                "type": "weaviate",
                "url": "http://localhost:8080"
            },
            "neo4j": {
                "type": "neo4j",
                "host": "localhost",
                "port": 7687
            },
            "minio": {
                "type": "minio",
                "url": "http://localhost:9000"
            },
            "knowledgehub_api": {
                "type": "api",
                "url": "http://localhost:3000",
                "health_endpoint": "/health"
            }
        }

    def _calculate_overall_status(self, services: Dict[str, ServiceHealth]) -> HealthStatus:
        """Calculate overall system status from service statuses"""
        if not services:
            return HealthStatus.UNKNOWN
        
        statuses = [service.status for service in services.values()]
        
        if any(status == HealthStatus.DOWN for status in statuses):
            return HealthStatus.CRITICAL
        elif any(status == HealthStatus.CRITICAL for status in statuses):
            return HealthStatus.CRITICAL  
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.DEGRADED

    async def _process_service_health(self, service_name: str, health: ServiceHealth) -> None:
        """Process service health result and trigger alerts/recovery"""
        # Store in history
        if service_name not in self.health_history:
            self.health_history[service_name] = []
        
        self.health_history[service_name].append(health)
        
        # Limit history size
        if len(self.health_history[service_name]) > 1000:
            self.health_history[service_name] = self.health_history[service_name][-500:]
        
        # Check for alerts
        if health.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
            await self._create_alert(
                f"service_{service_name}",
                f"Service {service_name} is {health.status.value}: {health.error_message}",
                "critical"
            )
            
            # Trigger recovery if available
            if service_name in self.recovery_callbacks:
                for callback in self.recovery_callbacks[service_name]:
                    try:
                        await callback(health)
                    except Exception as e:
                        logger.error(f"Recovery callback failed for {service_name}: {e}")
        
        elif health.status == HealthStatus.DEGRADED:
            await self._create_alert(
                f"service_{service_name}",
                f"Service {service_name} is degraded: {health.error_message}",
                "warning"
            )
        
        else:
            # Clear alerts if service is healthy
            alert_key = f"service_{service_name}"
            if alert_key in self.active_alerts:
                del self.active_alerts[alert_key]

    async def _create_alert(self, alert_id: str, message: str, severity: str) -> None:
        """Create or update an alert"""
        alert = {
            "id": alert_id,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": 1
        }
        
        if alert_id in self.active_alerts:
            # Update existing alert
            alert["count"] = self.active_alerts[alert_id]["count"] + 1
        else:
            # Send WebSocket event for new alert
            await self.websocket_events.send_event(
                EventType.SYSTEM_ALERT,
                "system",
                {
                    "alert_id": alert_id,
                    "message": message,
                    "severity": severity
                }
            )
        
        self.active_alerts[alert_id] = alert
        logger.warning(f"Alert {severity}: {message}")

    async def _process_alerts(self) -> None:
        """Process and manage alerts"""
        while self.is_monitoring:
            try:
                # Process alert handlers
                for handler in self.alert_handlers:
                    try:
                        await handler(list(self.active_alerts.values()))
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}")
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30)

    async def _cleanup_old_data(self) -> None:
        """Clean up old health data"""
        while self.is_monitoring:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
                
                # Clean up health history
                for service_name in self.health_history:
                    self.health_history[service_name] = [
                        h for h in self.health_history[service_name]
                        if h.last_check > cutoff_time
                    ]
                
                # Clean up old alerts (older than 24 hours)
                alert_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                old_alerts = []
                
                for alert_id, alert in self.active_alerts.items():
                    alert_time = datetime.fromisoformat(alert["timestamp"].replace('Z', '+00:00'))
                    if alert_time < alert_cutoff:
                        old_alerts.append(alert_id)
                
                for alert_id in old_alerts:
                    del self.active_alerts[alert_id]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(1800)

    async def _generate_health_reports(self) -> None:
        """Generate periodic health reports"""
        while self.is_monitoring:
            try:
                # Generate daily health report
                system_health = await self.get_system_health()
                
                # Send report via WebSocket
                await self.websocket_events.send_event(
                    EventType.HEALTH_REPORT,
                    "system",
                    {
                        "overall_status": system_health.overall_status.value,
                        "service_count": len(system_health.services),
                        "healthy_services": sum(1 for s in system_health.services.values() if s.status == HealthStatus.HEALTHY),
                        "alert_count": len(system_health.alerts),
                        "timestamp": system_health.timestamp.isoformat()
                    }
                )
                
                await asyncio.sleep(86400)  # Daily reports
                
            except Exception as e:
                logger.error(f"Health report generation error: {e}")
                await asyncio.sleep(3600)

# Global health monitor instance
real_health_monitor = RealHealthMonitor()