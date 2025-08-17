
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from typing import Dict, Any
import psutil
import asyncio

class MonitoringSystem:
    """Comprehensive monitoring and metrics collection"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database'],
            registry=self.registry
        )
        
        self.query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['query_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Start background metrics collection
        asyncio.create_task(self._collect_system_metrics())
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # CPU and memory metrics
                self.cpu_usage.set(psutil.cpu_percent())
                self.memory_usage.set(psutil.virtual_memory().percent)
                
                # Database connection metrics (example)
                # self.db_connections.labels(database="postgres").set(count)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
            
            await asyncio.sleep(10)  # Collect every 10 seconds
    
    def track_request(self, method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request metrics"""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def track_cache(self, cache_type: str, hit: bool):
        """Track cache metrics"""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict()
        }

monitoring = MonitoringSystem()
