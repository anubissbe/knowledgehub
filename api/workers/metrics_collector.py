"""
Metrics Collector Worker for Automated System Monitoring.

This worker provides:
- Automated system metrics collection
- Application performance monitoring
- Database performance tracking
- Custom business metrics
- Real-time data gathering
"""

import logging
import asyncio
import psutil
import time
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, text

from ..models.base import get_db_context
from ..models.memory import MemoryItem
from ..models.session import Session
from ..models.error_pattern import ErrorOccurrence
from ..models.workflow import WorkflowExecution
from ..services.metrics_service import metrics_service, MetricType
from ..services.cache import redis_client
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("metrics_collector")


@dataclass
class CollectionConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    interval_seconds: int = 30
    include_system: bool = True
    include_database: bool = True
    include_application: bool = True
    include_business: bool = True
    custom_metrics: List[str] = None


class SystemMetricsCollector:
    """Collects system-level metrics using psutil."""
    
    def __init__(self):
        self.process = psutil.Process()
        
    async def collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU metrics."""
        try:
            # System CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Process CPU
            process_cpu = self.process.cpu_percent()
            
            # Load average (Linux/Mac)
            try:
                load_avg = psutil.getloadavg()
                load_1m, load_5m, load_15m = load_avg
            except AttributeError:
                # Windows doesn't have load average
                load_1m = load_5m = load_15m = 0.0
            
            return {
                "cpu_usage_percent": cpu_percent,
                "cpu_count": cpu_count,
                "process_cpu_percent": process_cpu,
                "load_avg_1m": load_1m,
                "load_avg_5m": load_5m,
                "load_avg_15m": load_15m
            }
            
        except Exception as e:
            logger.error(f"Failed to collect CPU metrics: {e}")
            return {}
    
    async def collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory metrics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process memory
            process_memory = self.process.memory_info()
            process_memory_percent = self.process.memory_percent()
            
            return {
                "memory_total_bytes": memory.total,
                "memory_available_bytes": memory.available,
                "memory_used_bytes": memory.used,
                "memory_usage_percent": memory.percent,
                "swap_total_bytes": swap.total,
                "swap_used_bytes": swap.used,
                "swap_usage_percent": swap.percent,
                "process_memory_rss": process_memory.rss,
                "process_memory_vms": process_memory.vms,
                "process_memory_percent": process_memory_percent
            }
            
        except Exception as e:
            logger.error(f"Failed to collect memory metrics: {e}")
            return {}
    
    async def collect_disk_metrics(self) -> Dict[str, float]:
        """Collect disk metrics."""
        try:
            # Disk usage for root partition
            disk_usage = psutil.disk_usage('/')
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            metrics = {
                "disk_total_bytes": disk_usage.total,
                "disk_used_bytes": disk_usage.used,
                "disk_free_bytes": disk_usage.free,
                "disk_usage_percent": (disk_usage.used / disk_usage.total) * 100
            }
            
            if disk_io:
                metrics.update({
                    "disk_read_bytes": disk_io.read_bytes,
                    "disk_write_bytes": disk_io.write_bytes,
                    "disk_read_count": disk_io.read_count,
                    "disk_write_count": disk_io.write_count
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect disk metrics: {e}")
            return {}
    
    async def collect_network_metrics(self) -> Dict[str, float]:
        """Collect network metrics."""
        try:
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Network connections
            connections = psutil.net_connections()
            
            metrics = {
                "network_bytes_sent": network_io.bytes_sent,
                "network_bytes_recv": network_io.bytes_recv,
                "network_packets_sent": network_io.packets_sent,
                "network_packets_recv": network_io.packets_recv,
                "network_connections_total": len(connections)
            }
            
            # Count connections by status
            connection_states = {}
            for conn in connections:
                state = conn.status
                connection_states[state] = connection_states.get(state, 0) + 1
            
            for state, count in connection_states.items():
                metrics[f"network_connections_{state.lower()}"] = count
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect network metrics: {e}")
            return {}


class DatabaseMetricsCollector:
    """Collects database performance metrics."""
    
    async def collect_connection_metrics(self) -> Dict[str, float]:
        """Collect database connection metrics."""
        try:
            with get_db_context() as db:
                # Connection pool stats (PostgreSQL specific)
                result = db.execute(text("""
                    SELECT 
                        state,
                        COUNT(*) as count
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                    GROUP BY state
                """)).fetchall()
                
                metrics = {}
                total_connections = 0
                
                for row in result:
                    state = row[0] or "unknown"
                    count = row[1]
                    metrics[f"db_connections_{state}"] = count
                    total_connections += count
                
                metrics["db_connections_total"] = total_connections
                
                # Max connections
                max_conn_result = db.execute(text(
                    "SHOW max_connections"
                )).fetchone()
                
                if max_conn_result:
                    max_connections = int(max_conn_result[0])
                    metrics["db_connections_max"] = max_connections
                    metrics["db_connections_usage_percent"] = (
                        total_connections / max_connections * 100
                    )
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect database connection metrics: {e}")
            return {}
    
    async def collect_query_metrics(self) -> Dict[str, float]:
        """Collect database query performance metrics."""
        try:
            with get_db_context() as db:
                # Query statistics
                result = db.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        n_tup_ins,
                        n_tup_upd,
                        n_tup_del
                    FROM pg_stat_user_tables
                """)).fetchall()
                
                metrics = {
                    "db_sequential_scans": 0,
                    "db_index_scans": 0,
                    "db_rows_inserted": 0,
                    "db_rows_updated": 0,
                    "db_rows_deleted": 0
                }
                
                for row in result:
                    metrics["db_sequential_scans"] += row[2] or 0
                    metrics["db_index_scans"] += row[4] or 0
                    metrics["db_rows_inserted"] += row[6] or 0
                    metrics["db_rows_updated"] += row[7] or 0
                    metrics["db_rows_deleted"] += row[8] or 0
                
                # Active queries
                active_queries = db.execute(text("""
                    SELECT COUNT(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND datname = current_database()
                """)).scalar()
                
                metrics["db_active_queries"] = active_queries
                
                # Long-running queries
                long_queries = db.execute(text("""
                    SELECT COUNT(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND datname = current_database()
                    AND now() - query_start > interval '1 minute'
                """)).scalar()
                
                metrics["db_long_running_queries"] = long_queries
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect database query metrics: {e}")
            return {}
    
    async def collect_size_metrics(self) -> Dict[str, float]:
        """Collect database size metrics."""
        try:
            with get_db_context() as db:
                # Database size
                db_size = db.execute(text(
                    "SELECT pg_database_size(current_database())"
                )).scalar()
                
                # Table sizes
                table_sizes = db.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_total_relation_size(schemaname||'.'||tablename) as size
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                """)).fetchall()
                
                metrics = {
                    "db_size_bytes": db_size
                }
                
                total_table_size = 0
                for row in table_sizes:
                    table_name = row[1]
                    size = row[2]
                    metrics[f"db_table_size_{table_name}"] = size
                    total_table_size += size
                
                metrics["db_tables_total_size"] = total_table_size
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect database size metrics: {e}")
            return {}


class ApplicationMetricsCollector:
    """Collects application-specific metrics."""
    
    async def collect_memory_system_metrics(self) -> Dict[str, float]:
        """Collect memory system metrics."""
        try:
            with get_db_context() as db:
                # Total memories
                total_memories = db.query(MemoryItem).count()
                
                # Memories by type
                memory_types = db.query(
                    MemoryItem.memory_type,
                    func.count(MemoryItem.id)
                ).group_by(MemoryItem.memory_type).all()
                
                # Recent memories (last 24 hours)
                recent_memories = db.query(MemoryItem).filter(
                    MemoryItem.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                # Memory content size
                total_content_size = db.query(
                    func.sum(func.length(MemoryItem.content))
                ).scalar() or 0
                
                metrics = {
                    "memory_total_count": total_memories,
                    "memory_recent_24h": recent_memories,
                    "memory_content_size_bytes": total_content_size
                }
                
                # Add metrics by type
                for memory_type, count in memory_types:
                    metrics[f"memory_type_{memory_type}_count"] = count
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect memory system metrics: {e}")
            return {}
    
    async def collect_session_metrics(self) -> Dict[str, float]:
        """Collect session metrics."""
        try:
            with get_db_context() as db:
                # Active sessions
                active_sessions = db.query(Session).filter(
                    Session.is_active == True
                ).count()
                
                # Sessions by type
                session_types = db.query(
                    Session.session_type,
                    func.count(Session.id)
                ).group_by(Session.session_type).all()
                
                # Recent sessions (last 24 hours)
                recent_sessions = db.query(Session).filter(
                    Session.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                # Average session duration
                avg_duration = db.query(
                    func.avg(
                        func.extract('epoch', Session.ended_at - Session.started_at)
                    )
                ).filter(
                    Session.ended_at.isnot(None)
                ).scalar() or 0
                
                metrics = {
                    "sessions_active": active_sessions,
                    "sessions_recent_24h": recent_sessions,
                    "sessions_avg_duration_seconds": avg_duration
                }
                
                # Add metrics by type
                for session_type, count in session_types:
                    metrics[f"sessions_type_{session_type}_count"] = count
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect session metrics: {e}")
            return {}
    
    async def collect_error_metrics(self) -> Dict[str, float]:
        """Collect error tracking metrics."""
        try:
            with get_db_context() as db:
                # Total errors
                total_errors = db.query(ErrorOccurrence).count()
                
                # Recent errors (last 24 hours)
                recent_errors = db.query(ErrorOccurrence).filter(
                    ErrorOccurrence.occurred_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                # Errors by severity
                error_severities = db.query(
                    ErrorOccurrence.severity,
                    func.count(ErrorOccurrence.id)
                ).group_by(ErrorOccurrence.severity).all()
                
                # Error rate (errors per hour)
                error_rate = recent_errors / 24.0
                
                metrics = {
                    "errors_total_count": total_errors,
                    "errors_recent_24h": recent_errors,
                    "errors_rate_per_hour": error_rate
                }
                
                # Add metrics by severity
                for severity, count in error_severities:
                    metrics[f"errors_severity_{severity}_count"] = count
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect error metrics: {e}")
            return {}
    
    async def collect_workflow_metrics(self) -> Dict[str, float]:
        """Collect workflow execution metrics."""
        try:
            with get_db_context() as db:
                # Total executions
                total_executions = db.query(WorkflowExecution).count()
                
                # Recent executions (last 24 hours)
                recent_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                # Success rate
                successful_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.success == True,
                    WorkflowExecution.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                success_rate = (
                    successful_executions / recent_executions * 100
                    if recent_executions > 0 else 0
                )
                
                # Average execution time
                avg_execution_time = db.query(
                    func.avg(WorkflowExecution.execution_time)
                ).filter(
                    WorkflowExecution.execution_time.isnot(None)
                ).scalar() or 0
                
                # Failed executions
                failed_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.success == False,
                    WorkflowExecution.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                metrics = {
                    "workflows_total_executions": total_executions,
                    "workflows_recent_24h": recent_executions,
                    "workflows_success_rate_percent": success_rate,
                    "workflows_avg_execution_time_ms": avg_execution_time,
                    "workflows_failed_24h": failed_executions
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect workflow metrics: {e}")
            return {}


class BusinessMetricsCollector:
    """Collects business-specific metrics."""
    
    async def collect_user_activity_metrics(self) -> Dict[str, float]:
        """Collect user activity metrics."""
        try:
            with get_db_context() as db:
                # Active users (users with activity in last 24h)
                active_users = db.query(
                    func.count(func.distinct(MemoryItem.user_id))
                ).filter(
                    MemoryItem.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).scalar() or 0
                
                # Total users
                total_users = db.query(
                    func.count(func.distinct(MemoryItem.user_id))
                ).scalar() or 0
                
                # User engagement rate
                engagement_rate = (
                    active_users / total_users * 100
                    if total_users > 0 else 0
                )
                
                # Memory creation rate
                memory_creation_rate = db.query(MemoryItem).filter(
                    MemoryItem.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count() / 24.0
                
                metrics = {
                    "users_active_24h": active_users,
                    "users_total": total_users,
                    "users_engagement_rate_percent": engagement_rate,
                    "memories_creation_rate_per_hour": memory_creation_rate
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect user activity metrics: {e}")
            return {}
    
    async def collect_api_metrics(self) -> Dict[str, float]:
        """Collect API usage metrics."""
        try:
            # Get API metrics from Redis cache
            api_calls_24h = await redis_client.get("api_calls_24h") or 0
            api_errors_24h = await redis_client.get("api_errors_24h") or 0
            avg_response_time = await redis_client.get("avg_response_time") or 0
            
            # Convert string values to float
            api_calls_24h = float(api_calls_24h)
            api_errors_24h = float(api_errors_24h)
            avg_response_time = float(avg_response_time)
            
            # Calculate error rate
            error_rate = (
                api_errors_24h / api_calls_24h * 100
                if api_calls_24h > 0 else 0
            )
            
            # API calls per hour
            api_rate = api_calls_24h / 24.0
            
            metrics = {
                "api_calls_24h": api_calls_24h,
                "api_errors_24h": api_errors_24h,
                "api_error_rate_percent": error_rate,
                "api_calls_per_hour": api_rate,
                "api_avg_response_time_ms": avg_response_time
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect API metrics: {e}")
            return {}


class MetricsCollectorWorker:
    """
    Main metrics collector worker that orchestrates all metric collection.
    
    Features:
    - Automated metric collection
    - Configurable collection intervals
    - Error handling and retry logic
    - Performance monitoring
    - Extensible collector system
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize collectors
        self.system_collector = SystemMetricsCollector()
        self.database_collector = DatabaseMetricsCollector()
        self.application_collector = ApplicationMetricsCollector()
        self.business_collector = BusinessMetricsCollector()
        
        # Worker state
        self._running = False
        self._collection_config = CollectionConfig()
        self._last_collection = {}
        self._collection_errors = {}
        
        # Performance tracking
        self._performance_stats = {
            "collections_completed": 0,
            "collections_failed": 0,
            "avg_collection_time": 0.0,
            "last_collection_time": None
        }
        
        logger.info("Initialized MetricsCollectorWorker")
    
    async def start(self):
        """Start the metrics collector worker."""
        if self._running:
            logger.warning("Metrics collector already running")
            return
        
        try:
            # Initialize services
            await metrics_service.initialize()
            await redis_client.initialize()
            
            self._running = True
            logger.info("Starting metrics collector worker")
            
            # Start collection loop
            asyncio.create_task(self._collection_loop())
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            logger.info("Metrics collector worker started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start metrics collector: {e}")
            self._running = False
            raise
    
    async def stop(self):
        """Stop the metrics collector worker."""
        logger.info("Stopping metrics collector worker")
        self._running = False
        
        # Wait for current collection to complete
        await asyncio.sleep(5)
        
        logger.info("Metrics collector worker stopped")
    
    async def collect_all_metrics(self):
        """Collect all metrics once."""
        start_time = time.time()
        
        try:
            logger.debug("Starting metrics collection cycle")
            
            # Collect different metric categories
            if self._collection_config.include_system:
                await self._collect_system_metrics()
            
            if self._collection_config.include_database:
                await self._collect_database_metrics()
            
            if self._collection_config.include_application:
                await self._collect_application_metrics()
            
            if self._collection_config.include_business:
                await self._collect_business_metrics()
            
            # Record collection performance
            collection_time = (time.time() - start_time) * 1000
            await metrics_service.record_timer(
                "metrics_collection_time",
                collection_time,
                tags={"type": "full_collection"}
            )
            
            # Update performance stats
            self._performance_stats["collections_completed"] += 1
            self._performance_stats["avg_collection_time"] = (
                (self._performance_stats["avg_collection_time"] * 0.9) +
                (collection_time * 0.1)
            )
            self._performance_stats["last_collection_time"] = datetime.utcnow()
            
            logger.debug(f"Completed metrics collection cycle in {collection_time:.2f}ms")
            
        except Exception as e:
            self._performance_stats["collections_failed"] += 1
            logger.error(f"Failed to collect metrics: {e}")
            logger.error(traceback.format_exc())
            
            # Record collection failure
            await metrics_service.record_counter(
                "metrics_collection_errors",
                tags={"error_type": type(e).__name__}
            )
    
    async def get_worker_status(self) -> Dict[str, Any]:
        """Get worker status and statistics."""
        return {
            "running": self._running,
            "config": {
                "interval_seconds": self._collection_config.interval_seconds,
                "include_system": self._collection_config.include_system,
                "include_database": self._collection_config.include_database,
                "include_application": self._collection_config.include_application,
                "include_business": self._collection_config.include_business
            },
            "performance": self._performance_stats,
            "last_collection_times": self._last_collection,
            "collection_errors": self._collection_errors
        }
    
    async def update_collection_config(self, config: Dict[str, Any]):
        """Update collection configuration."""
        try:
            if "interval_seconds" in config:
                self._collection_config.interval_seconds = config["interval_seconds"]
            
            if "include_system" in config:
                self._collection_config.include_system = config["include_system"]
            
            if "include_database" in config:
                self._collection_config.include_database = config["include_database"]
            
            if "include_application" in config:
                self._collection_config.include_application = config["include_application"]
            
            if "include_business" in config:
                self._collection_config.include_business = config["include_business"]
            
            logger.info(f"Updated collection config: {config}")
            
        except Exception as e:
            logger.error(f"Failed to update collection config: {e}")
    
    # Internal methods
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self._running:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(self._collection_config.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _monitoring_loop(self):
        """Monitor worker health and performance."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Record worker health metrics
                await metrics_service.record_gauge(
                    "metrics_collector_health",
                    1 if self._running else 0
                )
                
                await metrics_service.record_gauge(
                    "metrics_collector_collections_completed",
                    self._performance_stats["collections_completed"]
                )
                
                await metrics_service.record_gauge(
                    "metrics_collector_collections_failed",
                    self._performance_stats["collections_failed"]
                )
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_system_metrics(self):
        """Collect all system metrics."""
        try:
            # CPU metrics
            cpu_metrics = await self.system_collector.collect_cpu_metrics()
            for name, value in cpu_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "system", "type": "cpu"}
                )
            
            # Memory metrics
            memory_metrics = await self.system_collector.collect_memory_metrics()
            for name, value in memory_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "system", "type": "memory"}
                )
            
            # Disk metrics
            disk_metrics = await self.system_collector.collect_disk_metrics()
            for name, value in disk_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "system", "type": "disk"}
                )
            
            # Network metrics
            network_metrics = await self.system_collector.collect_network_metrics()
            for name, value in network_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "system", "type": "network"}
                )
            
            self._last_collection["system"] = datetime.utcnow()
            
        except Exception as e:
            self._collection_errors["system"] = str(e)
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _collect_database_metrics(self):
        """Collect all database metrics."""
        try:
            # Connection metrics
            conn_metrics = await self.database_collector.collect_connection_metrics()
            for name, value in conn_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "database", "type": "connections"}
                )
            
            # Query metrics
            query_metrics = await self.database_collector.collect_query_metrics()
            for name, value in query_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "database", "type": "queries"}
                )
            
            # Size metrics
            size_metrics = await self.database_collector.collect_size_metrics()
            for name, value in size_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "database", "type": "size"}
                )
            
            self._last_collection["database"] = datetime.utcnow()
            
        except Exception as e:
            self._collection_errors["database"] = str(e)
            logger.error(f"Failed to collect database metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect all application metrics."""
        try:
            # Memory system metrics
            memory_metrics = await self.application_collector.collect_memory_system_metrics()
            for name, value in memory_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "application", "type": "memory_system"}
                )
            
            # Session metrics
            session_metrics = await self.application_collector.collect_session_metrics()
            for name, value in session_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "application", "type": "sessions"}
                )
            
            # Error metrics
            error_metrics = await self.application_collector.collect_error_metrics()
            for name, value in error_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "application", "type": "errors"}
                )
            
            # Workflow metrics
            workflow_metrics = await self.application_collector.collect_workflow_metrics()
            for name, value in workflow_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "application", "type": "workflows"}
                )
            
            self._last_collection["application"] = datetime.utcnow()
            
        except Exception as e:
            self._collection_errors["application"] = str(e)
            logger.error(f"Failed to collect application metrics: {e}")
    
    async def _collect_business_metrics(self):
        """Collect all business metrics."""
        try:
            # User activity metrics
            user_metrics = await self.business_collector.collect_user_activity_metrics()
            for name, value in user_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "business", "type": "users"}
                )
            
            # API metrics
            api_metrics = await self.business_collector.collect_api_metrics()
            for name, value in api_metrics.items():
                await metrics_service.record_gauge(
                    name, value, tags={"category": "business", "type": "api"}
                )
            
            self._last_collection["business"] = datetime.utcnow()
            
        except Exception as e:
            self._collection_errors["business"] = str(e)
            logger.error(f"Failed to collect business metrics: {e}")


# Global metrics collector worker instance
metrics_collector_worker = MetricsCollectorWorker()


# Convenience functions for starting/stopping the worker

async def start_metrics_collector():
    """Start the metrics collector worker."""
    await metrics_collector_worker.start()


async def stop_metrics_collector():
    """Stop the metrics collector worker."""
    await metrics_collector_worker.stop()


async def collect_metrics_once():
    """Collect all metrics once (for testing or manual collection)."""
    await metrics_collector_worker.collect_all_metrics()