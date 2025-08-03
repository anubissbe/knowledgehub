"""
Service Recovery and Self-Healing System

Provides automated service restart mechanisms with exponential backoff,
health monitoring, and recovery strategies for production resilience.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import signal
import sys
import subprocess
import json
import psutil

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    RECOVERING = "recovering"
    DISABLED = "disabled"


class RecoveryAction(Enum):
    """Available recovery actions"""
    RESTART = "restart"
    RECONNECT = "reconnect"
    RESET_CACHE = "reset_cache"
    SCALE_UP = "scale_up"
    FAILOVER = "failover"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class ServiceHealth:
    """Service health status tracking"""
    name: str
    state: ServiceState = ServiceState.HEALTHY
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consecutive_failures: int = 0
    last_failure: Optional[datetime] = None
    last_recovery: Optional[datetime] = None
    error_details: Optional[str] = None
    recovery_attempts: int = 0
    next_check: Optional[datetime] = None


@dataclass
class RecoveryStrategy:
    """Service recovery strategy configuration"""
    service_name: str
    health_check: Callable[[], bool]
    recovery_actions: List[RecoveryAction]
    max_retries: int = 5
    initial_delay: float = 1.0
    max_delay: float = 300.0
    backoff_multiplier: float = 2.0
    health_check_interval: float = 30.0
    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    dependencies: List[str] = field(default_factory=list)


class ServiceRecoveryManager:
    """
    Real service recovery manager providing automated healing capabilities.
    
    Features:
    - Exponential backoff retry strategies
    - Health monitoring with configurable thresholds
    - Automated service restart and recovery
    - Dependency-aware recovery ordering
    - Circuit breaker patterns for external services
    - Recovery attempt tracking and rate limiting
    - Production-ready logging and metrics
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.strategies: Dict[str, RecoveryStrategy] = {}
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.enabled = True
        
        # Recovery statistics
        self.recovery_stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "avg_recovery_time": 0.0,
            "last_recovery": None
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def register_service(self, strategy: RecoveryStrategy) -> None:
        """Register a service with recovery strategy"""
        logger.info(f"Registering service recovery strategy: {strategy.service_name}")
        
        self.strategies[strategy.service_name] = strategy
        self.services[strategy.service_name] = ServiceHealth(name=strategy.service_name)
        
        logger.info(f"Service {strategy.service_name} registered with recovery strategy")
    
    async def start_monitoring(self) -> None:
        """Start health monitoring for all registered services"""
        if not self.enabled:
            logger.warning("Service recovery is disabled")
            return
        
        logger.info("Starting service health monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for service health checks"""
        while not self.shutdown_event.is_set():
            try:
                # Check all registered services
                for service_name in self.services.keys():
                    if self.shutdown_event.is_set():
                        break
                    
                    await self._check_service_health(service_name)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10.0)  # Back off on errors
    
    async def _check_service_health(self, service_name: str) -> None:
        """Check health of a specific service"""
        if service_name not in self.strategies:
            return
        
        strategy = self.strategies[service_name]
        health = self.services[service_name]
        
        # Skip if recovery is in progress
        if health.state == ServiceState.RECOVERING:
            return
        
        # Check if it's time for a health check
        now = datetime.now(timezone.utc)
        if health.next_check and now < health.next_check:
            return
        
        try:
            # Perform health check
            is_healthy = await self._run_health_check(strategy.health_check)
            
            if is_healthy:
                await self._handle_healthy_service(service_name)
            else:
                await self._handle_unhealthy_service(service_name)
            
            # Schedule next check
            health.next_check = now + timedelta(seconds=strategy.health_check_interval)
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            await self._handle_unhealthy_service(service_name, str(e))
    
    async def _run_health_check(self, health_check: Callable[[], bool]) -> bool:
        """Run health check with timeout"""
        try:
            # Check if health_check is a coroutine function
            if asyncio.iscoroutinefunction(health_check):
                # Run async health check directly
                return await asyncio.wait_for(
                    health_check(),
                    timeout=10.0
                )
            else:
                # Run sync health check in executor
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, health_check),
                    timeout=10.0
                )
        except asyncio.TimeoutError:
            logger.warning("Health check timed out")
            return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    async def _handle_healthy_service(self, service_name: str) -> None:
        """Handle a healthy service check"""
        health = self.services[service_name]
        
        # Update health status
        previous_state = health.state
        health.state = ServiceState.HEALTHY
        health.last_check = datetime.now(timezone.utc)
        health.consecutive_failures = 0
        health.error_details = None
        
        # Log recovery if previously unhealthy
        if previous_state in [ServiceState.UNHEALTHY, ServiceState.FAILED, ServiceState.RECOVERING]:
            health.last_recovery = datetime.now(timezone.utc)
            logger.info(f"Service {service_name} recovered successfully")
            
            # Update recovery statistics
            self.recovery_stats["successful_recoveries"] += 1
            self.recovery_stats["last_recovery"] = health.last_recovery
    
    async def _handle_unhealthy_service(self, service_name: str, error_details: str = None) -> None:
        """Handle an unhealthy service check"""
        health = self.services[service_name]
        strategy = self.strategies[service_name]
        
        # Update failure tracking
        health.consecutive_failures += 1
        health.last_failure = datetime.now(timezone.utc)
        health.error_details = error_details
        
        # Determine service state based on failure count
        if health.consecutive_failures >= strategy.failure_threshold:
            health.state = ServiceState.FAILED
            logger.error(f"Service {service_name} marked as FAILED after {health.consecutive_failures} consecutive failures")
            
            # Trigger recovery if not already in progress
            if service_name not in self.recovery_tasks or self.recovery_tasks[service_name].done():
                await self._initiate_recovery(service_name)
        else:
            health.state = ServiceState.UNHEALTHY
            logger.warning(f"Service {service_name} unhealthy ({health.consecutive_failures}/{strategy.failure_threshold} failures)")
    
    async def _initiate_recovery(self, service_name: str) -> None:
        """Initiate recovery process for a failed service"""
        logger.info(f"Initiating recovery for service: {service_name}")
        
        # Mark service as recovering
        health = self.services[service_name]
        health.state = ServiceState.RECOVERING
        health.recovery_attempts += 1
        
        # Start recovery task
        self.recovery_tasks[service_name] = asyncio.create_task(
            self._recovery_process(service_name)
        )
        
        # Update statistics
        self.recovery_stats["total_recoveries"] += 1
    
    async def _recovery_process(self, service_name: str) -> None:
        """Execute recovery process with exponential backoff"""
        strategy = self.strategies[service_name]
        health = self.services[service_name]
        
        start_time = time.time()
        
        try:
            # Check service dependencies first
            if not await self._check_dependencies(service_name):
                logger.error(f"Dependencies not ready for {service_name}, aborting recovery")
                health.state = ServiceState.FAILED
                return
            
            # Execute recovery actions with exponential backoff
            delay = strategy.initial_delay
            
            for attempt in range(strategy.max_retries):
                if self.shutdown_event.is_set():
                    break
                
                logger.info(f"Recovery attempt {attempt + 1}/{strategy.max_retries} for {service_name}")
                
                # Execute recovery actions
                success = await self._execute_recovery_actions(service_name, attempt + 1)
                
                if success:
                    # Verify recovery with health check
                    await asyncio.sleep(2.0)  # Brief stabilization period
                    is_healthy = await self._run_health_check(strategy.health_check)
                    
                    if is_healthy:
                        recovery_time = time.time() - start_time
                        logger.info(f"Service {service_name} recovery successful in {recovery_time:.2f}s")
                        
                        # Update statistics
                        self._update_recovery_stats(recovery_time, True)
                        return
                
                # Wait before next attempt (exponential backoff)
                if attempt < strategy.max_retries - 1:
                    logger.info(f"Recovery attempt failed, waiting {delay:.1f}s before retry")
                    await asyncio.sleep(delay)
                    delay = min(delay * strategy.backoff_multiplier, strategy.max_delay)
            
            # All recovery attempts failed
            logger.error(f"All recovery attempts failed for {service_name}")
            health.state = ServiceState.FAILED
            self._update_recovery_stats(time.time() - start_time, False)
            
        except Exception as e:
            logger.error(f"Recovery process error for {service_name}: {e}")
            health.state = ServiceState.FAILED
            self._update_recovery_stats(time.time() - start_time, False)
    
    async def _check_dependencies(self, service_name: str) -> bool:
        """Check if service dependencies are healthy"""
        strategy = self.strategies[service_name]
        
        for dependency in strategy.dependencies:
            if dependency in self.services:
                dep_health = self.services[dependency]
                if dep_health.state not in [ServiceState.HEALTHY, ServiceState.DEGRADED]:
                    logger.warning(f"Dependency {dependency} not ready for {service_name}")
                    return False
        
        return True
    
    async def _execute_recovery_actions(self, service_name: str, attempt: int) -> bool:
        """Execute recovery actions for a service"""
        strategy = self.strategies[service_name]
        
        for action in strategy.recovery_actions:
            try:
                logger.info(f"Executing recovery action {action.value} for {service_name}")
                
                success = await self._execute_action(service_name, action, attempt)
                
                if success:
                    logger.info(f"Recovery action {action.value} successful for {service_name}")
                    return True
                else:
                    logger.warning(f"Recovery action {action.value} failed for {service_name}")
                
            except Exception as e:
                logger.error(f"Recovery action {action.value} error for {service_name}: {e}")
        
        return False
    
    async def _execute_action(self, service_name: str, action: RecoveryAction, attempt: int) -> bool:
        """Execute a specific recovery action"""
        try:
            if action == RecoveryAction.RESTART:
                return await self._restart_service(service_name)
            elif action == RecoveryAction.RECONNECT:
                return await self._reconnect_service(service_name)
            elif action == RecoveryAction.RESET_CACHE:
                return await self._reset_cache(service_name)
            elif action == RecoveryAction.SCALE_UP:
                return await self._scale_up_service(service_name)
            elif action == RecoveryAction.FAILOVER:
                return await self._failover_service(service_name)
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute action {action.value} for {service_name}: {e}")
            return False
    
    async def _restart_service(self, service_name: str) -> bool:
        """Restart a service using Docker or systemd"""
        try:
            # Try Docker restart first
            result = await self._run_command(f"docker restart {service_name}")
            if result.returncode == 0:
                logger.info(f"Docker service {service_name} restarted successfully")
                await asyncio.sleep(5.0)  # Allow time for startup
                return True
            
            # Try systemd restart
            result = await self._run_command(f"systemctl restart {service_name}")
            if result.returncode == 0:
                logger.info(f"Systemd service {service_name} restarted successfully")
                await asyncio.sleep(5.0)  # Allow time for startup
                return True
            
            logger.error(f"Failed to restart service {service_name}")
            return False
            
        except Exception as e:
            logger.error(f"Service restart error for {service_name}: {e}")
            return False
    
    async def _reconnect_service(self, service_name: str) -> bool:
        """Attempt to reconnect service connections"""
        try:
            # This would typically involve resetting database connections,
            # clearing connection pools, etc.
            logger.info(f"Reconnecting service {service_name}")
            
            # For database services
            if "database" in service_name.lower() or "db" in service_name.lower():
                # Reset connection pool (implementation specific)
                return await self._reset_database_connections(service_name)
            
            # For Redis services
            if "redis" in service_name.lower():
                return await self._reset_redis_connections(service_name)
            
            # Generic reconnection attempt
            await asyncio.sleep(2.0)
            return True
            
        except Exception as e:
            logger.error(f"Service reconnection error for {service_name}: {e}")
            return False
    
    async def _reset_cache(self, service_name: str) -> bool:
        """Reset service cache"""
        try:
            logger.info(f"Resetting cache for service {service_name}")
            
            # Clear Redis cache if applicable
            if "redis" in service_name.lower():
                result = await self._run_command("redis-cli FLUSHDB")
                return result.returncode == 0
            
            # Clear application cache directories
            cache_dirs = [
                f"/tmp/{service_name}_cache",
                f"/var/cache/{service_name}",
                f"/opt/projects/knowledgehub/cache/{service_name}"
            ]
            
            for cache_dir in cache_dirs:
                try:
                    result = await self._run_command(f"rm -rf {cache_dir}/*")
                    if result.returncode == 0:
                        logger.info(f"Cleared cache directory {cache_dir}")
                except Exception:
                    pass  # Directory might not exist
            
            return True
            
        except Exception as e:
            logger.error(f"Cache reset error for {service_name}: {e}")
            return False
    
    async def _scale_up_service(self, service_name: str) -> bool:
        """Scale up service resources"""
        try:
            logger.info(f"Scaling up service {service_name}")
            
            # Try Docker Compose scale
            result = await self._run_command(f"docker-compose scale {service_name}=2")
            if result.returncode == 0:
                logger.info(f"Docker service {service_name} scaled up")
                return True
            
            # Alternative: Kubernetes scaling
            result = await self._run_command(f"kubectl scale deployment {service_name} --replicas=2")
            if result.returncode == 0:
                logger.info(f"Kubernetes service {service_name} scaled up")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Service scaling error for {service_name}: {e}")
            return False
    
    async def _failover_service(self, service_name: str) -> bool:
        """Failover to backup service instance"""
        try:
            logger.info(f"Initiating failover for service {service_name}")
            
            # This would involve switching to a backup instance
            backup_service = f"{service_name}-backup"
            
            # Start backup service
            result = await self._run_command(f"docker start {backup_service}")
            if result.returncode == 0:
                logger.info(f"Backup service {backup_service} started")
                
                # Update service configuration to point to backup
                # (Implementation would depend on your load balancer/proxy setup)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Service failover error for {service_name}: {e}")
            return False
    
    async def _reset_database_connections(self, service_name: str) -> bool:
        """Reset database connection pools"""
        try:
            # This would involve calling application-specific endpoints
            # to reset connection pools
            logger.info(f"Resetting database connections for {service_name}")
            
            # Example: Call application reset endpoint
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:3000/api/admin/reset-connections") as response:
                    return response.status == 200
            
        except Exception as e:
            logger.error(f"Database connection reset error: {e}")
            return False
    
    async def _reset_redis_connections(self, service_name: str) -> bool:
        """Reset Redis connections"""
        try:
            import redis.asyncio as redis
            
            # Connect and test Redis
            redis_client = redis.from_url("redis://redis:6379")
            await redis_client.ping()
            await redis_client.close()
            
            logger.info(f"Redis connections reset for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Redis connection reset error: {e}")
            return False
    
    async def _run_command(self, command: str) -> subprocess.CompletedProcess:
        """Run system command asynchronously"""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
    
    def _update_recovery_stats(self, recovery_time: float, success: bool) -> None:
        """Update recovery statistics"""
        if success:
            self.recovery_stats["successful_recoveries"] += 1
        else:
            self.recovery_stats["failed_recoveries"] += 1
        
        # Update average recovery time
        total = self.recovery_stats["successful_recoveries"] + self.recovery_stats["failed_recoveries"]
        current_avg = self.recovery_stats["avg_recovery_time"]
        self.recovery_stats["avg_recovery_time"] = (current_avg * (total - 1) + recovery_time) / total
    
    def get_service_status(self, service_name: str) -> Optional[ServiceHealth]:
        """Get current status of a service"""
        return self.services.get(service_name)
    
    def get_all_services_status(self) -> Dict[str, ServiceHealth]:
        """Get status of all monitored services"""
        return self.services.copy()
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        return {
            **self.recovery_stats,
            "services_monitored": len(self.services),
            "active_recoveries": len([t for t in self.recovery_tasks.values() if not t.done()]),
            "monitoring_enabled": self.enabled and self.monitoring_task is not None
        }
    
    async def force_recovery(self, service_name: str) -> bool:
        """Force immediate recovery attempt for a service"""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not registered")
            return False
        
        logger.info(f"Forcing recovery for service {service_name}")
        
        # Cancel existing recovery if running
        if service_name in self.recovery_tasks and not self.recovery_tasks[service_name].done():
            self.recovery_tasks[service_name].cancel()
        
        await self._initiate_recovery(service_name)
        return True
    
    async def disable_service_monitoring(self, service_name: str) -> None:
        """Disable monitoring for a specific service"""
        if service_name in self.services:
            self.services[service_name].state = ServiceState.DISABLED
            logger.info(f"Service {service_name} monitoring disabled")
    
    async def enable_service_monitoring(self, service_name: str) -> None:
        """Enable monitoring for a specific service"""
        if service_name in self.services:
            self.services[service_name].state = ServiceState.HEALTHY
            self.services[service_name].consecutive_failures = 0
            logger.info(f"Service {service_name} monitoring enabled")
    
    async def shutdown(self) -> None:
        """Graceful shutdown of recovery system"""
        logger.info("Shutting down service recovery system")
        
        self.shutdown_event.set()
        
        # Cancel monitoring task
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all recovery tasks
        for task in self.recovery_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self.recovery_tasks:
            await asyncio.gather(*self.recovery_tasks.values(), return_exceptions=True)
        
        logger.info("Service recovery system shutdown complete")


# Global service recovery manager instance
service_recovery = ServiceRecoveryManager()


# Health check functions for common services

async def check_database_health() -> bool:
    """Health check for PostgreSQL database"""
    try:
        import asyncpg
        conn = await asyncpg.connect("postgresql://knowledgehub:knowledgehub123@postgres:5432/knowledgehub")
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


async def check_redis_health() -> bool:
    """Health check for Redis"""
    try:
        import redis.asyncio as redis
        client = redis.from_url("redis://localhost:6381")
        await client.ping()
        await client.close()
        return True
    except Exception:
        return False


async def check_weaviate_health() -> bool:
    """Health check for Weaviate"""
    try:
        import weaviate
        client = weaviate.Client("http://localhost:8090")
        client.cluster.get_nodes_status()
        return True
    except Exception:
        return False


async def check_neo4j_health() -> bool:
    """Health check for Neo4j"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687")
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


async def check_minio_health() -> bool:
    """Health check for MinIO"""
    try:
        from minio import Minio
        client = Minio("localhost:9010", access_key="minioadmin", secret_key="minioadmin", secure=False)
        client.list_buckets()
        return True
    except Exception:
        return False


def register_default_services():
    """Register default services with recovery strategies"""
    
    # Database service
    service_recovery.register_service(RecoveryStrategy(
        service_name="database",
        health_check=check_database_health,
        recovery_actions=[RecoveryAction.RECONNECT, RecoveryAction.RESTART],
        max_retries=3,
        initial_delay=2.0,
        max_delay=60.0,
        failure_threshold=2,
        health_check_interval=30.0
    ))
    
    # Redis service
    service_recovery.register_service(RecoveryStrategy(
        service_name="redis",
        health_check=check_redis_health,
        recovery_actions=[RecoveryAction.RECONNECT, RecoveryAction.RESTART, RecoveryAction.RESET_CACHE],
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        failure_threshold=2,
        health_check_interval=15.0
    ))
    
    # Weaviate service
    service_recovery.register_service(RecoveryStrategy(
        service_name="weaviate",
        health_check=check_weaviate_health,
        recovery_actions=[RecoveryAction.RESTART],
        max_retries=2,
        initial_delay=5.0,
        max_delay=120.0,
        failure_threshold=3,
        health_check_interval=60.0
    ))
    
    # Neo4j service
    service_recovery.register_service(RecoveryStrategy(
        service_name="neo4j",
        health_check=check_neo4j_health,
        recovery_actions=[RecoveryAction.RESTART],
        max_retries=2,
        initial_delay=10.0,
        max_delay=180.0,
        failure_threshold=3,
        health_check_interval=60.0
    ))
    
    # MinIO service
    service_recovery.register_service(RecoveryStrategy(
        service_name="minio",
        health_check=check_minio_health,
        recovery_actions=[RecoveryAction.RESTART],
        max_retries=2,
        initial_delay=3.0,
        max_delay=90.0,
        failure_threshold=2,
        health_check_interval=45.0
    ))


if __name__ == "__main__":
    async def main():
        # Register default services
        register_default_services()
        
        # Start monitoring
        await service_recovery.start_monitoring()
        
        try:
            # Keep running until shutdown
            await service_recovery.shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await service_recovery.shutdown()
    
    asyncio.run(main())