"""
Real Startup Service for KnowledgeHub.

This service initializes all real AI and WebSocket services in the correct order.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import time

from ..services.real_embeddings_service import real_embeddings_service, start_embeddings_service
from ..services.real_websocket_events import real_websocket_events, start_websocket_events
from ..services.real_ai_intelligence import real_ai_intelligence, start_ai_intelligence
from ..websocket.manager import websocket_manager, start_websocket_manager
from ..services.cache import redis_client
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("real_startup")


class RealStartupService:
    """
    Startup service for initializing all real KnowledgeHub services.
    
    Features:
    - Proper service initialization order
    - Dependency management
    - Health checking
    - Graceful shutdown
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Service registry
        self.services = {
            "redis": redis_client,
            "embeddings": real_embeddings_service,
            "websocket_manager": websocket_manager,
            "websocket_events": real_websocket_events,
            "ai_intelligence": real_ai_intelligence
        }
        
        # Service status
        self.service_status = {}
        self.startup_time = None
        self.is_started = False
        
        logger.info("Initialized RealStartupService")
    
    async def start_all_services(self):
        """Start all services in the correct order."""
        if self.is_started:
            logger.warning("Services already started")
            return
        
        startup_start = time.time()
        logger.info("Starting all KnowledgeHub real services...")
        
        try:
            # Step 1: Initialize Redis (required by all other services)
            await self._start_service("redis", redis_client.initialize)
            
            # Step 2: Start embeddings service (required by AI intelligence)
            await self._start_service("embeddings", start_embeddings_service)
            
            # Step 3: Start WebSocket manager (required by events)
            await self._start_service("websocket_manager", start_websocket_manager)
            
            # Step 4: Start WebSocket events (depends on manager)
            await self._start_service("websocket_events", start_websocket_events)
            
            # Step 5: Start AI intelligence (depends on embeddings and events)
            await self._start_service("ai_intelligence", start_ai_intelligence)
            
            self.startup_time = time.time() - startup_start
            self.is_started = True
            
            logger.info(f"All services started successfully in {self.startup_time:.2f}s")
            
            # Print service status
            await self._log_service_status()
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            await self.stop_all_services()
            raise
    
    async def stop_all_services(self):
        """Stop all services in reverse order."""
        if not self.is_started:
            logger.warning("Services not started")
            return
        
        logger.info("Stopping all KnowledgeHub real services...")
        
        # Stop in reverse order
        stop_order = [
            ("ai_intelligence", real_ai_intelligence.stop),
            ("websocket_events", real_websocket_events.stop),
            ("websocket_manager", websocket_manager.stop),
            ("embeddings", real_embeddings_service.stop),
            # Redis cleanup is handled automatically
        ]
        
        for service_name, stop_func in stop_order:
            try:
                await stop_func()
                self.service_status[service_name] = "stopped"
                logger.info(f"Stopped {service_name}")
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
        
        self.is_started = False
        logger.info("All services stopped")
    
    async def restart_service(self, service_name: str):
        """Restart a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        logger.info(f"Restarting service: {service_name}")
        
        try:
            # Stop the service
            if service_name == "embeddings":
                await real_embeddings_service.stop()
            elif service_name == "websocket_events":
                await real_websocket_events.stop()
            elif service_name == "ai_intelligence":
                await real_ai_intelligence.stop()
            elif service_name == "websocket_manager":
                await websocket_manager.stop()
            
            # Wait a moment
            await asyncio.sleep(1)
            
            # Start the service
            if service_name == "embeddings":
                await start_embeddings_service()
            elif service_name == "websocket_events":
                await start_websocket_events()
            elif service_name == "ai_intelligence":
                await start_ai_intelligence()
            elif service_name == "websocket_manager":
                await start_websocket_manager()
            
            self.service_status[service_name] = "running"
            logger.info(f"Restarted {service_name} successfully")
            
        except Exception as e:
            self.service_status[service_name] = "error"
            logger.error(f"Failed to restart {service_name}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_status = {
            "overall": "healthy",
            "services": {},
            "startup_time": self.startup_time,
            "uptime": time.time() - (time.time() - (self.startup_time or 0))
        }
        
        try:
            # Check Redis
            try:
                await redis_client.ping()
                health_status["services"]["redis"] = {"status": "healthy", "details": "Connected"}
            except Exception as e:
                health_status["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall"] = "degraded"
            
            # Check embeddings service
            embeddings_stats = real_embeddings_service.get_embedding_stats()
            health_status["services"]["embeddings"] = {
                "status": "healthy" if embeddings_stats["running"] else "stopped",
                "details": embeddings_stats
            }
            if not embeddings_stats["running"]:
                health_status["overall"] = "degraded"
            
            # Check WebSocket manager
            ws_stats = websocket_manager.get_manager_stats()
            health_status["services"]["websocket_manager"] = {
                "status": "healthy",
                "details": {
                    "active_connections": ws_stats["active_connections"],
                    "authenticated_connections": ws_stats["authenticated_connections"]
                }
            }
            
            # Check WebSocket events
            events_stats = real_websocket_events.get_event_stats()
            health_status["services"]["websocket_events"] = {
                "status": "healthy" if events_stats["running"] else "stopped",
                "details": events_stats
            }
            if not events_stats["running"]:
                health_status["overall"] = "degraded"
            
            # Check AI intelligence
            ai_stats = real_ai_intelligence.get_ai_stats()
            health_status["services"]["ai_intelligence"] = {
                "status": "healthy" if ai_stats["running"] else "stopped",
                "details": ai_stats
            }
            if not ai_stats["running"]:
                health_status["overall"] = "degraded"
            
        except Exception as e:
            health_status["overall"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        try:
            return {
                "embeddings": real_embeddings_service.get_embedding_stats(),
                "websocket_manager": websocket_manager.get_manager_stats(),
                "websocket_events": real_websocket_events.get_event_stats(),
                "ai_intelligence": real_ai_intelligence.get_ai_stats(),
                "startup": {
                    "startup_time": self.startup_time,
                    "is_started": self.is_started,
                    "service_status": self.service_status
                }
            }
        except Exception as e:
            logger.error(f"Failed to get service metrics: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    async def _start_service(self, service_name: str, start_func):
        """Start a single service with error handling."""
        try:
            logger.info(f"Starting {service_name}...")
            start_time = time.time()
            
            await start_func()
            
            elapsed = time.time() - start_time
            self.service_status[service_name] = "running"
            
            logger.info(f"Started {service_name} in {elapsed:.2f}s")
            
        except Exception as e:
            self.service_status[service_name] = "error"
            logger.error(f"Failed to start {service_name}: {e}")
            raise
    
    async def _log_service_status(self):
        """Log the status of all services."""
        logger.info("Service Status Summary:")
        logger.info("=" * 50)
        
        for service_name, status in self.service_status.items():
            logger.info(f"  {service_name:<20}: {status}")
        
        logger.info("=" * 50)
        
        # Log service metrics
        try:
            embeddings_stats = real_embeddings_service.get_embedding_stats()
            logger.info(f"Embeddings: {embeddings_stats['embeddings_generated']} generated, "
                       f"{embeddings_stats['cache_hits']} cache hits")
            
            ws_stats = websocket_manager.get_manager_stats()
            logger.info(f"WebSocket: {ws_stats['active_connections']} active connections")
            
            events_stats = real_websocket_events.get_event_stats()
            logger.info(f"Events: {events_stats['events_published']} events published")
            
            ai_stats = real_ai_intelligence.get_ai_stats()
            logger.info(f"AI Intelligence: {ai_stats['patterns_recognized']} patterns recognized, "
                       f"{ai_stats['predictions_made']} predictions made")
            
        except Exception as e:
            logger.debug(f"Error logging service metrics: {e}")


# Global startup service instance
real_startup_service = RealStartupService()


# Convenience functions

async def start_all_real_services():
    """Start all real KnowledgeHub services."""
    await real_startup_service.start_all_services()


async def stop_all_real_services():
    """Stop all real KnowledgeHub services."""
    await real_startup_service.stop_all_services()


async def get_real_services_health():
    """Get health status of all real services."""
    return await real_startup_service.health_check()


async def get_real_services_metrics():
    """Get metrics from all real services."""
    return await real_startup_service.get_service_metrics()


# FastAPI startup/shutdown handlers

async def startup_handler():
    """FastAPI startup handler."""
    logger.info("FastAPI startup: initializing real services")
    try:
        await start_all_real_services()
        logger.info("FastAPI startup: all real services initialized")
    except Exception as e:
        logger.error(f"FastAPI startup failed: {e}")
        raise


async def shutdown_handler():
    """FastAPI shutdown handler."""
    logger.info("FastAPI shutdown: stopping real services")
    try:
        await stop_all_real_services()
        logger.info("FastAPI shutdown: all real services stopped")
    except Exception as e:
        logger.error(f"FastAPI shutdown error: {e}")