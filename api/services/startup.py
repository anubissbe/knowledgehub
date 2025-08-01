"""Application startup and shutdown logic"""

import logging
from typing import Optional

from ..models import init_db
from .cache import redis_client
from .vector_store import vector_store
from .message_queue import message_queue

logger = logging.getLogger(__name__)


async def initialize_services():
    """Initialize all services on startup"""
    logger.info("Initializing services...")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Initialize Redis connection
    try:
        await redis_client.initialize()
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        raise
    
    # Initialize vector store
    try:
        await vector_store.initialize()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        # Make vector store optional - just log the error but don't fail startup
        logger.warning("Continuing without vector store functionality")
    
    # Initialize message queue
    try:
        await message_queue.initialize()
        logger.info("Message queue initialized")
    except Exception as e:
        logger.error(f"Failed to initialize message queue: {e}")
        raise
    
    # Initialize session cleanup service
    try:
        from ..memory_system.services.session_cleanup import session_cleanup_service
        await session_cleanup_service.start()
        logger.info("Session cleanup service started")
    except ImportError:
        logger.info("Memory system not available, skipping session cleanup service")
    except Exception as e:
        logger.error(f"Failed to start session cleanup service: {e}")
        # Don't fail startup if cleanup service fails
        logger.warning("Continuing without session cleanup service")
    
    # Initialize scheduler service
    try:
        from .scheduler import SchedulerService
        scheduler_service = SchedulerService()
        await scheduler_service.start_background_scheduler()
        logger.info("Scheduler service started")
    except Exception as e:
        logger.error(f"Failed to start scheduler service: {e}")
        # Don't fail startup if scheduler service fails
        logger.warning("Continuing without scheduler service")
    
    # Initialize performance optimization system
    try:
        from ..performance.manager import initialize_performance_system
        await initialize_performance_system()
        logger.info("Performance optimization system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize performance system: {e}")
        # Don't fail startup if performance system fails
        logger.warning("Continuing without performance optimization system")
    
    logger.info("All services initialized successfully")


async def shutdown_services():
    """Cleanup services on shutdown"""
    logger.info("Shutting down services...")
    
    # Close Redis connection
    try:
        await redis_client.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis: {e}")
    
    # Close vector store connection
    try:
        await vector_store.close()
        logger.info("Vector store connection closed")
    except Exception as e:
        logger.error(f"Error closing vector store: {e}")
    
    # Close message queue
    try:
        await message_queue.close()
        logger.info("Message queue closed")
    except Exception as e:
        logger.error(f"Error closing message queue: {e}")
    
    # Stop session cleanup service
    try:
        from ..memory_system.services.session_cleanup import session_cleanup_service
        await session_cleanup_service.stop()
        logger.info("Session cleanup service stopped")
    except ImportError:
        pass  # Memory system not available
    except Exception as e:
        logger.error(f"Error stopping session cleanup service: {e}")
    
    # Shutdown performance optimization system
    try:
        from ..performance.manager import shutdown_performance_system
        await shutdown_performance_system()
        logger.info("Performance optimization system shut down")
    except Exception as e:
        logger.error(f"Error shutting down performance system: {e}")
    
    logger.info("All services shut down")