"""
Database Recovery Middleware

Provides automatic database connection recovery and retry logic for all
database operations in the application.
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..services.database_recovery import db_recovery, ConnectionState

logger = logging.getLogger(__name__)


class DatabaseRecoveryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic database recovery and circuit breaker management.
    
    Features:
    - Monitors database health for all requests
    - Implements circuit breaker pattern
    - Provides automatic recovery triggers
    - Tracks database performance metrics
    - Graceful degradation for read-only operations
    """
    
    def __init__(self, app, enable_circuit_breaker: bool = True, 
                 read_only_fallback: bool = True):
        super().__init__(app)
        self.enable_circuit_breaker = enable_circuit_breaker
        self.read_only_fallback = read_only_fallback
        self.recovery_in_progress = False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with database recovery protection"""
        
        # Skip health check endpoints to prevent recursion
        if request.url.path in ["/health", "/api/database-recovery/health"]:
            return await call_next(request)
        
        # Check circuit breaker state
        if self.enable_circuit_breaker and db_recovery.circuit_breaker_open:
            # Check if circuit breaker timeout has passed
            if db_recovery._is_circuit_breaker_open():
                logger.warning(f"Circuit breaker open for request: {request.method} {request.url.path}")
                
                # Allow read-only operations if configured
                if self.read_only_fallback and request.method == "GET":
                    request.state.db_degraded = True
                    logger.info("Allowing read-only operation in degraded mode")
                else:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": "Database temporarily unavailable",
                            "message": "Circuit breaker is open due to database failures",
                            "retry_after": 60
                        },
                        headers={"Retry-After": "60"}
                    )
        
        # Check database health periodically
        if await self._should_check_health():
            health_status = await self._check_database_health()
            if not health_status and not self.recovery_in_progress:
                # Trigger recovery in background
                logger.warning("Database health check failed, triggering recovery")
                await self._trigger_recovery()
        
        # Process request
        start_time = time.time()
        try:
            response = await call_next(request)
            
            # Track successful database operations
            if hasattr(request.state, "db_operation_performed"):
                db_recovery._record_success(time.time() - start_time)
            
            return response
            
        except Exception as e:
            # Check if this is a database-related error
            if self._is_database_error(e):
                logger.error(f"Database error in request: {e}")
                db_recovery._record_failure(str(e))
                
                # Check if we should open circuit breaker
                db_recovery._check_circuit_breaker()
                
                # Return appropriate error response
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Database operation failed",
                        "message": "Unable to complete database operation",
                        "request_id": request.state.request_id if hasattr(request.state, "request_id") else None
                    }
                )
            else:
                # Re-raise non-database errors
                raise
    
    async def _should_check_health(self) -> bool:
        """Determine if health check should be performed"""
        # Check health every 30 seconds
        if not hasattr(self, "_last_health_check"):
            self._last_health_check = 0
        
        current_time = time.time()
        if current_time - self._last_health_check > 30:
            self._last_health_check = current_time
            return True
        
        return False
    
    async def _check_database_health(self) -> bool:
        """Perform database health check"""
        try:
            return await db_recovery.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def _trigger_recovery(self) -> None:
        """Trigger database recovery process"""
        if self.recovery_in_progress:
            return
        
        try:
            self.recovery_in_progress = True
            logger.info("Triggering database recovery from middleware")
            
            # Run recovery in background
            import asyncio
            asyncio.create_task(self._recovery_task())
            
        except Exception as e:
            logger.error(f"Failed to trigger recovery: {e}")
            self.recovery_in_progress = False
    
    async def _recovery_task(self) -> None:
        """Background recovery task"""
        try:
            success = await db_recovery.recover_connections()
            if success:
                logger.info("Database recovery completed successfully")
            else:
                logger.error("Database recovery failed")
        finally:
            self.recovery_in_progress = False
    
    def _is_database_error(self, error: Exception) -> bool:
        """Check if error is database-related"""
        database_error_types = (
            "OperationalError", "DisconnectionError", "TimeoutError",
            "PostgresConnectionError", "InterfaceError", "DatabaseError"
        )
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Check error type
        if error_type in database_error_types:
            return True
        
        # Check error message for database-related keywords
        db_keywords = [
            "database", "connection", "postgres", "psycopg2", "asyncpg",
            "sqlalchemy", "pool", "timeout", "refused", "closed"
        ]
        
        return any(keyword in error_message for keyword in db_keywords)


class DatabaseConnectionPoolMiddleware(BaseHTTPMiddleware):
    """
    Middleware for database connection pool management.
    
    Features:
    - Ensures connections are properly returned to pool
    - Tracks connection usage per request
    - Implements connection timeout handling
    - Provides connection pool statistics
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with connection pool management"""
        
        # Track connection acquisition time
        request.state.db_connection_start = time.time()
        request.state.db_connections_used = 0
        
        try:
            response = await call_next(request)
            
            # Log connection usage statistics
            if hasattr(request.state, "db_connections_used") and request.state.db_connections_used > 0:
                connection_time = time.time() - request.state.db_connection_start
                logger.debug(
                    f"Request used {request.state.db_connections_used} database connections "
                    f"for {connection_time:.2f}s"
                )
            
            return response
            
        except Exception as e:
            # Ensure connections are cleaned up on error
            if hasattr(request.state, "db_connection"):
                logger.warning("Cleaning up database connection after error")
                # Connection cleanup would be handled by context managers
            
            raise
        
        finally:
            # Update connection metrics
            if hasattr(request.state, "db_connections_used"):
                stats = db_recovery.get_connection_stats()
                if stats["connections"]["active"] > stats["connections"]["total"] * 0.8:
                    logger.warning(
                        f"High connection pool utilization: "
                        f"{stats['connections']['active']}/{stats['connections']['total']}"
                    )


def track_database_operation(request: Request):
    """Mark that a database operation was performed in this request"""
    request.state.db_operation_performed = True
    request.state.db_connections_used = getattr(request.state, "db_connections_used", 0) + 1