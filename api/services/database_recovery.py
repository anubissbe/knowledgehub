"""
Database Connection Recovery and Retry Logic

Provides resilient database connection management with automatic recovery,
connection pooling, retry mechanisms, and health monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
import threading
from collections import deque

import asyncpg
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import Session, sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import DBAPIError, DisconnectionError, OperationalError, TimeoutError

from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConnectionState(Enum):
    """Database connection states"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    FAILED = "failed"
    RECOVERING = "recovering"


class RetryStrategy(Enum):
    """Retry strategies for database operations"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class ConnectionHealth:
    """Database connection health tracking"""
    state: ConnectionState = ConnectionState.DISCONNECTED
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consecutive_failures: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_details: Optional[str] = None
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connection_wait_time_ms: float = 0.0


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retry_on_exceptions: List[type] = field(default_factory=lambda: [
        OperationalError, DisconnectionError, TimeoutError, 
        asyncpg.PostgresConnectionError, psycopg2.OperationalError
    ])
    timeout: float = 30.0


class DatabaseRecoveryManager:
    """
    Real database recovery manager providing resilient connection management.
    
    Features:
    - Automatic connection recovery with configurable retry strategies
    - Connection pooling with health monitoring
    - Circuit breaker pattern for database operations
    - Performance tracking and metrics
    - Graceful degradation under load
    - Multi-database support (PostgreSQL sync/async)
    """
    
    def __init__(self):
        self.async_engine: Optional[AsyncEngine] = None
        self.sync_engine: Optional[Any] = None
        self.psycopg2_pool: Optional[ThreadedConnectionPool] = None
        self.asyncpg_pool: Optional[asyncpg.Pool] = None
        
        self.health = ConnectionHealth()
        self.retry_config = RetryConfig()
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = timedelta(minutes=1)
        self.circuit_breaker_opened_at: Optional[datetime] = None
        
        # Performance tracking
        self.query_metrics: deque = deque(maxlen=1000)
        self.connection_metrics: deque = deque(maxlen=100)
        
        # Lock for thread safety
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
        # Initialize database connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connection configurations"""
        try:
            # Build database URLs
            self.async_db_url = self._build_async_db_url()
            self.sync_db_url = self._build_sync_db_url()
            
            logger.info("Database connection configurations initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database configurations: {e}")
            raise
    
    def _build_async_db_url(self) -> str:
        """Build async database URL"""
        return (
            f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASS}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
    
    def _build_sync_db_url(self) -> str:
        """Build sync database URL"""
        return (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASS}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
    
    async def initialize_async_engine(self) -> AsyncEngine:
        """Initialize SQLAlchemy async engine with recovery capabilities"""
        if self.async_engine:
            return self.async_engine
        
        async with self._async_lock:
            if self.async_engine:  # Double-check
                return self.async_engine
            
            try:
                logger.info("Initializing async database engine")
                
                # Create engine with connection pooling
                self.async_engine = create_async_engine(
                    self.async_db_url,
                    pool_size=settings.DB_POOL_SIZE,
                    max_overflow=settings.DB_MAX_OVERFLOW,
                    pool_timeout=30,
                    pool_recycle=3600,
                    pool_pre_ping=True,  # Verify connections before use
                    echo=settings.DEBUG,
                    connect_args={
                        "server_settings": {
                            "application_name": "knowledgehub-api",
                            "jit": "off"
                        },
                        "timeout": 10,
                        "command_timeout": 10
                    }
                )
                
                # Test connection
                async with self.async_engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                
                self.health.state = ConnectionState.CONNECTED
                self.health.last_success = datetime.now(timezone.utc)
                logger.info("Async database engine initialized successfully")
                
                return self.async_engine
                
            except Exception as e:
                logger.error(f"Failed to initialize async engine: {e}")
                self.health.state = ConnectionState.FAILED
                self.health.error_details = str(e)
                raise
    
    def initialize_sync_engine(self) -> Any:
        """Initialize SQLAlchemy sync engine with recovery capabilities"""
        if self.sync_engine:
            return self.sync_engine
        
        with self._lock:
            if self.sync_engine:  # Double-check
                return self.sync_engine
            
            try:
                logger.info("Initializing sync database engine")
                
                # Create engine with connection pooling
                self.sync_engine = create_engine(
                    self.sync_db_url,
                    pool_size=settings.DB_POOL_SIZE,
                    max_overflow=settings.DB_MAX_OVERFLOW,
                    pool_timeout=30,
                    pool_recycle=3600,
                    pool_pre_ping=True,
                    echo=settings.DEBUG,
                    connect_args={
                        "connect_timeout": 10,
                        "options": "-c statement_timeout=30000"
                    }
                )
                
                # Add event listeners for connection management
                event.listen(self.sync_engine, "connect", self._on_connect)
                event.listen(self.sync_engine, "checkout", self._on_checkout)
                event.listen(self.sync_engine, "checkin", self._on_checkin)
                
                # Test connection
                with self.sync_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                self.health.state = ConnectionState.CONNECTED
                self.health.last_success = datetime.now(timezone.utc)
                logger.info("Sync database engine initialized successfully")
                
                return self.sync_engine
                
            except Exception as e:
                logger.error(f"Failed to initialize sync engine: {e}")
                self.health.state = ConnectionState.FAILED
                self.health.error_details = str(e)
                raise
    
    async def initialize_asyncpg_pool(self) -> asyncpg.Pool:
        """Initialize asyncpg connection pool with recovery"""
        if self.asyncpg_pool:
            return self.asyncpg_pool
        
        async with self._async_lock:
            if self.asyncpg_pool:  # Double-check
                return self.asyncpg_pool
            
            try:
                logger.info("Initializing asyncpg connection pool")
                
                self.asyncpg_pool = await asyncpg.create_pool(
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    user=settings.DB_USER,
                    password=settings.DB_PASS,
                    database=settings.DB_NAME,
                    min_size=5,
                    max_size=settings.DB_POOL_SIZE,
                    timeout=10,
                    command_timeout=10,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300
                )
                
                logger.info("Asyncpg pool initialized successfully")
                return self.asyncpg_pool
                
            except Exception as e:
                logger.error(f"Failed to initialize asyncpg pool: {e}")
                self.health.state = ConnectionState.FAILED
                self.health.error_details = str(e)
                raise
    
    def initialize_psycopg2_pool(self) -> ThreadedConnectionPool:
        """Initialize psycopg2 connection pool with recovery"""
        if self.psycopg2_pool:
            return self.psycopg2_pool
        
        with self._lock:
            if self.psycopg2_pool:  # Double-check
                return self.psycopg2_pool
            
            try:
                logger.info("Initializing psycopg2 connection pool")
                
                self.psycopg2_pool = ThreadedConnectionPool(
                    minconn=5,
                    maxconn=settings.DB_POOL_SIZE,
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    user=settings.DB_USER,
                    password=settings.DB_PASS,
                    database=settings.DB_NAME,
                    connect_timeout=10,
                    options="-c statement_timeout=30000"
                )
                
                logger.info("Psycopg2 pool initialized successfully")
                return self.psycopg2_pool
                
            except Exception as e:
                logger.error(f"Failed to initialize psycopg2 pool: {e}")
                self.health.state = ConnectionState.FAILED
                self.health.error_details = str(e)
                raise
    
    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> T:
        """Execute database operation with retry logic"""
        config = retry_config or self.retry_config
        
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            raise OperationalError("Circuit breaker is open", None, None)
        
        last_exception = None
        delay = config.initial_delay
        
        for attempt in range(config.max_retries + 1):
            try:
                # Record start time
                start_time = time.time()
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success
                self._record_success(time.time() - start_time)
                return result
                
            except tuple(config.retry_on_exceptions) as e:
                last_exception = e
                self._record_failure(str(e))
                
                if attempt < config.max_retries:
                    # Calculate retry delay
                    if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        delay = min(delay * config.backoff_multiplier, config.max_delay)
                    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
                        delay = min(config.initial_delay * (attempt + 1), config.max_delay)
                    elif config.strategy == RetryStrategy.FIXED_DELAY:
                        delay = config.initial_delay
                    
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{config.max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    
                    if asyncio.iscoroutinefunction(func):
                        await asyncio.sleep(delay)
                    else:
                        time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    self._check_circuit_breaker()
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable database error: {e}")
                self._record_failure(str(e))
                raise
        
        # All retries exhausted
        if last_exception:
            raise last_exception
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """Get async database session with automatic recovery"""
        if not self.async_engine:
            await self.initialize_async_engine()
        
        AsyncSessionLocal = sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async with AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_asyncpg_connection(self) -> asyncpg.Connection:
        """Get asyncpg connection with automatic recovery"""
        if not self.asyncpg_pool:
            await self.initialize_asyncpg_pool()
        
        # Retry logic for acquiring connection
        retry_count = 0
        while retry_count < 3:
            try:
                async with self.asyncpg_pool.acquire() as connection:
                    # Set statement timeout
                    await connection.execute("SET statement_timeout = '30s'")
                    yield connection
                    return
            except asyncpg.PoolConnectionLimitError:
                retry_count += 1
                if retry_count < 3:
                    await asyncio.sleep(1)
                else:
                    raise
            except Exception as e:
                logger.error(f"Error acquiring asyncpg connection: {e}")
                raise
    
    def get_sync_session(self) -> Session:
        """Get sync database session with automatic recovery"""
        if not self.sync_engine:
            self.initialize_sync_engine()
        
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.sync_engine
        )
        
        return SessionLocal()
    
    @contextmanager
    def get_psycopg2_connection(self):
        """Get psycopg2 connection with automatic recovery"""
        if not self.psycopg2_pool:
            self.initialize_psycopg2_pool()
        
        connection = None
        try:
            connection = self.psycopg2_pool.getconn()
            connection.autocommit = False
            yield connection
            connection.commit()
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                self.psycopg2_pool.putconn(connection)
    
    async def health_check(self) -> bool:
        """Perform database health check"""
        try:
            # Test async connection
            if self.async_engine:
                async with self.async_engine.connect() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    await result.fetchone()
            
            # Test asyncpg pool
            if self.asyncpg_pool:
                async with self.asyncpg_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
            
            self.health.state = ConnectionState.CONNECTED
            self.health.last_check = datetime.now(timezone.utc)
            self.health.consecutive_failures = 0
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            self.health.state = ConnectionState.FAILED
            self.health.consecutive_failures += 1
            self.health.last_failure = datetime.now(timezone.utc)
            self.health.error_details = str(e)
            return False
    
    async def recover_connections(self) -> bool:
        """Attempt to recover all database connections"""
        logger.info("Attempting to recover database connections")
        self.health.state = ConnectionState.RECOVERING
        
        recovery_successful = True
        
        # Recover async engine
        try:
            if self.async_engine:
                await self.async_engine.dispose()
            self.async_engine = None
            await self.initialize_async_engine()
        except Exception as e:
            logger.error(f"Failed to recover async engine: {e}")
            recovery_successful = False
        
        # Recover asyncpg pool
        try:
            if self.asyncpg_pool:
                await self.asyncpg_pool.close()
            self.asyncpg_pool = None
            await self.initialize_asyncpg_pool()
        except Exception as e:
            logger.error(f"Failed to recover asyncpg pool: {e}")
            recovery_successful = False
        
        # Recover sync engine
        try:
            if self.sync_engine:
                self.sync_engine.dispose()
            self.sync_engine = None
            self.initialize_sync_engine()
        except Exception as e:
            logger.error(f"Failed to recover sync engine: {e}")
            recovery_successful = False
        
        # Recover psycopg2 pool
        try:
            if self.psycopg2_pool:
                self.psycopg2_pool.closeall()
            self.psycopg2_pool = None
            self.initialize_psycopg2_pool()
        except Exception as e:
            logger.error(f"Failed to recover psycopg2 pool: {e}")
            recovery_successful = False
        
        if recovery_successful:
            self.health.state = ConnectionState.CONNECTED
            self.health.consecutive_failures = 0
            self.circuit_breaker_open = False
            self.circuit_breaker_failures = 0
            logger.info("Database connections recovered successfully")
        else:
            self.health.state = ConnectionState.FAILED
            logger.error("Database connection recovery failed")
        
        return recovery_successful
    
    def _on_connect(self, dbapi_conn, connection_record):
        """Handle new database connection"""
        connection_record.info['connect_time'] = time.time()
        self.health.total_connections += 1
        logger.debug("New database connection established")
    
    def _on_checkout(self, dbapi_conn, connection_record, connection_proxy):
        """Handle connection checkout from pool"""
        checkout_time = time.time()
        connection_record.info['checkout_time'] = checkout_time
        
        # Track connection wait time
        if 'connect_time' in connection_record.info:
            wait_time = (checkout_time - connection_record.info['connect_time']) * 1000
            self.health.connection_wait_time_ms = wait_time
        
        self.health.active_connections += 1
        self.health.idle_connections = max(0, self.health.idle_connections - 1)
    
    def _on_checkin(self, dbapi_conn, connection_record):
        """Handle connection checkin to pool"""
        if 'checkout_time' in connection_record.info:
            usage_time = time.time() - connection_record.info['checkout_time']
            self.connection_metrics.append({
                'timestamp': datetime.now(timezone.utc),
                'usage_time': usage_time
            })
        
        self.health.active_connections = max(0, self.health.active_connections - 1)
        self.health.idle_connections += 1
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_breaker_open:
            return False
        
        # Check if timeout has passed
        if self.circuit_breaker_opened_at:
            elapsed = datetime.now(timezone.utc) - self.circuit_breaker_opened_at
            if elapsed > self.circuit_breaker_timeout:
                logger.info("Circuit breaker timeout reached, attempting reset")
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                return False
        
        return True
    
    def _check_circuit_breaker(self):
        """Check and update circuit breaker state"""
        self.circuit_breaker_failures += 1
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            self.circuit_breaker_opened_at = datetime.now(timezone.utc)
            logger.error(
                f"Circuit breaker opened after {self.circuit_breaker_failures} failures"
            )
    
    def _record_success(self, execution_time: float):
        """Record successful operation"""
        self.circuit_breaker_failures = 0
        self.health.consecutive_failures = 0
        self.health.last_success = datetime.now(timezone.utc)
        
        self.query_metrics.append({
            'timestamp': datetime.now(timezone.utc),
            'execution_time': execution_time,
            'success': True
        })
    
    def _record_failure(self, error: str):
        """Record failed operation"""
        self.health.consecutive_failures += 1
        self.health.last_failure = datetime.now(timezone.utc)
        self.health.error_details = error
        
        self.query_metrics.append({
            'timestamp': datetime.now(timezone.utc),
            'execution_time': 0,
            'success': False,
            'error': error
        })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get database connection statistics"""
        # Calculate success rate
        recent_queries = list(self.query_metrics)
        if recent_queries:
            success_count = sum(1 for q in recent_queries if q.get('success', False))
            success_rate = (success_count / len(recent_queries)) * 100
        else:
            success_rate = 100.0
        
        # Calculate average execution time
        successful_queries = [q for q in recent_queries if q.get('success', False)]
        avg_execution_time = (
            sum(q['execution_time'] for q in successful_queries) / len(successful_queries)
            if successful_queries else 0
        )
        
        return {
            'health': {
                'state': self.health.state.value,
                'last_check': self.health.last_check.isoformat() if self.health.last_check else None,
                'consecutive_failures': self.health.consecutive_failures,
                'last_failure': self.health.last_failure.isoformat() if self.health.last_failure else None,
                'last_success': self.health.last_success.isoformat() if self.health.last_success else None,
                'error_details': self.health.error_details
            },
            'connections': {
                'total': self.health.total_connections,
                'active': self.health.active_connections,
                'idle': self.health.idle_connections,
                'wait_time_ms': self.health.connection_wait_time_ms
            },
            'performance': {
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time,
                'total_queries': len(recent_queries)
            },
            'circuit_breaker': {
                'open': self.circuit_breaker_open,
                'failures': self.circuit_breaker_failures,
                'threshold': self.circuit_breaker_threshold,
                'opened_at': (
                    self.circuit_breaker_opened_at.isoformat() 
                    if self.circuit_breaker_opened_at else None
                )
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown all database connections"""
        logger.info("Shutting down database connections")
        
        # Close async engine
        if self.async_engine:
            await self.async_engine.dispose()
            self.async_engine = None
        
        # Close asyncpg pool
        if self.asyncpg_pool:
            await self.asyncpg_pool.close()
            self.asyncpg_pool = None
        
        # Close sync engine
        if self.sync_engine:
            self.sync_engine.dispose()
            self.sync_engine = None
        
        # Close psycopg2 pool
        if self.psycopg2_pool:
            self.psycopg2_pool.closeall()
            self.psycopg2_pool = None
        
        self.health.state = ConnectionState.DISCONNECTED
        logger.info("Database connections shut down successfully")


# Global database recovery manager instance
db_recovery = DatabaseRecoveryManager()


# Decorator for database operations with retry
def with_db_retry(retry_config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to database operations"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            return await db_recovery.execute_with_retry(
                func, *args, retry_config=retry_config, **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    db_recovery.execute_with_retry(
                        func, *args, retry_config=retry_config, **kwargs
                    )
                )
            finally:
                loop.close()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage functions
@with_db_retry()
async def execute_query_with_retry(query: str, params: Optional[Dict] = None):
    """Execute a database query with automatic retry"""
    async with db_recovery.get_asyncpg_connection() as conn:
        if params:
            return await conn.fetch(query, *params.values())
        else:
            return await conn.fetch(query)


@with_db_retry(retry_config=RetryConfig(max_retries=5, initial_delay=2.0))
async def critical_database_operation(data: Dict[str, Any]):
    """Execute critical database operation with custom retry config"""
    async with db_recovery.get_async_session() as session:
        # Perform critical operation
        result = await session.execute(
            text("INSERT INTO critical_data (data) VALUES (:data)"),
            {"data": data}
        )
        return result