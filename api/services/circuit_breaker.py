"""
Circuit Breaker Service for External Service Calls

Provides resilient external service communication with circuit breaker patterns,
retry logic, fallback mechanisms, and comprehensive monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union, Generic
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
import aiohttp
import httpx
from functools import wraps
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class ServiceType(Enum):
    """Types of external services"""
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    CACHE = "cache"
    SEARCH = "search"
    ML_MODEL = "ml_model"


@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_max_calls: int = 3
    error_timeout: float = 30.0
    exclude_exceptions: List[type] = field(default_factory=lambda: [])
    include_exceptions: List[type] = field(default_factory=lambda: [
        aiohttp.ClientError, httpx.HTTPError, TimeoutError,
        ConnectionError, OSError
    ])


@dataclass
class ServiceHealth:
    """Service health tracking"""
    name: str
    service_type: ServiceType
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    half_open_calls: int = 0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    total_calls: int = 0


@dataclass
class CallMetrics:
    """Metrics for a service call"""
    timestamp: datetime
    duration: float
    success: bool
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_size: Optional[int] = None


class CircuitBreaker(Generic[T]):
    """
    Generic circuit breaker for protecting external service calls.
    
    Features:
    - State management (closed, open, half-open)
    - Configurable failure thresholds
    - Automatic recovery testing
    - Fallback mechanisms
    - Performance tracking
    - Error classification
    """
    
    def __init__(
        self,
        name: str,
        service_type: ServiceType = ServiceType.REST_API,
        config: Optional[CircuitConfig] = None,
        fallback: Optional[Callable[..., T]] = None
    ):
        self.name = name
        self.service_type = service_type
        self.config = config or CircuitConfig()
        self.fallback = fallback
        
        self.health = ServiceHealth(name=name, service_type=service_type)
        self.metrics: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
        # State change callbacks
        self.on_open: Optional[Callable] = None
        self.on_close: Optional[Callable] = None
        self.on_half_open: Optional[Callable] = None
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        
        # Check if circuit is open
        if self.health.state == CircuitState.OPEN:
            if not self._should_attempt_reset():
                logger.warning(f"Circuit breaker {self.name} is OPEN, rejecting call")
                if self.fallback:
                    return await self._execute_fallback(*args, **kwargs)
                raise CircuitBreakerOpen(f"Circuit breaker {self.name} is open")
            else:
                # Transition to half-open
                await self._transition_to_half_open()
        
        # Check half-open limit
        if self.health.state == CircuitState.HALF_OPEN:
            async with self._async_lock:
                if self.health.half_open_calls >= self.config.half_open_max_calls:
                    logger.warning(f"Circuit breaker {self.name} half-open limit reached")
                    if self.fallback:
                        return await self._execute_fallback(*args, **kwargs)
                    raise CircuitBreakerOpen(f"Circuit breaker {self.name} half-open limit reached")
                self.health.half_open_calls += 1
        
        # Execute the call
        start_time = time.time()
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            duration = time.time() - start_time
            await self._record_success(duration)
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            await self._record_failure(duration, e)
            
            # Check if we should count this failure
            if self._should_count_exception(e):
                # Use fallback if available
                if self.fallback:
                    logger.info(f"Using fallback for {self.name} due to: {e}")
                    return await self._execute_fallback(*args, **kwargs)
            
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if not self.health.last_state_change:
            return True
        
        time_since_open = (
            datetime.now(timezone.utc) - self.health.last_state_change
        ).total_seconds()
        
        return time_since_open >= self.config.timeout
    
    async def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        async with self._async_lock:
            if self.health.state == CircuitState.OPEN:
                logger.info(f"Circuit breaker {self.name} transitioning to HALF-OPEN")
                self.health.state = CircuitState.HALF_OPEN
                self.health.last_state_change = datetime.now(timezone.utc)
                self.health.half_open_calls = 0
                
                if self.on_half_open:
                    await self._execute_callback(self.on_half_open)
    
    async def _record_success(self, duration: float):
        """Record successful call"""
        async with self._async_lock:
            self.health.success_count += 1
            self.health.total_calls += 1
            self.health.last_success = datetime.now(timezone.utc)
            
            # Add metrics
            self.metrics.append(CallMetrics(
                timestamp=datetime.now(timezone.utc),
                duration=duration,
                success=True
            ))
            
            # Update average response time
            self._update_avg_response_time()
            
            # Handle state transitions
            if self.health.state == CircuitState.HALF_OPEN:
                if self.health.success_count >= self.config.success_threshold:
                    await self._close_circuit()
            elif self.health.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.health.failure_count = 0
    
    async def _record_failure(self, duration: float, error: Exception):
        """Record failed call"""
        if not self._should_count_exception(error):
            return
        
        async with self._async_lock:
            self.health.failure_count += 1
            self.health.total_calls += 1
            self.health.last_failure = datetime.now(timezone.utc)
            
            # Add metrics
            self.metrics.append(CallMetrics(
                timestamp=datetime.now(timezone.utc),
                duration=duration,
                success=False,
                error=str(error)
            ))
            
            # Update error rate
            self._update_error_rate()
            
            # Handle state transitions
            if self.health.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                await self._open_circuit()
            elif self.health.state == CircuitState.CLOSED:
                if self.health.failure_count >= self.config.failure_threshold:
                    await self._open_circuit()
    
    async def _open_circuit(self):
        """Open the circuit breaker"""
        logger.warning(f"Opening circuit breaker {self.name} after {self.health.failure_count} failures")
        self.health.state = CircuitState.OPEN
        self.health.last_state_change = datetime.now(timezone.utc)
        self.health.success_count = 0
        
        if self.on_open:
            await self._execute_callback(self.on_open)
    
    async def _close_circuit(self):
        """Close the circuit breaker"""
        logger.info(f"Closing circuit breaker {self.name} after successful recovery")
        self.health.state = CircuitState.CLOSED
        self.health.last_state_change = datetime.now(timezone.utc)
        self.health.failure_count = 0
        self.health.half_open_calls = 0
        
        if self.on_close:
            await self._execute_callback(self.on_close)
    
    def _should_count_exception(self, error: Exception) -> bool:
        """Check if exception should be counted as failure"""
        error_type = type(error)
        
        # Check exclude list first
        if error_type in self.config.exclude_exceptions:
            return False
        
        # Check include list
        return error_type in self.config.include_exceptions
    
    async def _execute_fallback(self, *args, **kwargs) -> T:
        """Execute fallback function"""
        if asyncio.iscoroutinefunction(self.fallback):
            return await self.fallback(*args, **kwargs)
        else:
            return self.fallback(*args, **kwargs)
    
    async def _execute_callback(self, callback: Callable):
        """Execute state change callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(self)
            else:
                callback(self)
        except Exception as e:
            logger.error(f"Error in circuit breaker callback: {e}")
    
    def _update_avg_response_time(self):
        """Update average response time metric"""
        recent_metrics = [m for m in self.metrics if m.success]
        if recent_metrics:
            self.health.avg_response_time = (
                sum(m.duration for m in recent_metrics) / len(recent_metrics)
            )
    
    def _update_error_rate(self):
        """Update error rate metric"""
        if self.metrics:
            error_count = sum(1 for m in self.metrics if not m.success)
            self.health.error_rate = (error_count / len(self.metrics)) * 100
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "name": self.name,
            "service_type": self.service_type.value,
            "state": self.health.state.value,
            "failure_count": self.health.failure_count,
            "success_count": self.health.success_count,
            "last_failure": self.health.last_failure.isoformat() if self.health.last_failure else None,
            "last_success": self.health.last_success.isoformat() if self.health.last_success else None,
            "error_rate": self.health.error_rate,
            "avg_response_time": self.health.avg_response_time,
            "total_calls": self.health.total_calls,
            "metrics_window": len(self.metrics)
        }
    
    async def reset(self):
        """Manually reset circuit breaker"""
        async with self._async_lock:
            logger.info(f"Manually resetting circuit breaker {self.name}")
            self.health.state = CircuitState.CLOSED
            self.health.failure_count = 0
            self.health.success_count = 0
            self.health.half_open_calls = 0
            self.health.last_state_change = datetime.now(timezone.utc)


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services.
    
    Features:
    - Centralized circuit breaker management
    - Service discovery and registration
    - Bulk operations and monitoring
    - Fallback strategies
    - Health aggregation
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        
        # Global fallback strategies
        self.fallback_strategies: Dict[ServiceType, Callable] = {}
        
        # Initialize default circuit breakers
        self._initialize_default_breakers()
    
    def _initialize_default_breakers(self):
        """Initialize circuit breakers for common services"""
        
        # External API circuit breaker
        self.register_circuit_breaker(
            name="external_api",
            service_type=ServiceType.REST_API,
            config=CircuitConfig(
                failure_threshold=3,
                timeout=30.0,
                half_open_max_calls=1
            )
        )
        
        # Search service circuit breaker (Weaviate)
        self.register_circuit_breaker(
            name="weaviate_search",
            service_type=ServiceType.SEARCH,
            config=CircuitConfig(
                failure_threshold=5,
                timeout=60.0,
                half_open_max_calls=2
            )
        )
        
        # ML model service circuit breaker
        self.register_circuit_breaker(
            name="ml_inference",
            service_type=ServiceType.ML_MODEL,
            config=CircuitConfig(
                failure_threshold=3,
                timeout=120.0,
                half_open_max_calls=1,
                error_timeout=60.0
            )
        )
        
        # Cache service circuit breaker (Redis)
        self.register_circuit_breaker(
            name="redis_cache",
            service_type=ServiceType.CACHE,
            config=CircuitConfig(
                failure_threshold=10,
                timeout=20.0,
                half_open_max_calls=3
            )
        )
    
    def register_circuit_breaker(
        self,
        name: str,
        service_type: ServiceType = ServiceType.REST_API,
        config: Optional[CircuitConfig] = None,
        fallback: Optional[Callable] = None
    ) -> CircuitBreaker:
        """Register a new circuit breaker"""
        with self._lock:
            if name in self.circuit_breakers:
                logger.warning(f"Circuit breaker {name} already exists, replacing")
            
            # Use global fallback if none provided
            if not fallback and service_type in self.fallback_strategies:
                fallback = self.fallback_strategies[service_type]
            
            breaker = CircuitBreaker(
                name=name,
                service_type=service_type,
                config=config,
                fallback=fallback
            )
            
            self.circuit_breakers[name] = breaker
            logger.info(f"Registered circuit breaker: {name}")
            
            return breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def register_global_fallback(self, service_type: ServiceType, fallback: Callable):
        """Register global fallback strategy for service type"""
        self.fallback_strategies[service_type] = fallback
        logger.info(f"Registered global fallback for {service_type.value}")
    
    async def call_with_breaker(
        self,
        breaker_name: str,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute function using named circuit breaker"""
        breaker = self.get_circuit_breaker(breaker_name)
        if not breaker:
            raise ValueError(f"Circuit breaker {breaker_name} not found")
        
        return await breaker.call(func, *args, **kwargs)
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self.circuit_breakers.items()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary across all services"""
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(
            1 for b in self.circuit_breakers.values()
            if b.health.state == CircuitState.OPEN
        )
        half_open_breakers = sum(
            1 for b in self.circuit_breakers.values()
            if b.health.state == CircuitState.HALF_OPEN
        )
        
        # Calculate average metrics
        total_calls = sum(b.health.total_calls for b in self.circuit_breakers.values())
        avg_error_rate = (
            sum(b.health.error_rate * b.health.total_calls for b in self.circuit_breakers.values()) / total_calls
            if total_calls > 0 else 0
        )
        
        return {
            "total_breakers": total_breakers,
            "open_breakers": open_breakers,
            "half_open_breakers": half_open_breakers,
            "closed_breakers": total_breakers - open_breakers - half_open_breakers,
            "total_calls": total_calls,
            "avg_error_rate": avg_error_rate,
            "health_score": ((total_breakers - open_breakers) / total_breakers * 100) if total_breakers > 0 else 100
        }
    
    async def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.circuit_breakers.values():
            await breaker.reset()
    
    async def reset_by_type(self, service_type: ServiceType):
        """Reset circuit breakers by service type"""
        for breaker in self.circuit_breakers.values():
            if breaker.service_type == service_type:
                await breaker.reset()


# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()


# Decorator for circuit breaker protection
def circuit_breaker(
    name: str,
    service_type: ServiceType = ServiceType.REST_API,
    config: Optional[CircuitConfig] = None,
    fallback: Optional[Callable] = None
):
    """Decorator to add circuit breaker protection to functions"""
    def decorator(func):
        # Register circuit breaker if not exists
        breaker = circuit_manager.get_circuit_breaker(name)
        if not breaker:
            breaker = circuit_manager.register_circuit_breaker(
                name=name,
                service_type=service_type,
                config=config,
                fallback=fallback
            )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    breaker.call(func, *args, **kwargs)
                )
            finally:
                loop.close()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Exception classes
class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors"""
    pass


class CircuitBreakerOpen(CircuitBreakerError):
    """Raised when circuit breaker is open"""
    pass


# Example fallback functions
async def cache_fallback(*args, **kwargs):
    """Fallback for cache operations"""
    logger.warning("Cache unavailable, returning None")
    return None


async def search_fallback(query: str, *args, **kwargs):
    """Fallback for search operations"""
    logger.warning(f"Search service unavailable for query: {query}")
    return {"results": [], "error": "Search service temporarily unavailable"}


async def ml_model_fallback(data: Any, *args, **kwargs):
    """Fallback for ML model operations"""
    logger.warning("ML model unavailable, using default response")
    return {"prediction": None, "confidence": 0.0, "fallback": True}


# Register default fallbacks
circuit_manager.register_global_fallback(ServiceType.CACHE, cache_fallback)
circuit_manager.register_global_fallback(ServiceType.SEARCH, search_fallback)
circuit_manager.register_global_fallback(ServiceType.ML_MODEL, ml_model_fallback)