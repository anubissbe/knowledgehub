"""
Resilience Patterns Implementation
Circuit breakers, health checks, and fallback mechanisms for all services
"""

import asyncio
import time
import random
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import logging
from dataclasses import dataclass
import httpx
import redis

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 30
    expected_exceptions: tuple = (Exception,)
    success_threshold: int = 2
    timeout: float = 10.0


class CircuitBreaker:
    """
    Advanced circuit breaker implementation with adaptive thresholds
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "circuit_opens": 0
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.metrics["total_calls"] += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                self.metrics["failed_calls"] += 1
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        # Execute function with timeout
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            self._on_success()
            return result
            
        except self.config.expected_exceptions as e:
            self._on_failure()
            raise e
        except asyncio.TimeoutError:
            self._on_failure()
            raise Exception(f"Circuit breaker '{self.name}' timeout after {self.config.timeout}s")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return False
        
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.metrics["successful_calls"] += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' is now CLOSED")
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call"""
        self.metrics["failed_calls"] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.metrics["circuit_opens"] += 1
            logger.warning(f"Circuit breaker '{self.name}' is now OPEN (HALF_OPEN test failed)")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.metrics["circuit_opens"] += 1
            logger.warning(f"Circuit breaker '{self.name}' is now OPEN (threshold reached)")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "metrics": self.metrics,
            "failure_count": self.failure_count,
            "success_rate": self.metrics["successful_calls"] / max(self.metrics["total_calls"], 1)
        }


class ServiceHealthChecker:
    """
    Advanced health checking with predictive failure detection
    """
    
    def __init__(self, service_name: str, health_endpoint: str):
        self.service_name = service_name
        self.health_endpoint = health_endpoint
        self.check_interval = 30  # seconds
        self.history = []
        self.max_history = 100
        self.client = httpx.AsyncClient(timeout=5.0)
        
    async def check_health(self) -> Dict[str, Any]:
        """Perform health check"""
        start_time = time.time()
        
        try:
            response = await self.client.get(self.health_endpoint)
            latency = (time.time() - start_time) * 1000
            
            health_status = {
                "service": self.service_name,
                "timestamp": datetime.now().isoformat(),
                "status": "healthy" if response.status_code == 200 else "degraded",
                "latency_ms": latency,
                "status_code": response.status_code
            }
            
            # Parse detailed health info if available
            if response.status_code == 200:
                try:
                    data = response.json()
                    health_status["details"] = data
                except:
                    pass
            
        except Exception as e:
            health_status = {
                "service": self.service_name,
                "timestamp": datetime.now().isoformat(),
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000
            }
        
        # Add to history
        self.history.append(health_status)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Predict failure
        health_status["predicted_failure"] = self._predict_failure()
        
        return health_status
    
    def _predict_failure(self) -> bool:
        """Predict potential failure based on historical data"""
        if len(self.history) < 10:
            return False
        
        # Check recent failure rate
        recent = self.history[-10:]
        unhealthy_count = sum(1 for h in recent if h["status"] != "healthy")
        
        # Check latency trend
        latencies = [h.get("latency_ms", 0) for h in recent if "latency_ms" in h]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            if avg_latency > 1000:  # 1 second average
                return True
        
        return unhealthy_count >= 3  # 30% failure rate
    
    async def continuous_monitoring(self):
        """Run continuous health monitoring"""
        while True:
            health = await self.check_health()
            
            if health["status"] == "unhealthy" or health.get("predicted_failure"):
                logger.warning(f"Service '{self.service_name}' health alert: {health}")
            
            await asyncio.sleep(self.check_interval)


class FallbackHandler:
    """
    Intelligent fallback mechanism with multiple strategies
    """
    
    def __init__(self, primary_service: str):
        self.primary_service = primary_service
        self.fallback_strategies = []
        self.cache = {}
        self.metrics = {
            "fallback_invocations": 0,
            "cache_hits": 0,
            "successful_recoveries": 0
        }
    
    def add_fallback(self, strategy: Callable, priority: int = 0):
        """Add fallback strategy"""
        self.fallback_strategies.append((priority, strategy))
        self.fallback_strategies.sort(key=lambda x: x[0])
    
    async def execute_with_fallback(self, primary_func: Callable, 
                                   *args, **kwargs) -> Any:
        """Execute function with fallback strategies"""
        # Try primary function
        try:
            result = await primary_func(*args, **kwargs)
            # Cache successful result
            cache_key = self._generate_cache_key(args, kwargs)
            self.cache[cache_key] = (result, time.time())
            return result
            
        except Exception as primary_error:
            logger.warning(f"Primary service '{self.primary_service}' failed: {primary_error}")
            self.metrics["fallback_invocations"] += 1
            
            # Try cache first
            cache_key = self._generate_cache_key(args, kwargs)
            if cache_key in self.cache:
                cached_result, cached_time = self.cache[cache_key]
                if time.time() - cached_time < 300:  # 5 minute cache
                    self.metrics["cache_hits"] += 1
                    logger.info(f"Using cached result for '{self.primary_service}'")
                    return cached_result
            
            # Try fallback strategies in order
            for priority, strategy in self.fallback_strategies:
                try:
                    result = await strategy(*args, **kwargs)
                    self.metrics["successful_recoveries"] += 1
                    logger.info(f"Fallback strategy succeeded for '{self.primary_service}'")
                    return result
                except Exception as fallback_error:
                    logger.warning(f"Fallback strategy failed: {fallback_error}")
                    continue
            
            # All strategies failed
            raise Exception(f"All fallback strategies failed for '{self.primary_service}'")
    
    def _generate_cache_key(self, args, kwargs) -> str:
        """Generate cache key from arguments"""
        import hashlib
        import json
        
        key_data = {"args": str(args), "kwargs": str(kwargs)}
        return hashlib.md5(json.dumps(key_data).encode()).hexdigest()


class RetryPolicy:
    """
    Advanced retry policy with exponential backoff and jitter
    """
    
    def __init__(self, max_retries: int = 3, 
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    # Add jitter
                    delay = delay * (0.5 + random.random())
                    
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
        
        raise last_exception


class RateLimiter:
    """
    Token bucket rate limiter for service protection
    """
    
    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
    
    async def acquire(self):
        """Acquire permission to proceed"""
        current = time.time()
        time_passed = current - self.last_check
        self.last_check = current
        
        self.allowance += time_passed * (self.rate / self.per)
        
        if self.allowance > self.rate:
            self.allowance = self.rate
        
        if self.allowance < 1.0:
            sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
            await asyncio.sleep(sleep_time)
            self.allowance = 0.0
        else:
            self.allowance -= 1.0


class ResilienceOrchestrator:
    """
    Orchestrates all resilience patterns for the system
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.circuit_breakers = {}
        self.health_checkers = {}
        self.fallback_handlers = {}
        self.retry_policies = {}
        self.rate_limiters = {}
        
        self._initialize_resilience_patterns()
    
    def _initialize_resilience_patterns(self):
        """Initialize resilience patterns for all services"""
        
        # Define service configurations
        services = {
            "zep": {
                "health_endpoint": "http://localhost:8100/health",
                "circuit_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30)
            },
            "firecrawl": {
                "health_endpoint": "http://localhost:3002/health",
                "circuit_config": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
            },
            "neo4j": {
                "health_endpoint": "http://localhost:7474/db/data/",
                "circuit_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30)
            },
            "weaviate": {
                "health_endpoint": "http://192.168.1.25:8090/v1/.well-known/ready",
                "circuit_config": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30)
            },
            "phoenix": {
                "health_endpoint": "http://localhost:6006/health",
                "circuit_config": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=45)
            }
        }
        
        # Initialize patterns for each service
        for service_name, config in services.items():
            # Circuit breaker
            self.circuit_breakers[service_name] = CircuitBreaker(
                service_name, 
                config["circuit_config"]
            )
            
            # Health checker
            self.health_checkers[service_name] = ServiceHealthChecker(
                service_name,
                config["health_endpoint"]
            )
            
            # Fallback handler
            self.fallback_handlers[service_name] = FallbackHandler(service_name)
            
            # Retry policy
            self.retry_policies[service_name] = RetryPolicy()
            
            # Rate limiter (100 requests per second)
            self.rate_limiters[service_name] = RateLimiter(rate=100, per=1.0)
        
        logger.info(f"Initialized resilience patterns for {len(services)} services")
    
    async def protected_service_call(self, service_name: str, 
                                    func: Callable, *args, **kwargs) -> Any:
        """Execute service call with full resilience protection"""
        
        # Rate limiting
        if service_name in self.rate_limiters:
            await self.rate_limiters[service_name].acquire()
        
        # Circuit breaker + Retry + Fallback
        circuit_breaker = self.circuit_breakers.get(service_name)
        retry_policy = self.retry_policies.get(service_name)
        fallback_handler = self.fallback_handlers.get(service_name)
        
        async def protected_func(*args, **kwargs):
            if circuit_breaker:
                return await circuit_breaker.call(func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        if retry_policy:
            protected_func = lambda *a, **k: retry_policy.execute_with_retry(
                protected_func, *a, **k
            )
        
        if fallback_handler:
            return await fallback_handler.execute_with_fallback(
                protected_func, *args, **kwargs
            )
        
        return await protected_func(*args, **kwargs)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "circuit_breakers": {},
            "overall_status": "healthy"
        }
        
        # Check all services
        for service_name, checker in self.health_checkers.items():
            health = await checker.check_health()
            health_status["services"][service_name] = health
            
            if health["status"] != "healthy":
                health_status["overall_status"] = "degraded"
        
        # Get circuit breaker metrics
        for service_name, breaker in self.circuit_breakers.items():
            health_status["circuit_breakers"][service_name] = breaker.get_metrics()
        
        # Store in Redis for monitoring
        self.redis_client.setex(
            "system_health",
            60,
            json.dumps(health_status, default=str)
        )
        
        return health_status
    
    async def start_continuous_monitoring(self):
        """Start continuous health monitoring for all services"""
        tasks = []
        
        for service_name, checker in self.health_checkers.items():
            task = asyncio.create_task(checker.continuous_monitoring())
            tasks.append(task)
        
        # Also monitor overall system health
        async def monitor_system():
            while True:
                health = await self.get_system_health()
                if health["overall_status"] != "healthy":
                    logger.warning(f"System health degraded: {health}")
                await asyncio.sleep(60)
        
        tasks.append(asyncio.create_task(monitor_system()))
        
        await asyncio.gather(*tasks)


# Decorator for easy resilience application
def with_resilience(service_name: str):
    """Decorator to apply resilience patterns to functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get orchestrator from context (would be injected in real app)
            orchestrator = getattr(wrapper, '_orchestrator', None)
            if orchestrator:
                return await orchestrator.protected_service_call(
                    service_name, func, *args, **kwargs
                )
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


async def demo_resilience_patterns():
    """Demonstrate resilience patterns in action"""
    import redis
    
    redis_client = redis.Redis(host='localhost', port=6381)
    orchestrator = ResilienceOrchestrator(redis_client)
    
    print("=" * 60)
    print("🛡️ Resilience Patterns Demo")
    print("=" * 60)
    
    # Simulate service calls
    async def flaky_service():
        """Simulates a flaky service"""
        if random.random() < 0.3:  # 30% failure rate
            raise Exception("Service temporarily unavailable")
        return {"data": "Success!", "timestamp": time.time()}
    
    # Add fallback strategy
    async def fallback_cache():
        return {"data": "Cached response", "from_cache": True}
    
    orchestrator.fallback_handlers["test_service"] = FallbackHandler("test_service")
    orchestrator.fallback_handlers["test_service"].add_fallback(fallback_cache)
    
    # Test with resilience
    print("\n📊 Testing service with resilience patterns:")
    for i in range(10):
        try:
            result = await orchestrator.protected_service_call(
                "test_service",
                flaky_service
            )
            print(f"  Call {i+1}: {result}")
        except Exception as e:
            print(f"  Call {i+1}: Failed - {e}")
        
        await asyncio.sleep(0.5)
    
    # Get system health
    health = await orchestrator.get_system_health()
    print("\n🏥 System Health:")
    print(f"  Overall Status: {health['overall_status']}")
    print(f"  Services Monitored: {len(health['services'])}")
    
    print("\n✅ Resilience patterns demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_resilience_patterns())