
from circuitbreaker import circuit
import asyncio
from typing import Any, Callable
import time

class ServiceCircuitBreaker:
    """Circuit breaker for service calls"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: int = 30,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

# Apply to all external services
CIRCUIT_BREAKERS = {
    "zep": ServiceCircuitBreaker(failure_threshold=5, recovery_timeout=30),
    "firecrawl": ServiceCircuitBreaker(failure_threshold=3, recovery_timeout=60),
    "neo4j": ServiceCircuitBreaker(failure_threshold=5, recovery_timeout=30),
    "weaviate": ServiceCircuitBreaker(failure_threshold=5, recovery_timeout=30),
    "phoenix": ServiceCircuitBreaker(failure_threshold=3, recovery_timeout=45)
}
