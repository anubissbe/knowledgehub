"""
External Service Client with Circuit Breaker Protection

Provides resilient HTTP client for external service calls with circuit breaker
protection, retry logic, timeout handling, and comprehensive monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import aiohttp
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .circuit_breaker import (
    circuit_breaker, circuit_manager, ServiceType, CircuitConfig,
    CircuitBreakerOpen
)
from ..config import settings

logger = logging.getLogger(__name__)


class ExternalServiceClient:
    """
    Resilient HTTP client for external service calls.
    
    Features:
    - Circuit breaker protection
    - Automatic retry with exponential backoff
    - Connection pooling
    - Request/response logging
    - Performance monitoring
    - Timeout handling
    - Error classification
    """
    
    def __init__(self, base_url: str, service_name: str, 
                 timeout: float = 30.0, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        # HTTP client configuration
        self.session: Optional[aiohttp.ClientSession] = None
        self.httpx_client: Optional[httpx.AsyncClient] = None
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        
        # Register circuit breaker
        self._register_circuit_breaker()
    
    def _register_circuit_breaker(self):
        """Register circuit breaker for this service"""
        circuit_manager.register_circuit_breaker(
            name=self.service_name,
            service_type=ServiceType.REST_API,
            config=CircuitConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=60.0,
                half_open_max_calls=3
            ),
            fallback=self._service_fallback
        )
    
    async def _service_fallback(self, *args, **kwargs):
        """Fallback when service is unavailable"""
        logger.warning(f"Service {self.service_name} unavailable, using fallback")
        return {
            "error": "Service temporarily unavailable",
            "service": self.service_name,
            "fallback": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def start(self):
        """Initialize HTTP clients"""
        if not self.session:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300
            )
            
            timeout_config = aiohttp.ClientTimeout(
                total=self.timeout,
                connect=10.0,
                sock_read=self.timeout
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config,
                headers={
                    "User-Agent": f"KnowledgeHub/{settings.APP_NAME}",
                    "X-Service": self.service_name
                }
            )
        
        if not self.httpx_client:
            self.httpx_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                headers={
                    "User-Agent": f"KnowledgeHub/{settings.APP_NAME}",
                    "X-Service": self.service_name
                }
            )
    
    async def close(self):
        """Close HTTP clients"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.httpx_client:
            await self.httpx_client.aclose()
            self.httpx_client = None
    
    @circuit_breaker("external_api")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, httpx.HTTPError))
    )
    async def get(self, endpoint: str, params: Optional[Dict] = None, 
                  headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute GET request with circuit breaker protection"""
        return await self._request("GET", endpoint, params=params, headers=headers)
    
    @circuit_breaker("external_api")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, httpx.HTTPError))
    )
    async def post(self, endpoint: str, data: Optional[Dict] = None,
                   json_data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute POST request with circuit breaker protection"""
        return await self._request("POST", endpoint, data=data, json=json_data, headers=headers)
    
    @circuit_breaker("external_api")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, httpx.HTTPError))
    )
    async def put(self, endpoint: str, data: Optional[Dict] = None,
                  json_data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute PUT request with circuit breaker protection"""
        return await self._request("PUT", endpoint, data=data, json=json_data, headers=headers)
    
    @circuit_breaker("external_api")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, httpx.HTTPError))
    )
    async def delete(self, endpoint: str, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute DELETE request with circuit breaker protection"""
        return await self._request("DELETE", endpoint, headers=headers)
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Execute HTTP request with monitoring"""
        if not self.session:
            await self.start()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        start_time = time.time()
        
        try:
            self.request_count += 1
            
            logger.debug(f"{method} {url}")
            
            async with self.session.request(method, url, **kwargs) as response:
                response_time = time.time() - start_time
                self.total_response_time += response_time
                
                # Log response
                logger.debug(f"{method} {url} - Status: {response.status} - Time: {response_time:.2f}s")
                
                # Handle different status codes
                if response.status >= 500:
                    # Server error - should trigger circuit breaker
                    self.error_count += 1
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Server error: {error_text}"
                    )
                elif response.status >= 400:
                    # Client error - don't trigger circuit breaker
                    self.error_count += 1
                    error_text = await response.text()
                    return {
                        "error": f"Client error: {response.status}",
                        "message": error_text,
                        "status_code": response.status
                    }
                
                # Success response
                try:
                    return await response.json()
                except Exception:
                    # Return text if not JSON
                    text = await response.text()
                    return {"data": text, "status_code": response.status}
                
        except aiohttp.ClientError as e:
            self.error_count += 1
            logger.error(f"HTTP client error for {url}: {e}")
            raise
        except Exception as e:
            self.error_count += 1
            logger.error(f"Unexpected error for {url}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        avg_response_time = (
            self.total_response_time / self.request_count
            if self.request_count > 0 else 0
        )
        
        error_rate = (
            self.error_count / self.request_count * 100
            if self.request_count > 0 else 0
        )
        
        # Get circuit breaker status
        breaker = circuit_manager.get_circuit_breaker(self.service_name)
        breaker_status = breaker.get_status() if breaker else None
        
        return {
            "service_name": self.service_name,
            "base_url": self.base_url,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "circuit_breaker": breaker_status
        }


class WeaviateServiceClient(ExternalServiceClient):
    """
    Specialized client for Weaviate vector search service.
    """
    
    def __init__(self):
        super().__init__(
            base_url=f"{settings.WEAVIATE_URL}/v1",
            service_name="weaviate_search",
            timeout=30.0
        )
        
        # Override circuit breaker config for search
        circuit_manager.register_circuit_breaker(
            name="weaviate_search",
            service_type=ServiceType.SEARCH,
            config=CircuitConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=60.0
            ),
            fallback=self._search_fallback
        )
    
    async def _search_fallback(self, *args, **kwargs):
        """Fallback for search operations"""
        return {
            "objects": [],
            "error": "Search service temporarily unavailable",
            "fallback": True
        }
    
    @circuit_breaker("weaviate_search")
    async def search_vectors(self, vector: List[float], limit: int = 10,
                           where_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """Search vectors with circuit breaker protection"""
        query = {
            "vector": vector,
            "limit": limit
        }
        
        if where_filter:
            query["where"] = where_filter
        
        return await self.post("/objects", json_data=query)
    
    @circuit_breaker("weaviate_search")
    async def get_object(self, object_id: str) -> Dict[str, Any]:
        """Get object by ID with circuit breaker protection"""
        return await self.get(f"/objects/{object_id}")


class MLModelServiceClient(ExternalServiceClient):
    """
    Specialized client for ML model inference service.
    """
    
    def __init__(self):
        super().__init__(
            base_url=settings.EMBEDDINGS_SERVICE_URL,
            service_name="ml_inference",
            timeout=60.0
        )
        
        # Override circuit breaker config for ML models
        circuit_manager.register_circuit_breaker(
            name="ml_inference",
            service_type=ServiceType.ML_MODEL,
            config=CircuitConfig(
                failure_threshold=3,
                success_threshold=1,
                timeout=120.0,
                error_timeout=60.0
            ),
            fallback=self._ml_fallback
        )
    
    async def _ml_fallback(self, *args, **kwargs):
        """Fallback for ML operations"""
        return {
            "embeddings": None,
            "error": "ML service temporarily unavailable",
            "fallback": True
        }
    
    @circuit_breaker("ml_inference")
    async def generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings with circuit breaker protection"""
        return await self.post("/embeddings", json_data={"texts": texts})
    
    @circuit_breaker("ml_inference")
    async def classify_text(self, text: str, labels: List[str]) -> Dict[str, Any]:
        """Classify text with circuit breaker protection"""
        return await self.post("/classify", json_data={"text": text, "labels": labels})


class RedisServiceClient:
    """
    Specialized client for Redis cache service with circuit breaker.
    """
    
    def __init__(self):
        self.service_name = "redis_cache"
        self._register_circuit_breaker()
    
    def _register_circuit_breaker(self):
        """Register circuit breaker for Redis"""
        circuit_manager.register_circuit_breaker(
            name=self.service_name,
            service_type=ServiceType.CACHE,
            config=CircuitConfig(
                failure_threshold=10,
                success_threshold=3,
                timeout=20.0
            ),
            fallback=self._cache_fallback
        )
    
    async def _cache_fallback(self, *args, **kwargs):
        """Fallback for cache operations"""
        logger.warning("Cache unavailable, using fallback")
        return None
    
    @circuit_breaker("redis_cache")
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with circuit breaker protection"""
        try:
            from ..services.cache import redis_client
            value = await redis_client.get(key)
            return value
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            raise
    
    @circuit_breaker("redis_cache")
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with circuit breaker protection"""
        try:
            from ..services.cache import redis_client
            await redis_client.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            raise
    
    @circuit_breaker("redis_cache")
    async def delete(self, key: str) -> bool:
        """Delete value from cache with circuit breaker protection"""
        try:
            from ..services.cache import redis_client
            result = await redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            raise


# Global service clients
weaviate_client = WeaviateServiceClient()
ml_model_client = MLModelServiceClient()
redis_client = RedisServiceClient()


# Convenience functions
async def search_with_fallback(vector: List[float], limit: int = 10) -> Dict[str, Any]:
    """Search vectors with automatic fallback"""
    return await weaviate_client.search_vectors(vector, limit)


async def generate_embeddings_with_fallback(texts: List[str]) -> Dict[str, Any]:
    """Generate embeddings with automatic fallback"""
    return await ml_model_client.generate_embeddings(texts)


async def get_from_cache_with_fallback(key: str) -> Optional[Any]:
    """Get from cache with automatic fallback"""
    return await redis_client.get(key)