"""
OpenTelemetry Tracing Middleware

FastAPI middleware for comprehensive distributed tracing integration.
Provides automatic span creation, performance analysis, and trace correlation.
"""

import time
import logging
import uuid
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.opentelemetry_tracing import otel_tracing

logger = logging.getLogger(__name__)

class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic OpenTelemetry tracing of HTTP requests.
    
    Features:
    - Automatic span creation for all HTTP requests
    - Performance analysis and slow request detection
    - Request/response metadata tracking
    - Error tracking and exception recording
    - Custom attribute enrichment
    - Trace correlation headers
    """
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ['/metrics', '/health', '/docs', '/redoc', '/openapi.json']
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip tracing for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Extract route pattern for consistent span naming
        route_pattern = self._get_route_pattern(request)
        span_name = f"{request.method} {route_pattern}"
        
        # Start distributed trace
        with otel_tracing.start_span(
            span_name,
            kind="server",
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": route_pattern,
                "http.scheme": request.url.scheme,
                "http.host": request.headers.get("host", "unknown"),
                "http.user_agent": request.headers.get("user-agent", "unknown"),
                "http.request_id": str(uuid.uuid4())
            }
        ) as span:
            
            # Add custom request attributes
            self._enrich_span_with_request_data(span, request)
            
            start_time = time.time()
            
            try:
                # Process request
                response = await call_next(request)
                
                # Calculate request duration
                duration = time.time() - start_time
                duration_ms = duration * 1000
                
                # Add response attributes
                if span:
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.response_size", 
                                     len(response.headers.get("content-length", "0")))
                    span.set_attribute("http.duration_ms", duration_ms)
                    
                    # Performance classification
                    if duration_ms > 1000:  # >1s is slow
                        span.set_attribute("performance.status", "slow")
                        span.add_event("slow_request_detected", {
                            "duration_ms": duration_ms,
                            "threshold_ms": 1000,
                            "route": route_pattern
                        })
                    elif duration_ms > 500:  # >500ms is medium
                        span.set_attribute("performance.status", "medium")
                    else:
                        span.set_attribute("performance.status", "fast")
                    
                    # Mark error responses
                    if response.status_code >= 400:
                        span.set_attribute("error", True)
                        if response.status_code >= 500:
                            span.add_event("server_error", {
                                "status_code": response.status_code,
                                "route": route_pattern
                            })
                
                # Add trace headers to response
                self._add_trace_headers(response)
                
                return response
                
            except Exception as e:
                # Record exception in span
                if span:
                    otel_tracing.record_exception(e)
                    span.add_event("request_exception", {
                        "exception.type": type(e).__name__,
                        "exception.message": str(e),
                        "route": route_pattern
                    })
                
                # Re-raise the exception
                raise e
    
    def _get_route_pattern(self, request: Request) -> str:
        """Extract route pattern from request for consistent span naming"""
        try:
            # Try to get the route pattern from FastAPI
            if hasattr(request, 'scope') and 'route' in request.scope:
                route = request.scope['route']
                if hasattr(route, 'path'):
                    return route.path
            
            # Fallback to path normalization
            path = request.url.path
            
            # Normalize common patterns
            if path.startswith('/api/v1/'):
                return path
            elif path.startswith('/api/'):
                return path
            elif path.startswith('/ws/'):
                return '/ws/{connection}'
            elif path == '/':
                return '/'
            else:
                # Group other paths
                return '/other'
                
        except Exception:
            return request.url.path or '/unknown'
    
    def _enrich_span_with_request_data(self, span, request: Request) -> None:
        """Add custom attributes to span based on request data"""
        if not span:
            return
            
        try:
            # Add query parameters count
            if request.query_params:
                span.set_attribute("http.query_params_count", len(request.query_params))
            
            # Add content type
            content_type = request.headers.get("content-type")
            if content_type:
                span.set_attribute("http.request_content_type", content_type)
            
            # Add request size
            content_length = request.headers.get("content-length")
            if content_length:
                span.set_attribute("http.request_size", int(content_length))
            
            # Add custom headers
            authorization = request.headers.get("authorization")
            if authorization:
                # Don't log the actual token, just that it exists
                span.set_attribute("http.has_authorization", True)
                
                # Extract auth type
                auth_parts = authorization.split(' ', 1)
                if len(auth_parts) > 0:
                    span.set_attribute("http.auth_type", auth_parts[0])
            
            # Add user agent details
            user_agent = request.headers.get("user-agent", "")
            if user_agent:
                if "curl" in user_agent.lower():
                    span.set_attribute("http.client_type", "curl")
                elif "postman" in user_agent.lower():
                    span.set_attribute("http.client_type", "postman")
                elif "python" in user_agent.lower():
                    span.set_attribute("http.client_type", "python")
                elif "browser" in user_agent.lower() or "mozilla" in user_agent.lower():
                    span.set_attribute("http.client_type", "browser")
                else:
                    span.set_attribute("http.client_type", "unknown")
            
            # Add API version if present in path
            if "/v1/" in request.url.path:
                span.set_attribute("api.version", "v1")
            
            # Add request context
            if hasattr(request.state, 'user_id'):
                span.set_attribute("user.id", request.state.user_id)
            
            if hasattr(request.state, 'session_id'):
                span.set_attribute("session.id", request.state.session_id)
                
        except Exception as e:
            logger.warning(f"Failed to enrich span with request data: {e}")
    
    def _add_trace_headers(self, response: Response) -> None:
        """Add trace correlation headers to response"""
        try:
            trace_id = otel_tracing.get_trace_id()
            span_id = otel_tracing.get_span_id()
            
            if trace_id:
                response.headers["X-Trace-ID"] = trace_id
                
            if span_id:
                response.headers["X-Span-ID"] = span_id
                
        except Exception as e:
            logger.warning(f"Failed to add trace headers: {e}")

class DatabaseTracingMiddleware:
    """
    Middleware for tracing database operations.
    Can be used as a decorator or context manager.
    """
    
    @staticmethod
    def trace_query(operation: str, table: Optional[str] = None):
        """Decorator for tracing database queries"""
        return otel_tracing.trace_database_operation(operation, table, "query")
    
    @staticmethod
    def trace_transaction(operation: str):
        """Decorator for tracing database transactions"""
        return otel_tracing.trace_database_operation(operation, None, "transaction")

class AITracingMiddleware:
    """
    Middleware for tracing AI operations.
    Provides specialized tracing for AI model interactions.
    """
    
    @staticmethod
    def trace_embedding(model_name: str):
        """Decorator for tracing embedding generation"""
        return otel_tracing.trace_ai_operation(model_name, "embedding")
    
    @staticmethod
    def trace_inference(model_name: str):
        """Decorator for tracing model inference"""
        return otel_tracing.trace_ai_operation(model_name, "inference")
    
    @staticmethod
    def trace_search(search_type: str = "semantic"):
        """Decorator for tracing search operations"""
        return otel_tracing.trace_memory_operation("search", "system", search_type)

class ExternalServiceTracingMiddleware:
    """
    Middleware for tracing external service calls.
    """
    
    @staticmethod
    def trace_http_call(service_name: str, endpoint: str, method: str = "GET"):
        """Decorator for tracing HTTP calls to external services"""
        return otel_tracing.trace_external_call(service_name, endpoint, method)
    
    @staticmethod
    def trace_redis_operation(operation: str):
        """Decorator for tracing Redis operations"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                with otel_tracing.start_span(
                    f"redis.{operation}",
                    kind="database",
                    attributes={
                        "db.type": "redis",
                        "db.operation": operation,
                        "component": "redis"
                    }
                ):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

# Convenience decorators for common operations
trace_db_query = DatabaseTracingMiddleware.trace_query
trace_db_transaction = DatabaseTracingMiddleware.trace_transaction
trace_ai_embedding = AITracingMiddleware.trace_embedding
trace_ai_inference = AITracingMiddleware.trace_inference
trace_memory_search = AITracingMiddleware.trace_search
trace_http_call = ExternalServiceTracingMiddleware.trace_http_call
trace_redis_op = ExternalServiceTracingMiddleware.trace_redis_operation