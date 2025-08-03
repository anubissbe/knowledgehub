"""
Prometheus Metrics Middleware

FastAPI middleware for automatic metrics collection and exposure.
Integrates with the Prometheus metrics service to track all HTTP requests.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.prometheus_metrics import prometheus_metrics

logger = logging.getLogger(__name__)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically collect Prometheus metrics for all HTTP requests.
    
    Tracks:
    - Request count by method, endpoint, and status code
    - Request duration histograms
    - Active request gauge
    - Error rates by endpoint
    """
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ['/metrics', '/health']
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics collection for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Handle metrics endpoint
        if request.url.path == '/metrics':
            return PlainTextResponse(
                prometheus_metrics.get_metrics(),
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract route pattern for consistent labeling
            endpoint = self._get_route_pattern(request)
            method = request.method
            status_code = response.status_code
            
            # Record metrics
            prometheus_metrics.record_http_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )
            
            # Record error if status code indicates error
            if status_code >= 400:
                error_type = "client_error" if status_code < 500 else "server_error"
                prometheus_metrics.record_error(
                    error_type=error_type,
                    component="api"
                )
            
            return response
            
        except Exception as e:
            # Record exception
            duration = time.time() - start_time
            
            endpoint = self._get_route_pattern(request)
            method = request.method
            
            # Record failed request
            prometheus_metrics.record_http_request(
                method=method,
                endpoint=endpoint,
                status_code=500,
                duration=duration
            )
            
            # Record error
            prometheus_metrics.record_error(
                error_type="exception",
                component="api"
            )
            
            # Re-raise the exception
            raise e
    
    def _get_route_pattern(self, request: Request) -> str:
        """Extract route pattern from request for consistent metric labeling"""
        try:
            # Try to get the route pattern from FastAPI
            if hasattr(request, 'scope') and 'route' in request.scope:
                route = request.scope['route']
                if hasattr(route, 'path'):
                    return route.path
            
            # Fallback to path
            path = request.url.path
            
            # Normalize common patterns
            if path.startswith('/api/'):
                # Keep API paths as-is for detailed monitoring
                return path
            elif path.startswith('/static/'):
                return '/static/*'
            elif path == '/':
                return '/'
            else:
                # Group other paths
                return '/other'
                
        except Exception:
            return request.url.path or '/unknown'