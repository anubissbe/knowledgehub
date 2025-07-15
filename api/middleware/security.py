"""Security middleware for adding security headers and content validation"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import logging

from ..security import create_security_headers, validate_content_type

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response"""
        response = await call_next(request)
        
        # Add security headers
        security_headers = create_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class ContentValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate request content types and headers"""
    
    async def dispatch(self, request: Request, call_next):
        """Validate request content and headers"""
        
        # Skip validation for certain paths
        skip_paths = ['/health', '/api/docs', '/api/redoc', '/api/openapi.json', '/ws']
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Validate content type for requests with body
        if request.method in ['POST', 'PUT', 'PATCH']:
            content_type = request.headers.get('content-type', '')
            
            if content_type and not validate_content_type(content_type):
                logger.warning(f"Invalid content type: {content_type} from {request.client.host}")
                return Response(
                    content='{"error": "Invalid content type"}',
                    status_code=400,
                    media_type="application/json"
                )
        
        # Check for suspicious user agents
        user_agent = request.headers.get('user-agent', '')
        suspicious_agents = [
            'sqlmap',
            'nikto',
            'nmap',
            'masscan',
            'w3af',
            'havij',
            'beef',
        ]
        
        if any(agent.lower() in user_agent.lower() for agent in suspicious_agents):
            logger.warning(f"Suspicious user agent: {user_agent} from {request.client.host}")
            return Response(
                content='{"error": "Access denied"}',
                status_code=403,
                media_type="application/json"
            )
        
        # Check for suspicious headers
        suspicious_headers = [
            'x-forwarded-host',
            'x-cluster-client-ip',
            'x-real-ip',
        ]
        
        for header in suspicious_headers:
            if header in request.headers:
                value = request.headers[header]
                # Log for monitoring but don't block
                logger.info(f"Received header {header}: {value} from {request.client.host}")
        
        return await call_next(request)