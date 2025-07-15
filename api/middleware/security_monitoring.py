"""
Security Monitoring Middleware

Integrates security monitoring into the FastAPI request/response cycle,
automatically detecting and logging security events.
"""

import time
import re
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi.responses import JSONResponse
import logging

from ..security.monitoring import (
    security_monitor,
    SecurityEventType,
    ThreatLevel,
    log_security_event,
    log_suspicious_request,
    log_rate_limit_exceeded
)

logger = logging.getLogger(__name__)


class SecurityMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic security monitoring and threat detection
    
    Features:
    - Request/response monitoring
    - Malicious payload detection
    - IP-based threat analysis
    - Real-time blocking of dangerous requests
    - Integration with security monitoring system
    """
    
    def __init__(self, app, environment: str = "development"):
        super().__init__(app)
        self.environment = environment
        
        # Security patterns to detect in requests
        self.injection_patterns = [
            # SQL Injection
            r"(\bUNION\b.*\bSELECT\b|\bSELECT\b.*\bFROM\b.*\bWHERE\b)",
            r"(\bDROP\b.*\bTABLE\b|\bDELETE\b.*\bFROM\b|\bINSERT\b.*\bINTO\b)",
            r"(\'\s*OR\s*\'\d+\'\s*=\s*\'\d+|\'\s*OR\s*\d+\s*=\s*\d+)",
            r"(\bexec\s*\(|\beval\s*\(|script\s*:)",
            
            # XSS Attempts
            r"(<script[^>]*>.*?</script>|javascript\s*:|vbscript\s*:)",
            r"(onload\s*=|onerror\s*=|onclick\s*=|onmouseover\s*=)",
            r"(<iframe[^>]*>|<object[^>]*>|<embed[^>]*>)",
            
            # Command Injection
            r"(\|\s*nc\s+|\|\s*netcat\s+|\|\s*telnet\s+)",
            r"(\$\(.*\)|`.*`|;.*whoami|;.*id|;.*passwd)",
            r"(\.\.\/.*\/etc\/|\.\.\\.*\\windows\\)",
            
            # Path Traversal
            r"(\.\./|\.\.\x5c|\.\.%2f|\.\.%5c)",
            r"(%2e%2e%2f|%2e%2e%5c|\.\.\x2f|\.\.\x5c)",
            
            # LDAP Injection
            r"(\)\(\||\)\(&|\*\)\(|=\*\)|>\*\))",
            
            # NoSQL Injection
            r"(\$ne\s*:|[\{\}]\s*\$|regex\s*:|javascript\s*:)",
            
            # Server-Side Template Injection
            r"(\{\{.*\}\}|\{\%.*\%\}|\$\{.*\})",
            
            # XML/XXE
            r"(<!ENTITY|<!DOCTYPE.*ENTITY|SYSTEM\s+[\'\"]file\:|PUBLIC\s+[\'\"])",
        ]
        
        # Compile patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns]
        
        # Suspicious file extensions and paths
        # NOTE: Exclude legitimate API endpoints from suspicious paths
        self.suspicious_paths = [
            r"\.php$", r"\.asp$", r"\.jsp$", r"\.cgi$",
            # r"/admin", # Commented out - we have legitimate admin endpoints
            r"/login\.php", r"/config\.php", r"/backup\.sql",
            r"\.git/", r"\.env", r"\.sql$", r"\.bak$",
            r"/etc/passwd", r"/windows/system32", r"\.htaccess"
        ]
        
        self.compiled_path_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.suspicious_paths]
        
        # Rate limiting tracking
        self.request_counts = {}
        self.blocked_requests = {}
        
        logger.info(f"Security monitoring middleware initialized for {environment} environment")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with security monitoring"""
        # Skip security monitoring for internal endpoints (trusted services)
        if str(request.url.path).startswith('/api/internal/'):
            return await call_next(request)
        
        # Skip security monitoring for performance endpoints to avoid timeouts
        if str(request.url.path).startswith('/api/performance/'):
            return await call_next(request)
        
        # Skip security monitoring for memory endpoints to avoid timeouts
        if str(request.url.path).startswith('/api/memory/') or str(request.url.path).startswith('/api/v1/memories/'):
            return await call_next(request)
        
        # Skip security monitoring for search endpoints to avoid timeouts
        if str(request.url.path).startswith('/api/v1/search'):
            return await call_next(request)
        
        # Skip security monitoring for Claude endpoints
        if str(request.url.path).startswith('/api/claude'):
            return await call_next(request)
        
        # Skip security monitoring for decision reasoning endpoints  
        if str(request.url.path).startswith('/api/decisions'):
            return await call_next(request)
        
        # Skip security monitoring for code evolution endpoints
        if str(request.url.path).startswith('/api/code-evolution'):
            return await call_next(request)
        
        # Skip security monitoring for performance metrics endpoints
        if str(request.url.path).startswith('/api/performance'):
            return await call_next(request)
            
        start_time = time.time()
        source_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Check if IP is blocked
        if security_monitor.is_ip_blocked(source_ip):
            logger.warning(f"Blocked request from banned IP: {source_ip}")
            return self._create_blocked_response("IP address is blocked")
        
        # Pre-request security checks
        security_result = await self._perform_security_checks(request, source_ip, user_agent)
        if security_result:
            return security_result
        
        # Process request
        try:
            response = await call_next(request)
            
            # Post-request analysis
            await self._analyze_response(request, response, source_ip, user_agent, start_time)
            
            return response
            
        except Exception as e:
            # Log potential security-related errors
            await log_security_event(
                SecurityEventType.MALFORMED_REQUEST,
                ThreatLevel.MEDIUM,
                source_ip,
                user_agent,
                str(request.url.path),
                request.method,
                f"Request processing error: {str(e)}"
            )
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check X-Forwarded-For header first (for proxied requests)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP from the chain
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"
    
    async def _perform_security_checks(self, request: Request, source_ip: str, user_agent: str) -> Optional[Response]:
        """Perform comprehensive security checks on the request"""
        
        # Check for suspicious paths
        if await self._check_suspicious_paths(request, source_ip, user_agent):
            return self._create_blocked_response("Suspicious path detected")
        
        # Check for injection attempts in URL
        if await self._check_url_injection(request, source_ip, user_agent):
            return self._create_blocked_response("Injection attempt detected")
        
        # Check for malicious user agents
        if await self._check_malicious_user_agent(request, source_ip, user_agent):
            return self._create_blocked_response("Malicious user agent detected")
        
        # Check request headers for anomalies
        if await self._check_suspicious_headers(request, source_ip, user_agent):
            return self._create_blocked_response("Suspicious headers detected")
        
        # For POST/PUT requests, check body content
        if request.method in ["POST", "PUT", "PATCH"]:
            # Skip body checks for performance-critical endpoints in development
            if self.environment == "development" and str(request.url.path).startswith('/api/v1/jobs/'):
                # In development, skip expensive body checks for job endpoints
                pass
            elif await self._check_request_body(request, source_ip, user_agent):
                return self._create_blocked_response("Malicious payload detected")
        
        return None
    
    async def _check_suspicious_paths(self, request: Request, source_ip: str, user_agent: str) -> bool:
        """Check for suspicious path patterns"""
        path = str(request.url.path)
        
        for pattern in self.compiled_path_patterns:
            if pattern.search(path):
                await log_suspicious_request(
                    source_ip, user_agent, path,
                    f"Suspicious path pattern: {pattern.pattern}"
                )
                return True
        
        # Check for directory traversal attempts
        if "../" in path or "..\\" in path or "%2e%2e" in path.lower():
            await log_suspicious_request(
                source_ip, user_agent, path,
                "Directory traversal attempt"
            )
            return True
        
        return False
    
    async def _check_url_injection(self, request: Request, source_ip: str, user_agent: str) -> bool:
        """Check URL parameters for injection attempts"""
        url_str = str(request.url)
        
        for pattern in self.compiled_patterns:
            if pattern.search(url_str):
                await log_security_event(
                    SecurityEventType.INJECTION_ATTEMPT,
                    ThreatLevel.HIGH,
                    source_ip,
                    user_agent,
                    str(request.url.path),
                    request.method,
                    f"URL injection pattern detected: {pattern.pattern}",
                    blocked=True
                )
                return True
        
        return False
    
    async def _check_malicious_user_agent(self, request: Request, source_ip: str, user_agent: str) -> bool:
        """Check for known malicious user agents"""
        user_agent_lower = user_agent.lower()
        
        # Known attack tools and scanners
        malicious_agents = [
            "sqlmap", "nikto", "nmap", "dirbuster", "gobuster", "wfuzz",
            "burp", "nessus", "openvas", "acunetix", "metasploit", "hydra",
            "masscan", "zgrab", "curl/7.1", "python-requests/", "python-urllib",
            "scanner", "bot", "crawler", "spider"
        ]
        
        for agent in malicious_agents:
            if agent in user_agent_lower:
                await log_security_event(
                    SecurityEventType.SECURITY_SCAN,
                    ThreatLevel.HIGH,
                    source_ip,
                    user_agent,
                    str(request.url.path),
                    request.method,
                    f"Malicious user agent detected: {agent}",
                    blocked=True
                )
                return True
        
        # Check for empty or suspicious user agents
        if not user_agent or len(user_agent) < 10:
            await log_suspicious_request(
                source_ip, user_agent, str(request.url.path),
                "Empty or too short user agent"
            )
            return True
        
        return False
    
    async def _check_suspicious_headers(self, request: Request, source_ip: str, user_agent: str) -> bool:
        """Check for suspicious request headers"""
        headers = dict(request.headers)
        
        # Check for header injection attempts
        for header_name, header_value in headers.items():
            if any(char in header_value for char in ['\r', '\n', '\0']):
                await log_security_event(
                    SecurityEventType.INJECTION_ATTEMPT,
                    ThreatLevel.HIGH,
                    source_ip,
                    user_agent,
                    str(request.url.path),
                    request.method,
                    f"Header injection attempt in {header_name}",
                    blocked=True
                )
                return True
        
        # Check for abnormally long headers (potential buffer overflow)
        for header_name, header_value in headers.items():
            if len(header_value) > 8192:  # 8KB limit
                await log_suspicious_request(
                    source_ip, user_agent, str(request.url.path),
                    f"Abnormally long header: {header_name}"
                )
                return True
        
        # Check for suspicious authorization headers
        auth_header = headers.get("authorization", "")
        if auth_header and any(pattern.search(auth_header) for pattern in self.compiled_patterns):
            await log_security_event(
                SecurityEventType.INJECTION_ATTEMPT,
                ThreatLevel.HIGH,
                source_ip,
                user_agent,
                str(request.url.path),
                request.method,
                "Injection attempt in authorization header",
                blocked=True
            )
            return True
        
        return False
    
    async def _check_request_body(self, request: Request, source_ip: str, user_agent: str) -> bool:
        """Check request body for malicious content"""
        # Skip body checks for certain safe endpoints to improve performance
        safe_endpoints = [
            '/api/v1/jobs/',  # Job operations are internally validated
            '/api/v1/sources/',  # Source operations have their own validation
            '/api/v1/search',  # Search queries are sanitized separately
            '/api/v1/memories/',  # Memory operations are validated
            '/ws/',  # WebSocket connections
            '/api/persistent-context/',  # Persistent context operations
            '/api/memory/',  # Memory system operations
            '/api/v1/admin/',  # Admin operations are internally validated
            '/api/v1/diagrams/',  # Diagram operations are internally validated
        ]
        
        path = str(request.url.path)
        if any(path.startswith(endpoint) for endpoint in safe_endpoints):
            # Only check payload size for safe endpoints
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                await log_security_event(
                    SecurityEventType.DOS_ATTEMPT,
                    ThreatLevel.MEDIUM,
                    source_ip,
                    user_agent,
                    path,
                    request.method,
                    f"Large payload detected: {content_length} bytes",
                    blocked=True
                )
                return True
            return False
        
        try:
            # For other endpoints, perform full security checks
            # Read body content
            body = await request.body()
            if not body:
                return False
            
            # Limit body size for pattern checking to prevent performance issues
            max_check_size = 100 * 1024  # 100KB
            body_str = body[:max_check_size].decode('utf-8', errors='ignore')
            
            # Check for injection patterns in body (limited to first 100KB)
            for pattern in self.compiled_patterns:
                if pattern.search(body_str):
                    await log_security_event(
                        SecurityEventType.INJECTION_ATTEMPT,
                        ThreatLevel.HIGH,
                        source_ip,
                        user_agent,
                        path,
                        request.method,
                        f"Injection pattern in request body: {pattern.pattern}",
                        blocked=True
                    )
                    return True
            
            # Check for abnormally large payloads (potential DoS)
            if len(body) > 10 * 1024 * 1024:  # 10MB limit
                await log_security_event(
                    SecurityEventType.DOS_ATTEMPT,
                    ThreatLevel.MEDIUM,
                    source_ip,
                    user_agent,
                    path,
                    request.method,
                    f"Large payload detected: {len(body)} bytes",
                    blocked=True
                )
                return True
            
        except Exception as e:
            logger.warning(f"Error checking request body: {e}")
            # Don't block on parsing errors, but log for investigation
            await log_suspicious_request(
                source_ip, user_agent, path,
                f"Request body parsing error: {str(e)}"
            )
        
        return False
    
    async def _analyze_response(self, request: Request, response: Response, 
                               source_ip: str, user_agent: str, start_time: float) -> None:
        """Analyze response for security insights"""
        processing_time = time.time() - start_time
        
        # Log successful authentication
        if (response.status_code == 200 and 
            str(request.url.path).endswith(('/login', '/auth', '/token'))):
            await log_security_event(
                SecurityEventType.AUTHENTICATION_SUCCESS,
                ThreatLevel.LOW,
                source_ip,
                user_agent,
                str(request.url.path),
                request.method,
                "Successful authentication"
            )
        
        # Log authorization failures
        if response.status_code == 403:
            await log_security_event(
                SecurityEventType.AUTHORIZATION_FAILURE,
                ThreatLevel.MEDIUM,
                source_ip,
                user_agent,
                str(request.url.path),
                request.method,
                "Authorization failed"
            )
        
        # Log authentication failures
        if response.status_code == 401:
            await log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                ThreatLevel.MEDIUM,
                source_ip,
                user_agent,
                str(request.url.path),
                request.method,
                "Authentication failed"
            )
        
        # Log rate limiting
        if response.status_code == 429:
            await log_rate_limit_exceeded(source_ip, user_agent, str(request.url.path))
        
        # Log abnormally slow requests (potential DoS)
        if processing_time > 30.0:  # 30 second threshold
            await log_security_event(
                SecurityEventType.DOS_ATTEMPT,
                ThreatLevel.LOW,
                source_ip,
                user_agent,
                str(request.url.path),
                request.method,
                f"Slow request: {processing_time:.2f}s processing time"
            )
    
    def _create_blocked_response(self, reason: str) -> JSONResponse:
        """Create a standardized blocked response"""
        return JSONResponse(
            status_code=403,
            content={
                "error": "Request blocked",
                "message": reason,
                "code": "SECURITY_VIOLATION"
            }
        )