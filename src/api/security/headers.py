"""
Security Headers and CSRF Protection

Implements comprehensive HTTP security headers and CSRF protection
to prevent various web-based attacks and security vulnerabilities.
"""

import secrets
import hashlib
import time
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class SecurityHeaderLevel(str, Enum):
    """Security header strictness levels"""
    STRICT = "strict"      # Maximum security, may break some functionality
    MODERATE = "moderate"  # Balanced security and compatibility
    PERMISSIVE = "permissive"  # Minimal security, maximum compatibility


@dataclass
class CSRFConfig:
    """CSRF protection configuration"""
    enabled: bool = True
    cookie_name: str = "csrf_token"
    header_name: str = "X-CSRF-Token"
    token_length: int = 32
    token_lifetime: int = 3600  # 1 hour
    same_site: str = "Strict"
    secure: bool = True
    httponly: bool = True
    require_referer: bool = True
    trusted_origins: Set[str] = None
    
    def __post_init__(self):
        if self.trusted_origins is None:
            self.trusted_origins = set()


class SecurityHeadersManager:
    """Manages HTTP security headers and CSRF protection"""
    
    def __init__(self, level: SecurityHeaderLevel = SecurityHeaderLevel.MODERATE,
                 csrf_config: Optional[CSRFConfig] = None):
        self.level = level
        self.csrf_config = csrf_config or CSRFConfig()
        
        # Active CSRF tokens
        self.csrf_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Security headers configuration
        self.security_headers = self._get_security_headers()
        
        # Content Security Policy configuration
        self.csp_config = self._get_csp_config()
        
        # Feature Policy configuration
        self.feature_policy = self._get_feature_policy()
        
        logger.info(f"Security headers manager initialized with {level.value} level")
    
    def _get_security_headers(self) -> Dict[str, str]:
        """Get security headers based on configuration level"""
        
        base_headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Remove server information
            "Server": "KnowledgeHub",
            
            # Prevent caching of sensitive content
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        
        if self.level == SecurityHeaderLevel.STRICT:
            base_headers.update({
                # Strict transport security (HTTPS only)
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
                
                # Expect certificate transparency
                "Expect-CT": "max-age=86400, enforce",
                
                # Cross-origin policies
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Resource-Policy": "same-origin",
                
                # Disable DNS prefetching
                "X-DNS-Prefetch-Control": "off",
                
                # Prevent information disclosure
                "X-Powered-By": "",
            })
        
        elif self.level == SecurityHeaderLevel.MODERATE:
            base_headers.update({
                # Moderate HSTS
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                
                # Relaxed cross-origin policies
                "Cross-Origin-Embedder-Policy": "unsafe-none",
                "Cross-Origin-Opener-Policy": "same-origin-allow-popups",
                "Cross-Origin-Resource-Policy": "cross-origin",
            })
        
        else:  # PERMISSIVE
            base_headers.update({
                # Minimal HSTS
                "Strict-Transport-Security": "max-age=86400",
                
                # No cross-origin restrictions
                "Cross-Origin-Resource-Policy": "cross-origin",
            })
        
        return base_headers
    
    def _get_csp_config(self) -> Dict[str, str]:
        """Get Content Security Policy configuration"""
        
        if self.level == SecurityHeaderLevel.STRICT:
            return {
                "default-src": "'self'",
                "script-src": "'self'",
                "style-src": "'self' 'unsafe-inline'",
                "img-src": "'self' data: https:",
                "font-src": "'self'",
                "connect-src": "'self'",
                "media-src": "'self'",
                "object-src": "'none'",
                "child-src": "'none'",
                "worker-src": "'self'",
                "frame-ancestors": "'none'",
                "form-action": "'self'",
                "base-uri": "'self'",
                "manifest-src": "'self'",
                "upgrade-insecure-requests": "",
                "block-all-mixed-content": ""
            }
        
        elif self.level == SecurityHeaderLevel.MODERATE:
            return {
                "default-src": "'self'",
                "script-src": "'self' 'unsafe-inline'",
                "style-src": "'self' 'unsafe-inline'",
                "img-src": "'self' data: https:",
                "font-src": "'self' https:",
                "connect-src": "'self' https:",
                "media-src": "'self' https:",
                "object-src": "'none'",
                "child-src": "'self'",
                "worker-src": "'self'",
                "frame-ancestors": "'self'",
                "form-action": "'self'",
                "base-uri": "'self'",
                "manifest-src": "'self'"
            }
        
        else:  # PERMISSIVE
            return {
                "default-src": "'self' 'unsafe-inline' 'unsafe-eval'",
                "script-src": "'self' 'unsafe-inline' 'unsafe-eval'",
                "style-src": "'self' 'unsafe-inline'",
                "img-src": "'self' data: https: http:",
                "font-src": "'self' https: http:",
                "connect-src": "'self' https: http: ws: wss:",
                "media-src": "'self' https: http:",
                "object-src": "'self'",
                "child-src": "'self'",
                "worker-src": "'self'",
                "frame-ancestors": "'self'",
                "form-action": "'self'",
                "base-uri": "'self'"
            }
    
    def _get_feature_policy(self) -> Dict[str, str]:
        """Get Feature Policy/Permissions Policy configuration"""
        
        if self.level == SecurityHeaderLevel.STRICT:
            return {
                "accelerometer": "'none'",
                "camera": "'none'",
                "geolocation": "'none'",
                "gyroscope": "'none'",
                "magnetometer": "'none'",
                "microphone": "'none'",
                "payment": "'none'",
                "usb": "'none'",
                "interest-cohort": "()",  # Disable FLoC
                "browsing-topics": "()",  # Disable Topics API
            }
        
        elif self.level == SecurityHeaderLevel.MODERATE:
            return {
                "accelerometer": "'self'",
                "camera": "'none'",
                "geolocation": "'none'",
                "gyroscope": "'self'",
                "magnetometer": "'none'",
                "microphone": "'none'",
                "payment": "'none'",
                "usb": "'none'",
                "interest-cohort": "()",
                "browsing-topics": "()",
            }
        
        else:  # PERMISSIVE
            return {
                "interest-cohort": "()",
                "browsing-topics": "()",
            }
    
    def apply_security_headers(self, response: Response, request: Request) -> Response:
        """Apply security headers to response"""
        
        # Apply basic security headers
        for header, value in self.security_headers.items():
            if value:  # Only set non-empty values
                response.headers[header] = value
        
        # Apply Content Security Policy
        csp_header = self._build_csp_header(request)
        if csp_header:
            response.headers["Content-Security-Policy"] = csp_header
        
        # Apply Feature Policy/Permissions Policy
        feature_policy_header = self._build_feature_policy_header()
        if feature_policy_header:
            response.headers["Permissions-Policy"] = feature_policy_header
        
        # Apply CSRF token if enabled
        if self.csrf_config.enabled:
            self._apply_csrf_headers(response, request)
        
        return response
    
    def _build_csp_header(self, request: Request) -> str:
        """Build Content Security Policy header"""
        csp_parts = []
        
        # Get base CSP configuration
        csp_config = self.csp_config.copy()
        
        # Add nonce for inline scripts if needed
        if hasattr(request.state, 'csp_nonce'):
            script_src = csp_config.get("script-src", "'self'")
            csp_config["script-src"] = f"{script_src} 'nonce-{request.state.csp_nonce}'"
        
        # Build CSP string
        for directive, value in csp_config.items():
            if value:
                csp_parts.append(f"{directive} {value}")
            else:
                csp_parts.append(directive)
        
        return "; ".join(csp_parts)
    
    def _build_feature_policy_header(self) -> str:
        """Build Feature Policy/Permissions Policy header"""
        policy_parts = []
        
        for feature, value in self.feature_policy.items():
            policy_parts.append(f"{feature}={value}")
        
        return ", ".join(policy_parts)
    
    def _apply_csrf_headers(self, response: Response, request: Request):
        """Apply CSRF protection headers"""
        if not self.csrf_config.enabled:
            return
        
        # Generate or retrieve CSRF token
        csrf_token = self._get_or_create_csrf_token(request)
        
        # Set CSRF token in cookie
        response.set_cookie(
            key=self.csrf_config.cookie_name,
            value=csrf_token,
            max_age=self.csrf_config.token_lifetime,
            httponly=self.csrf_config.httponly,
            secure=self.csrf_config.secure,
            samesite=self.csrf_config.same_site
        )
        
        # Add CSRF token to response header for JavaScript access
        response.headers["X-CSRF-Token"] = csrf_token
    
    def generate_csrf_token(self, request: Request) -> str:
        """Generate a new CSRF token"""
        # Create token with timestamp and random data
        timestamp = str(int(time.time()))
        random_data = secrets.token_urlsafe(self.csrf_config.token_length)
        
        # Get session identifier (IP + User-Agent for now)
        session_id = self._get_session_id(request)
        
        # Create token payload
        token_data = f"{timestamp}:{random_data}:{session_id}"
        
        # Generate token hash
        token_hash = hashlib.sha256(token_data.encode()).hexdigest()
        
        # Store token info
        self.csrf_tokens[token_hash] = {
            "timestamp": int(timestamp),
            "session_id": session_id,
            "created_at": datetime.now(),
            "used": False
        }
        
        return token_hash
    
    def _get_or_create_csrf_token(self, request: Request) -> str:
        """Get existing or create new CSRF token"""
        # Check if token exists in cookie
        existing_token = request.cookies.get(self.csrf_config.cookie_name)
        
        if existing_token and self.validate_csrf_token(existing_token, request):
            return existing_token
        
        # Generate new token
        return self.generate_csrf_token(request)
    
    def validate_csrf_token(self, token: str, request: Request, 
                          consume_token: bool = False) -> bool:
        """Validate CSRF token"""
        
        if not self.csrf_config.enabled:
            return True
        
        if not token or token not in self.csrf_tokens:
            return False
        
        token_info = self.csrf_tokens[token]
        
        # Check if token is expired
        if time.time() - token_info["timestamp"] > self.csrf_config.token_lifetime:
            # Remove expired token
            del self.csrf_tokens[token]
            return False
        
        # Check if token was already used (for one-time tokens)
        if token_info.get("used", False):
            return False
        
        # Validate session ID
        current_session_id = self._get_session_id(request)
        if token_info["session_id"] != current_session_id:
            return False
        
        # Mark token as used if consuming
        if consume_token:
            token_info["used"] = True
        
        return True
    
    def require_csrf_token(self, request: Request) -> bool:
        """Check if request requires CSRF token validation"""
        
        if not self.csrf_config.enabled:
            return False
        
        # Only require CSRF for state-changing methods
        if request.method not in ["POST", "PUT", "PATCH", "DELETE"]:
            return False
        
        # Skip CSRF for API endpoints with API key authentication
        if request.headers.get("X-API-Key"):
            return False
        
        # Skip CSRF for certain content types (API calls)
        content_type = request.headers.get("content-type", "").lower()
        if content_type.startswith("application/json") and "X-Requested-With" in request.headers:
            return False
        
        return True
    
    def validate_csrf_request(self, request: Request) -> bool:
        """Validate CSRF token for request"""
        
        if not self.require_csrf_token(request):
            return True
        
        # Get token from header or form data
        token = request.headers.get(self.csrf_config.header_name)
        
        if not token:
            # Try to get from form data (would need to be implemented based on framework)
            token = request.cookies.get(self.csrf_config.cookie_name)
        
        if not token:
            return False
        
        # Validate the token
        is_valid = self.validate_csrf_token(token, request, consume_token=True)
        
        # Additional referer validation if required
        if is_valid and self.csrf_config.require_referer:
            is_valid = self._validate_referer(request)
        
        return is_valid
    
    def _validate_referer(self, request: Request) -> bool:
        """Validate referer header for CSRF protection"""
        
        referer = request.headers.get("referer")
        if not referer:
            return False
        
        # Parse referer and request URLs
        from urllib.parse import urlparse
        
        referer_parsed = urlparse(referer)
        request_host = request.headers.get("host", "")
        
        # Check if referer matches request host
        if referer_parsed.netloc != request_host:
            # Check if referer is in trusted origins
            if referer not in self.csrf_config.trusted_origins:
                return False
        
        return True
    
    def _get_session_id(self, request: Request) -> str:
        """Get session identifier for CSRF token binding"""
        
        # Use IP address and User-Agent as session identifier
        ip_address = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Create session ID hash
        session_data = f"{ip_address}:{user_agent}"
        session_id = hashlib.sha256(session_data.encode()).hexdigest()[:16]
        
        return session_id
    
    def cleanup_expired_tokens(self):
        """Clean up expired CSRF tokens"""
        
        current_time = time.time()
        expired_tokens = []
        
        for token, token_info in self.csrf_tokens.items():
            if current_time - token_info["timestamp"] > self.csrf_config.token_lifetime:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.csrf_tokens[token]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired CSRF tokens")
    
    def get_csrf_stats(self) -> Dict[str, Any]:
        """Get CSRF protection statistics"""
        
        active_tokens = len(self.csrf_tokens)
        expired_tokens = 0
        used_tokens = 0
        
        current_time = time.time()
        
        for token_info in self.csrf_tokens.values():
            if current_time - token_info["timestamp"] > self.csrf_config.token_lifetime:
                expired_tokens += 1
            elif token_info.get("used", False):
                used_tokens += 1
        
        return {
            "csrf_enabled": self.csrf_config.enabled,
            "active_tokens": active_tokens,
            "expired_tokens": expired_tokens,
            "used_tokens": used_tokens,
            "token_lifetime": self.csrf_config.token_lifetime,
            "security_level": self.level.value,
            "trusted_origins": len(self.csrf_config.trusted_origins)
        }
    
    def add_trusted_origin(self, origin: str):
        """Add trusted origin for CSRF protection"""
        self.csrf_config.trusted_origins.add(origin)
        logger.info(f"Added trusted origin for CSRF: {origin}")
    
    def remove_trusted_origin(self, origin: str):
        """Remove trusted origin from CSRF protection"""
        self.csrf_config.trusted_origins.discard(origin)
        logger.info(f"Removed trusted origin for CSRF: {origin}")
    
    def generate_csp_nonce(self) -> str:
        """Generate CSP nonce for inline scripts"""
        return secrets.token_urlsafe(16)


# Global security headers manager
security_headers_manager = SecurityHeadersManager()


# Utility functions
def get_security_headers_for_response(response: Response, request: Request) -> Response:
    """Apply security headers to response"""
    return security_headers_manager.apply_security_headers(response, request)


def validate_csrf_token_for_request(request: Request) -> bool:
    """Validate CSRF token for request"""
    return security_headers_manager.validate_csrf_request(request)


def generate_csrf_token_for_request(request: Request) -> str:
    """Generate CSRF token for request"""
    return security_headers_manager.generate_csrf_token(request)


def cleanup_expired_csrf_tokens():
    """Clean up expired CSRF tokens"""
    security_headers_manager.cleanup_expired_tokens()


def get_csrf_protection_stats() -> Dict[str, Any]:
    """Get CSRF protection statistics"""
    return security_headers_manager.get_csrf_stats()