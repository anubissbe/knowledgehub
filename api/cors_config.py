"""
Secure CORS Configuration for KnowledgeHub

This module provides secure CORS settings that follow security best practices
while maintaining functionality for legitimate cross-origin requests.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import logging

logger = logging.getLogger(__name__)


class CORSSecurityConfig(BaseModel):
    """Secure CORS configuration with strict security settings"""
    
    # Allowed HTTP methods (restricted to necessary methods only)
    allowed_methods: List[str] = [
        "GET",      # Read operations
        "POST",     # Create operations  
        "PUT",      # Update operations
        "PATCH",    # Partial updates
        "DELETE",   # Delete operations
        "OPTIONS"   # Preflight requests (required for CORS)
    ]
    
    # Allowed headers (restricted to necessary headers only)
    allowed_headers: List[str] = [
        # Standard headers
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        
        # Authentication headers
        "Authorization",
        "X-API-Key",
        "X-Requested-With",
        
        # Custom application headers
        "X-Claude-Session-ID",
        "X-Session-ID",
        "X-User-Agent",
        
        # Security headers
        "X-CSRF-Token",
        "X-Frame-Options",
        
        # Cache control
        "Cache-Control",
        "Pragma"
    ]
    
    # Headers exposed to the client
    exposed_headers: List[str] = [
        "Content-Length",
        "Content-Type", 
        "X-Rate-Limit-Remaining",
        "X-Rate-Limit-Reset",
        "X-Total-Count",
        "X-Page-Count"
    ]
    
    # Maximum age for preflight cache (24 hours)
    max_age: int = 86400
    
    # Development vs production origins
    development_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3100",  # KnowledgeHub frontend
        "http://localhost:3102",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3100",  # KnowledgeHub frontend
        "http://127.0.0.1:3102",
        "http://127.0.0.1:5173"
    ]
    
    production_origins: List[str] = [
        "https://knowledgehub.example.com",
        "https://api.knowledgehub.example.com",
        "https://app.knowledgehub.example.com"
    ]
    
    # Local network origins (for internal deployments)
    local_network_origins: List[str] = [
        "http://192.168.1.25:3000",  # Backend API
        "http://192.168.1.25:3100",  # Main frontend
        "http://192.168.1.25:3101",  # Frontend (port auto-selected)
        "http://192.168.1.25:3102",  # Frontend dev server
        "http://192.168.1.25:5173",  # Vite dev server
        "http://192.168.1.24:3000",  # Synology NAS
        "http://192.168.1.24:5174",  # ProjectHub UI
        "http://192.168.1.24:8090"   # ProjectHub frontend
    ]
    
    # HTTPS development origins (for local HTTPS testing)
    https_development_origins: List[str] = [
        "https://localhost:8443",
        "https://api.localhost:8443",
        "https://127.0.0.1:8443"
    ]


def get_cors_origins(environment: str = "development") -> List[str]:
    """
    Get appropriate CORS origins based on environment
    
    Args:
        environment: Environment name (development, production, staging)
        
    Returns:
        List of allowed CORS origins
    """
    config = CORSSecurityConfig()
    origins = []
    
    if environment.lower() == "production":
        origins.extend(config.production_origins)
        # In production, also allow local network for admin access
        origins.extend(config.local_network_origins)
        logger.info(f"Using production CORS origins: {len(origins)} origins")
        
    elif environment.lower() == "staging":
        origins.extend(config.production_origins)
        origins.extend(config.development_origins) 
        origins.extend(config.local_network_origins)
        logger.info(f"Using staging CORS origins: {len(origins)} origins")
        
    else:  # development
        origins.extend(config.development_origins)
        origins.extend(config.local_network_origins)
        origins.extend(config.https_development_origins)
        logger.info(f"Using development CORS origins: {len(origins)} origins")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_origins = []
    for origin in origins:
        if origin not in seen:
            seen.add(origin)
            unique_origins.append(origin)
    
    return unique_origins


def get_cors_config(environment: str = "development", 
                   allow_credentials: bool = True) -> Dict[str, Any]:
    """
    Get complete secure CORS configuration
    
    Args:
        environment: Environment name
        allow_credentials: Whether to allow credentials
        
    Returns:
        Dictionary with CORS configuration
    """
    config = CORSSecurityConfig()
    
    cors_config = {
        "allow_origins": get_cors_origins(environment),
        "allow_credentials": allow_credentials,
        "allow_methods": config.allowed_methods,
        "allow_headers": config.allowed_headers,
        "expose_headers": config.exposed_headers,
        "max_age": config.max_age
    }
    
    logger.info("Secure CORS configuration generated")
    logger.debug(f"Allowed origins: {len(cors_config['allow_origins'])}")
    logger.debug(f"Allowed methods: {cors_config['allow_methods']}")
    logger.debug(f"Allowed headers: {len(cors_config['allow_headers'])}")
    
    return cors_config


def validate_cors_origin(origin: str, environment: str = "development") -> bool:
    """
    Validate if an origin is allowed for the given environment
    
    Args:
        origin: Origin to validate
        environment: Environment name
        
    Returns:
        True if origin is allowed, False otherwise
    """
    allowed_origins = get_cors_origins(environment)
    return origin in allowed_origins


def log_cors_security_info():
    """Log CORS security configuration information"""
    config = CORSSecurityConfig()
    
    logger.info("=== CORS Security Configuration ===")
    logger.info(f"Allowed Methods: {', '.join(config.allowed_methods)}")
    logger.info(f"Allowed Headers: {len(config.allowed_headers)} headers")
    logger.info(f"Exposed Headers: {', '.join(config.exposed_headers)}")
    logger.info(f"Max Age: {config.max_age} seconds")
    logger.info(f"Development Origins: {len(config.development_origins)}")
    logger.info(f"Production Origins: {len(config.production_origins)}")
    logger.info("==================================")


# Security validation functions
def is_secure_origin(origin: str) -> bool:
    """Check if origin uses HTTPS (secure) protocol"""
    return origin.startswith("https://")


def is_localhost_origin(origin: str) -> bool:
    """Check if origin is localhost/127.0.0.1"""
    return "localhost" in origin or "127.0.0.1" in origin


def is_local_network_origin(origin: str) -> bool:
    """Check if origin is from local network (RFC 1918)"""
    import re
    # Match private IP ranges: 10.x.x.x, 172.16-31.x.x, 192.168.x.x
    private_ip_pattern = r'https?://(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)'
    return bool(re.match(private_ip_pattern, origin))


def get_cors_security_headers() -> Dict[str, str]:
    """Get additional security headers for CORS responses"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "cross-origin"
    }


# Environment-specific configurations
CORS_CONFIGS = {
    "development": {
        "strict_mode": False,
        "log_violations": True,
        "allow_local_network": True,
        "require_https": False
    },
    "staging": {
        "strict_mode": True,
        "log_violations": True,
        "allow_local_network": True, 
        "require_https": False
    },
    "production": {
        "strict_mode": True,
        "log_violations": True,
        "allow_local_network": True,  # For admin access
        "require_https": True
    }
}


def get_environment_config(environment: str) -> Dict[str, Any]:
    """Get environment-specific CORS configuration"""
    return CORS_CONFIGS.get(environment.lower(), CORS_CONFIGS["development"])