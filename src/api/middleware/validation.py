"""
Input Validation Middleware

Automatically validates and sanitizes all incoming requests to protect
against malicious input and ensure data integrity.
"""

import json
import logging
from typing import Dict, Any, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException
import asyncio

from ..security.validation import (
    RequestValidator,
    ValidationLevel,
    ContentType,
    moderate_validator,
    strict_validator
)
from ..security.monitoring import log_security_event, SecurityEventType, ThreatLevel

logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic input validation and sanitization
    
    Features:
    - Automatic request payload validation
    - Query parameter sanitization
    - Header validation
    - Content-type specific validation
    - Security event logging for validation failures
    """
    
    def __init__(self, app, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        super().__init__(app)
        self.validation_level = validation_level
        
        # Use appropriate validator based on level
        if validation_level == ValidationLevel.STRICT:
            self.validator = strict_validator
        else:
            self.validator = moderate_validator
        
        # Endpoints that require strict validation
        self.strict_endpoints = {
            '/api/auth/',
            '/api/security/',
            '/api/memory/',
            '/api/v1/sources',
            '/api/v1/jobs'
        }
        
        # Endpoints to skip validation (performance critical)
        self.skip_validation = {
            '/health',
            '/metrics',
            '/api/docs',
            '/api/redoc',
            '/api/openapi.json'
        }
        
        # Field validation rules for common endpoints
        self.validation_rules = {
            'default': {
                'name': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 255},
                'description': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 2000},
                'url': {'content_type': ContentType.URL, 'required': False},
                'email': {'content_type': ContentType.EMAIL, 'required': False},
                'filename': {'content_type': ContentType.FILENAME, 'required': False},
                'content': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 100000},
                'query': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 1000},
                'id': {'content_type': ContentType.UUID, 'required': False},
                'api_key': {'content_type': ContentType.API_KEY, 'required': False}
            },
            '/api/v1/sources': {
                'source_type': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 50},
                'url': {'content_type': ContentType.URL, 'required': True},
                'name': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 255},
                'description': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 2000},
                'config': {'content_type': ContentType.JSON, 'required': False}
            },
            '/api/v1/search': {
                'query': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 1000},
                'filters': {'content_type': ContentType.JSON, 'required': False},
                'limit': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 10},
                'offset': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 10}
            },
            '/api/v1/chunks/batch': {
                'document_id': {'content_type': ContentType.UUID, 'required': False},
                'source_id': {'content_type': ContentType.UUID, 'required': False},
                'content': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 100000},
                'chunk_type': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 50},
                'chunk_index': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 10},
                'metadata': {'content_type': ContentType.JSON, 'required': False},
                'embedding_id': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 255},
                'parent_heading': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 500},
                'content_hash': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 255},
                'title': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 500},
                'url': {'content_type': ContentType.URL, 'required': False}
            },
            '/api/memory/': {
                'content': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 50000},
                'memory_type': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 50},
                'importance': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 10},
                'context': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 10000},
                'session_id': {'content_type': ContentType.UUID, 'required': False}
            },
            '/api/auth/': {
                'username': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 255},
                'password': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 255},
                'email': {'content_type': ContentType.EMAIL, 'required': False},
                'api_key': {'content_type': ContentType.API_KEY, 'required': False}
            },
            '/api/v1/jobs': {
                # Jobs endpoints generally don't require body validation
                # Cancel endpoint takes empty body, other endpoints are minimal
            }
        }
        
        logger.info(f"Validation middleware initialized with {validation_level.value} level")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with validation"""
        
        # Skip validation for certain endpoints
        if any(skip_path in str(request.url.path) for skip_path in self.skip_validation):
            return await call_next(request)
        
        # Get client info for logging
        source_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        try:
            # Validate headers
            await self._validate_headers(request, source_ip, user_agent)
            
            # Validate query parameters
            await self._validate_query_params(request, source_ip, user_agent)
            
            # Validate request body for POST/PUT/PATCH
            if request.method in ["POST", "PUT", "PATCH"]:
                await self._validate_request_body(request, source_ip, user_agent)
            
            # Continue with request processing
            response = await call_next(request)
            
            # Log successful validation
            logger.debug(f"Request validation passed for {request.method} {request.url.path}")
            
            return response
            
        except HTTPException as e:
            # Log validation failure
            await log_security_event(
                SecurityEventType.MALFORMED_REQUEST,
                ThreatLevel.MEDIUM,
                source_ip,
                user_agent,
                str(request.url.path),
                request.method,
                f"Request validation failed: {e.detail}"
            )
            
            # Return validation error
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "Request validation failed",
                    "detail": e.detail,
                    "path": str(request.url.path)
                }
            )
            
        except Exception as e:
            # Log unexpected validation error
            logger.error(f"Validation middleware error: {e}")
            
            # Only log security event if it's not a recursion issue
            if "recursion" not in str(e).lower() and "maximum" not in str(e).lower():
                try:
                    await log_security_event(
                        SecurityEventType.MALFORMED_REQUEST,
                        ThreatLevel.HIGH,
                        source_ip,
                        user_agent,
                        str(request.url.path),
                        request.method,
                        f"Validation middleware error: {str(e)}"
                    )
                except Exception as log_error:
                    logger.error(f"Failed to log security event: {log_error}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Request processing failed",
                    "detail": "Internal validation error"
                }
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check X-Forwarded-For header first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"
    
    async def _validate_headers(self, request: Request, source_ip: str, user_agent: str):
        """Validate HTTP headers"""
        try:
            # Get headers as dict
            headers = dict(request.headers)
            
            # Validate headers
            validated_headers = self.validator.validate_headers(headers)
            
            # Check for suspicious header patterns
            for header_name, header_value in headers.items():
                if self._is_suspicious_header(header_name, header_value):
                    await log_security_event(
                        SecurityEventType.SUSPICIOUS_REQUEST,
                        ThreatLevel.MEDIUM,
                        source_ip,
                        user_agent,
                        str(request.url.path),
                        request.method,
                        f"Suspicious header detected: {header_name}"
                    )
                    
                    if self.validation_level == ValidationLevel.STRICT:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Suspicious header detected: {header_name}"
                        )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Header validation error: {e}")
            raise HTTPException(status_code=400, detail="Header validation failed")
    
    async def _validate_query_params(self, request: Request, source_ip: str, user_agent: str):
        """Validate query parameters"""
        try:
            # Get query parameters
            query_params = dict(request.query_params)
            
            if not query_params:
                return
            
            # Validate query parameters
            validated_params = self.validator.validate_query_params(query_params)
            
            # Check for excessive number of parameters (potential DoS)
            if len(query_params) > 50:
                await log_security_event(
                    SecurityEventType.DOS_ATTEMPT,
                    ThreatLevel.MEDIUM,
                    source_ip,
                    user_agent,
                    str(request.url.path),
                    request.method,
                    f"Excessive query parameters: {len(query_params)}"
                )
                
                if self.validation_level == ValidationLevel.STRICT:
                    raise HTTPException(
                        status_code=400,
                        detail="Too many query parameters"
                    )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Query parameter validation error: {e}")
            raise HTTPException(status_code=400, detail="Query parameter validation failed")
    
    async def _validate_request_body(self, request: Request, source_ip: str, user_agent: str):
        """Validate request body"""
        try:
            # Read request body
            body = await request.body()
            
            if not body:
                return
            
            # Check content size
            if len(body) > 10 * 1024 * 1024:  # 10MB limit
                await log_security_event(
                    SecurityEventType.DOS_ATTEMPT,
                    ThreatLevel.HIGH,
                    source_ip,
                    user_agent,
                    str(request.url.path),
                    request.method,
                    f"Oversized request body: {len(body)} bytes"
                )
                raise HTTPException(status_code=413, detail="Request body too large")
            
            # Parse JSON if content-type is JSON
            content_type = request.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                try:
                    body_str = body.decode('utf-8')
                    data = json.loads(body_str)
                    
                    # Handle different payload types
                    if isinstance(data, list):
                        # For list payloads (like batch endpoints), validate each item
                        validated_data = []
                        for item in data:
                            if isinstance(item, dict):
                                # Get validation rules for this endpoint
                                validation_rules = self._get_validation_rules(request.url.path)
                                validated_item = self.validator.validate_request_data(item, validation_rules)
                                validated_data.append(validated_item)
                            else:
                                # Skip validation for non-dict items in list
                                validated_data.append(item)
                    elif isinstance(data, dict):
                        # Get validation rules for this endpoint
                        validation_rules = self._get_validation_rules(request.url.path)
                        
                        # Validate the data
                        validated_data = self.validator.validate_request_data(data, validation_rules)
                    else:
                        # For other data types, skip validation
                        validated_data = data
                    
                    # Store validated data for use by the endpoint
                    # Note: This is a simplified approach; in production you might want
                    # to use a more sophisticated method to pass validated data
                    request.state.validated_data = validated_data
                    
                except json.JSONDecodeError as e:
                    await log_security_event(
                        SecurityEventType.MALFORMED_REQUEST,
                        ThreatLevel.MEDIUM,
                        source_ip,
                        user_agent,
                        str(request.url.path),
                        request.method,
                        f"Invalid JSON: {str(e)}"
                    )
                    raise HTTPException(status_code=400, detail="Invalid JSON format")
                
                except Exception as e:
                    # Log the specific validation error
                    logger.error(f"JSON validation failed for {request.url.path}: {str(e)}")
                    await log_security_event(
                        SecurityEventType.MALFORMED_REQUEST,
                        ThreatLevel.MEDIUM,
                        source_ip,
                        user_agent,
                        str(request.url.path),
                        request.method,
                        f"JSON validation failed: {str(e)}"
                    )
                    raise HTTPException(status_code=400, detail="Request validation failed")
            
            elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
                # Handle form data validation
                # This would require more complex parsing
                logger.info(f"Form data validation not implemented for {content_type}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Request body validation error: {e}")
            raise HTTPException(status_code=400, detail="Request body validation failed")
    
    def _get_validation_rules(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Get validation rules for a specific endpoint"""
        
        # Check for exact match first
        if path in self.validation_rules:
            return self.validation_rules[path]
        
        # Check for prefix matches
        for rule_path, rules in self.validation_rules.items():
            if rule_path != 'default' and path.startswith(rule_path):
                return rules
        
        # Use default rules
        return self.validation_rules['default']
    
    def _is_suspicious_header(self, header_name: str, header_value: str) -> bool:
        """Check if header contains suspicious patterns"""
        
        # Check header name
        suspicious_headers = {
            'x-forwarded-host', 'x-rewrite-url', 'x-http-method-override'
        }
        
        if header_name.lower() in suspicious_headers:
            return True
        
        # Check for injection patterns in header values
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'\$\(',
            r'`.*`',
            r'\|\s*[a-zA-Z]+'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, header_value, re.IGNORECASE):
                return True
        
        # Check for excessively long headers
        if len(header_value) > 8192:  # 8KB limit
            return True
        
        return False


# Import at the end to avoid circular imports
import re