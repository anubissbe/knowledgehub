"""
Enhanced Input Validation Middleware

Extends the existing validation middleware with context-aware validation,
advanced threat detection, and performance optimizations.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException
import ipaddress

from ..security.validation_enhanced import (
    ContextAwareRequestValidator,
    ValidationContext,
    AdvancedThreatType,
    validation_metrics
)
from ..security.validation import ValidationLevel, ContentType
from ..security.monitoring import log_security_event, SecurityEventType, ThreatLevel

logger = logging.getLogger(__name__)


class EnhancedValidationMiddleware(BaseHTTPMiddleware):
    """
    Enhanced validation middleware with context-aware validation
    
    Features:
    - Context-aware validation based on user, location, time
    - Advanced threat pattern detection
    - Performance optimization with caching
    - Real-time metrics collection
    - Machine learning integration ready
    """
    
    def __init__(self, app, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        super().__init__(app)
        self.validation_level = validation_level
        self.validator = ContextAwareRequestValidator(validation_level)
        
        # Enhanced validation rules with context awareness
        self.context_rules = {
            '/api/v1/sources': {
                'guest_restrictions': {
                    'max_url_length': 200,
                    'allowed_schemes': ['https'],
                    'blocked_tlds': ['.local', '.internal']
                },
                'high_risk_restrictions': {
                    'no_file_uploads': True,
                    'max_description_length': 100
                }
            },
            '/api/v1/search': {
                'guest_restrictions': {
                    'max_query_length': 100,
                    'no_special_chars': True
                },
                'rate_limits': {
                    'guest': 10,
                    'user': 100,
                    'admin': 1000
                }
            }
        }
        
        # User session tracking for rate limiting
        self.user_sessions = {}
        
        # Geographic IP ranges (simplified)
        self.high_risk_countries = {
            'CN', 'RU', 'KP', 'IR', 'SY', 'AF'
        }
        
        logger.info(f"Enhanced validation middleware initialized with {validation_level.value} level")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with enhanced validation"""
        start_time = time.time()
        
        # Skip validation for certain endpoints
        skip_paths = {'/health', '/metrics', '/api/docs', '/api/redoc', '/api/openapi.json'}
        if any(skip_path in str(request.url.path) for skip_path in skip_paths):
            return await call_next(request)
        
        # Build validation context
        context = await self._build_validation_context(request)
        
        try:
            # Enhanced header validation
            await self._validate_headers_enhanced(request, context)
            
            # Enhanced query parameter validation
            await self._validate_query_params_enhanced(request, context)
            
            # Enhanced request body validation for POST/PUT/PATCH
            if request.method in ["POST", "PUT", "PATCH"]:
                await self._validate_request_body_enhanced(request, context)
            
            # Process request
            response = await call_next(request)
            
            # Record successful validation metrics
            validation_time = time.time() - start_time
            validation_metrics.record_validation(
                "request", "full", True, validation_time
            )
            
            return response
            
        except HTTPException as e:
            # Enhanced error logging
            await self._log_validation_failure(request, context, e)
            
            # Record failure metrics
            validation_time = time.time() - start_time
            validation_metrics.record_validation(
                "request", "full", False, validation_time, [e.detail]
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "Enhanced validation failed",
                    "detail": e.detail,
                    "context": {
                        "endpoint": str(request.url.path),
                        "validation_level": self.validation_level.value,
                        "timestamp": time.time()
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Enhanced validation middleware error: {e}")
            
            # Log unexpected error
            await log_security_event(
                SecurityEventType.MALFORMED_REQUEST,
                ThreatLevel.HIGH,
                context.source_ip,
                context.user_agent,
                context.endpoint,
                context.method,
                f"Enhanced validation error: {str(e)}"
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Request processing failed",
                    "detail": "Internal validation error"
                }
            )
    
    async def _build_validation_context(self, request: Request) -> ValidationContext:
        """Build comprehensive validation context"""
        
        # Get basic request info
        source_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        endpoint = str(request.url.path)
        method = request.method
        
        # Determine user info (simplified - would integrate with auth system)
        user_role = self._determine_user_role(request)
        user_id = self._get_user_id(request)
        is_authenticated = self._is_authenticated(request)
        
        # Geographic info (simplified)
        source_country = self._get_country_from_ip(source_ip)
        
        # Rate limiting info
        request_count = self._get_request_count(source_ip, user_id)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            source_ip, source_country, user_agent, request_count, is_authenticated
        )
        
        return ValidationContext(
            user_role=user_role,
            user_id=user_id,
            source_ip=source_ip,
            source_country=source_country,
            request_count=request_count,
            endpoint=endpoint,
            method=method,
            user_agent=user_agent,
            is_authenticated=is_authenticated,
            risk_score=risk_score
        )
    
    async def _validate_headers_enhanced(self, request: Request, context: ValidationContext):
        """Enhanced header validation with context awareness"""
        
        headers = dict(request.headers)
        
        # Check for suspicious header combinations
        suspicious_combinations = [
            # Command injection attempts in headers
            ('x-forwarded-for', 'user-agent'),
            ('x-real-ip', 'authorization'),
        ]
        
        for header1, header2 in suspicious_combinations:
            if header1 in headers and header2 in headers:
                value1 = headers[header1].lower()
                value2 = headers[header2].lower()
                
                # Check for injection patterns across headers
                combined = f"{value1} {value2}"
                if any(pattern in combined for pattern in ['$(', '`', '|', ';']):
                    await log_security_event(
                        SecurityEventType.INJECTION_ATTEMPT,
                        ThreatLevel.HIGH,
                        context.source_ip,
                        context.user_agent,
                        context.endpoint,
                        context.method,
                        f"Suspicious header combination: {header1}, {header2}"
                    )
                    
                    if self.validation_level == ValidationLevel.STRICT:
                        raise HTTPException(
                            status_code=400,
                            detail="Suspicious header combination detected"
                        )
        
        # Context-aware header validation
        if context.risk_score > 0.7:
            # High risk - stricter validation
            for header_name, header_value in headers.items():
                if len(header_value) > 1024:  # Shorter limit for high risk
                    raise HTTPException(
                        status_code=400,
                        detail=f"Header too long for risk level: {header_name}"
                    )
    
    async def _validate_query_params_enhanced(self, request: Request, context: ValidationContext):
        """Enhanced query parameter validation"""
        
        query_params = dict(request.query_params)
        
        if not query_params:
            return
        
        # Context-aware parameter limits
        max_params = 50
        if context.user_role == 'guest':
            max_params = 10
        elif context.risk_score > 0.8:
            max_params = 5
        
        if len(query_params) > max_params:
            raise HTTPException(
                status_code=400,
                detail=f"Too many query parameters for your access level: {len(query_params)} > {max_params}"
            )
        
        # Advanced parameter validation
        for param_name, param_value in query_params.items():
            # Check for parameter pollution
            if isinstance(param_value, list) and len(param_value) > 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Parameter pollution detected: {param_name}"
                )
            
            # Context-aware value validation
            if context.user_role == 'guest':
                # Guests cannot use certain parameters
                restricted_params = {'debug', 'admin', 'internal', 'system'}
                if any(restricted in param_name.lower() for restricted in restricted_params):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Parameter not allowed for guest users: {param_name}"
                    )
    
    async def _validate_request_body_enhanced(self, request: Request, context: ValidationContext):
        """Enhanced request body validation with context awareness"""
        
        try:
            body = await request.body()
            
            if not body:
                return
            
            # Context-aware size limits
            max_size = 10 * 1024 * 1024  # 10MB default
            if context.user_role == 'guest':
                max_size = 1 * 1024 * 1024  # 1MB for guests
            elif context.risk_score > 0.8:
                max_size = 100 * 1024  # 100KB for high risk
            
            if len(body) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large for your access level: {len(body)} > {max_size}"
                )
            
            # JSON validation with context
            content_type = request.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                import json
                
                try:
                    body_str = body.decode('utf-8')
                    data = json.loads(body_str)
                    
                    # Get validation rules for this endpoint
                    validation_rules = self._get_enhanced_validation_rules(
                        context.endpoint, context
                    )
                    
                    # Validate with context
                    if isinstance(data, dict):
                        validated_data = await self.validator.validate_request_with_context(
                            data, validation_rules, context
                        )
                        
                        # Store validated data
                        request.state.validated_data = validated_data
                    
                    elif isinstance(data, list):
                        # Validate list items
                        validated_list = []
                        for item in data:
                            if isinstance(item, dict):
                                validated_item = await self.validator.validate_request_with_context(
                                    item, validation_rules, context
                                )
                                validated_list.append(validated_item)
                            else:
                                validated_list.append(item)
                        
                        request.state.validated_data = validated_list
                
                except json.JSONDecodeError as e:
                    await log_security_event(
                        SecurityEventType.MALFORMED_REQUEST,
                        ThreatLevel.MEDIUM,
                        context.source_ip,
                        context.user_agent,
                        context.endpoint,
                        context.method,
                        f"Invalid JSON: {str(e)}"
                    )
                    raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Enhanced body validation error: {e}")
            raise HTTPException(status_code=400, detail="Request body validation failed")
    
    def _get_enhanced_validation_rules(self, endpoint: str, 
                                     context: ValidationContext) -> Dict[str, Dict[str, Any]]:
        """Get enhanced validation rules based on context"""
        
        # Start with base rules
        base_rules = {
            'content': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 10000},
            'description': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 2000},
            'url': {'content_type': ContentType.URL, 'required': False},
            'name': {'content_type': ContentType.TEXT, 'required': False, 'max_length': 255},
        }
        
        # Apply context-specific modifications
        if context.user_role == 'guest':
            # Stricter limits for guests
            base_rules['content']['max_length'] = 1000
            base_rules['description']['max_length'] = 500
        
        if context.risk_score > 0.8:
            # Very strict limits for high risk
            base_rules['content']['max_length'] = 500
            base_rules['description']['max_length'] = 200
        
        # Endpoint-specific rules
        if '/api/v1/sources' in endpoint:
            base_rules.update({
                'source_type': {'content_type': ContentType.TEXT, 'required': True, 'max_length': 50},
                'config': {'content_type': ContentType.JSON, 'required': False}
            })
        
        return base_rules
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else "unknown"
    
    def _determine_user_role(self, request: Request) -> str:
        """Determine user role from request (simplified)"""
        # This would integrate with your actual auth system
        auth_header = request.headers.get("authorization", "")
        
        if "admin" in auth_header.lower():
            return "admin"
        elif auth_header:
            return "user"
        else:
            return "guest"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Get user ID from request (simplified)"""
        # This would integrate with your actual auth system
        return request.headers.get("x-user-id")
    
    def _is_authenticated(self, request: Request) -> bool:
        """Check if user is authenticated"""
        return bool(request.headers.get("authorization"))
    
    def _get_country_from_ip(self, ip: str) -> str:
        """Get country from IP address (simplified)"""
        # This would integrate with a GeoIP service
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private:
                return "LOCAL"
        except ValueError:
            pass
        
        # Simplified country detection (in production use GeoIP)
        if ip.startswith("192.168."):
            return "LOCAL"
        
        return "UNKNOWN"
    
    def _get_request_count(self, source_ip: str, user_id: Optional[str]) -> int:
        """Get request count for rate limiting"""
        key = user_id or source_ip
        current_time = time.time()
        
        # Clean old entries
        if key in self.user_sessions:
            self.user_sessions[key] = [
                timestamp for timestamp in self.user_sessions[key]
                if current_time - timestamp < 60  # Last minute
            ]
        else:
            self.user_sessions[key] = []
        
        # Add current request
        self.user_sessions[key].append(current_time)
        
        return len(self.user_sessions[key])
    
    def _calculate_risk_score(self, source_ip: str, source_country: str,
                            user_agent: str, request_count: int,
                            is_authenticated: bool) -> float:
        """Calculate risk score for the request"""
        
        risk_score = 0.0
        
        # IP-based risk
        try:
            ip_obj = ipaddress.ip_address(source_ip)
            if ip_obj.is_private:
                risk_score += 0.1  # Local IPs are lower risk
            else:
                risk_score += 0.3  # External IPs are higher risk
        except ValueError:
            risk_score += 0.5  # Invalid IP is suspicious
        
        # Country-based risk
        if source_country in self.high_risk_countries:
            risk_score += 0.4
        
        # User agent risk
        user_agent_lower = user_agent.lower()
        suspicious_agents = ['curl', 'wget', 'python', 'bot', 'scanner']
        if any(agent in user_agent_lower for agent in suspicious_agents):
            risk_score += 0.3
        
        # Rate limiting risk
        if request_count > 100:
            risk_score += 0.3
        elif request_count > 50:
            risk_score += 0.2
        
        # Authentication status
        if not is_authenticated:
            risk_score += 0.2
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    async def _log_validation_failure(self, request: Request, 
                                    context: ValidationContext, error: HTTPException):
        """Enhanced logging for validation failures"""
        
        # Determine threat level based on error and context
        threat_level = ThreatLevel.MEDIUM
        if context.risk_score > 0.8:
            threat_level = ThreatLevel.HIGH
        elif "injection" in str(error.detail).lower():
            threat_level = ThreatLevel.CRITICAL
        
        await log_security_event(
            SecurityEventType.MALFORMED_REQUEST,
            threat_level,
            context.source_ip,
            context.user_agent,
            context.endpoint,
            context.method,
            f"Enhanced validation failed - Risk: {context.risk_score:.2f}, Error: {error.detail}"
        )