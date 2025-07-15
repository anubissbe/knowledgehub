"""
Validation Utility Functions and Decorators

Provides convenient functions and decorators for input validation
throughout the application.
"""

import functools
import inspect
import logging
from typing import Any, Dict, List, Callable, Optional, Union
from fastapi import HTTPException, Request
from pydantic import ValidationError

from ..security.validation import (
    RequestValidator,
    ValidationLevel,
    ContentType,
    SecurityValidator
)
from ..security.monitoring import log_security_event, SecurityEventType, ThreatLevel

logger = logging.getLogger(__name__)


def validate_request_data(validation_level: ValidationLevel = ValidationLevel.MODERATE):
    """
    Decorator to automatically validate request data using Pydantic models
    
    Usage:
        @validate_request_data()
        async def create_source(source_data: SecureSourceCreate):
            # source_data is automatically validated
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request object if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            try:
                # Call original function - Pydantic validation happens automatically
                return await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)
                
            except ValidationError as e:
                # Log validation failure
                if request:
                    source_ip = _get_client_ip(request)
                    user_agent = request.headers.get("user-agent", "")
                    
                    await log_security_event(
                        SecurityEventType.MALFORMED_REQUEST,
                        ThreatLevel.MEDIUM,
                        source_ip,
                        user_agent,
                        str(request.url.path),
                        request.method,
                        f"Pydantic validation failed: {str(e)}"
                    )
                
                # Convert Pydantic validation error to HTTP exception
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Request validation failed",
                        "validation_errors": e.errors(),
                        "message": "The request data is invalid"
                    }
                )
            
            except Exception as e:
                logger.error(f"Validation decorator error: {e}")
                raise
        
        return wrapper
    return decorator


def validate_file_upload(max_size: int = 10*1024*1024, allowed_types: Optional[List[str]] = None):
    """
    Decorator to validate file uploads
    
    Args:
        max_size: Maximum file size in bytes
        allowed_types: List of allowed MIME types
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Implementation would depend on the file upload framework
            # This is a placeholder for file validation logic
            return await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    return decorator


def sanitize_output(fields: Optional[List[str]] = None):
    """
    Decorator to sanitize output data before returning to client
    
    Args:
        fields: List of fields to sanitize (if None, sanitizes all string fields)
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Sanitize result if it's a dictionary
            if isinstance(result, dict):
                return _sanitize_dict(result, fields)
            elif isinstance(result, list):
                return [_sanitize_dict(item, fields) if isinstance(item, dict) else item for item in result]
            
            return result
        
        return wrapper
    return decorator


def require_validation_level(level: ValidationLevel):
    """
    Decorator to enforce a specific validation level for an endpoint
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # This would integrate with the validation middleware
            # For now, it's a placeholder
            return await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    return decorator


class ValidationUtils:
    """Utility class for common validation operations"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validator = SecurityValidator(validation_level)
        self.request_validator = RequestValidator(validation_level)
    
    def validate_search_query(self, query: str) -> str:
        """Validate and sanitize search queries"""
        result = self.validator.validate_input(
            query, 
            ContentType.TEXT, 
            "search_query", 
            required=True
        )
        
        if not result.is_valid:
            raise ValueError(f"Invalid search query: {'; '.join(result.issues)}")
        
        return result.sanitized_value
    
    def validate_pagination_params(self, limit: Optional[int] = None, 
                                 offset: Optional[int] = None) -> Dict[str, int]:
        """Validate pagination parameters"""
        # Default values
        validated_limit = 20
        validated_offset = 0
        
        # Validate limit
        if limit is not None:
            if not isinstance(limit, int) or limit < 1 or limit > 1000:
                raise ValueError("Limit must be between 1 and 1000")
            validated_limit = limit
        
        # Validate offset
        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                raise ValueError("Offset must be non-negative")
            validated_offset = offset
        
        return {"limit": validated_limit, "offset": validated_offset}
    
    def validate_sort_params(self, sort_by: Optional[str] = None, 
                           sort_order: Optional[str] = None,
                           allowed_fields: Optional[List[str]] = None) -> Dict[str, str]:
        """Validate sorting parameters"""
        validated_sort_by = "created_at"  # Default
        validated_sort_order = "desc"     # Default
        
        # Validate sort field
        if sort_by is not None:
            if allowed_fields and sort_by not in allowed_fields:
                raise ValueError(f"Invalid sort field. Allowed: {', '.join(allowed_fields)}")
            
            # Sanitize field name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', sort_by):
                raise ValueError("Invalid sort field format")
            
            validated_sort_by = sort_by
        
        # Validate sort order
        if sort_order is not None:
            if sort_order.lower() not in ['asc', 'desc']:
                raise ValueError("Sort order must be 'asc' or 'desc'")
            validated_sort_order = sort_order.lower()
        
        return {"sort_by": validated_sort_by, "sort_order": validated_sort_order}
    
    def validate_date_range(self, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Validate date range parameters"""
        from datetime import datetime
        
        validated_start = None
        validated_end = None
        
        if start_date:
            try:
                # Try to parse the date
                parsed_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                validated_start = parsed_date.isoformat()
            except ValueError:
                raise ValueError("Invalid start date format. Use ISO 8601 format.")
        
        if end_date:
            try:
                parsed_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                validated_end = parsed_date.isoformat()
            except ValueError:
                raise ValueError("Invalid end date format. Use ISO 8601 format.")
        
        # Validate date range logic
        if validated_start and validated_end:
            start_dt = datetime.fromisoformat(validated_start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(validated_end.replace('Z', '+00:00'))
            
            if start_dt >= end_dt:
                raise ValueError("Start date must be before end date")
        
        return {"start_date": validated_start, "end_date": validated_end}
    
    def validate_filter_params(self, filters: Optional[Dict[str, Any]] = None,
                             allowed_filters: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate filter parameters"""
        if not filters:
            return {}
        
        if not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary")
        
        validated_filters = {}
        
        for key, value in filters.items():
            # Check allowed filters
            if allowed_filters and key not in allowed_filters:
                raise ValueError(f"Invalid filter: {key}. Allowed: {', '.join(allowed_filters)}")
            
            # Validate filter key format
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid filter key format: {key}")
            
            # Validate filter value
            if isinstance(value, str):
                result = self.validator.validate_input(
                    value, ContentType.TEXT, f"filter_{key}", required=False
                )
                if result.is_valid:
                    validated_filters[key] = result.sanitized_value
                else:
                    raise ValueError(f"Invalid filter value for {key}: {'; '.join(result.issues)}")
            
            elif isinstance(value, (int, float, bool)):
                validated_filters[key] = value
            
            elif isinstance(value, list):
                # Validate list values
                validated_list = []
                for item in value:
                    if isinstance(item, str):
                        result = self.validator.validate_input(
                            item, ContentType.TEXT, f"filter_{key}_item", required=False
                        )
                        if result.is_valid:
                            validated_list.append(result.sanitized_value)
                        else:
                            raise ValueError(f"Invalid filter list item for {key}")
                    else:
                        validated_list.append(item)
                
                validated_filters[key] = validated_list
            
            else:
                raise ValueError(f"Unsupported filter value type for {key}")
        
        return validated_filters


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    return request.client.host if request.client else "unknown"


def _sanitize_dict(data: Dict[str, Any], fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Sanitize dictionary values"""
    import html
    
    sanitized = {}
    
    for key, value in data.items():
        if fields is None or key in fields:
            if isinstance(value, str):
                # Basic HTML escaping for output
                sanitized[key] = html.escape(value, quote=True)
            elif isinstance(value, dict):
                sanitized[key] = _sanitize_dict(value, fields)
            elif isinstance(value, list):
                sanitized[key] = [
                    html.escape(item, quote=True) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    
    return sanitized


# Import at the end to avoid circular imports
import re