"""
Enhanced Pydantic Models with Security Validation

Provides Pydantic models with built-in security validation for all API endpoints.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, root_validator
from datetime import datetime
from enum import Enum

from ..security.validation import (
    SecurityValidator,
    ValidationLevel,
    ContentType,
    validate_text,
    validate_email,
    validate_url,
    sanitize_filename
)


class ValidationConfig:
    """Shared validation configuration"""
    # Enable field validation
    validate_assignment = True
    # Forbid extra fields to prevent injection
    extra = "forbid"
    # Use enum values
    use_enum_values = True


class SecureBaseModel(BaseModel):
    """Base model with security validation"""
    
    class Config(ValidationConfig):
        pass
    
    @root_validator(pre=True)
    def validate_no_suspicious_fields(cls, values):
        """Check for suspicious field names"""
        if isinstance(values, dict):
            suspicious_fields = {
                '__proto__', 'constructor', 'prototype', 
                'eval', 'function', 'script'
            }
            
            for field_name in values.keys():
                if field_name in suspicious_fields:
                    raise ValueError(f"Suspicious field name detected: {field_name}")
                
                # Check for injection patterns in field names
                if re.search(r'[<>"\']|javascript:|data:', str(field_name), re.IGNORECASE):
                    raise ValueError(f"Invalid characters in field name: {field_name}")
        
        return values


class SecureTextField(str):
    """Secure text field with automatic validation"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v, field=None):
        if v is None:
            return v
        
        # Get field name and constraints
        field_name = field.name if field else "text"
        max_length = field.field_info.max_length if field and field.field_info else 10000
        
        # Validate and sanitize
        try:
            return validate_text(str(v), max_length=max_length, required=True)
        except ValueError as e:
            raise ValueError(f"Text validation failed for {field_name}: {str(e)}")


class SecureEmailField(str):
    """Secure email field with validation"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if v is None:
            return v
        
        try:
            return validate_email(str(v))
        except ValueError as e:
            raise ValueError(f"Email validation failed: {str(e)}")


class SecureUrlField(str):
    """Secure URL field with validation"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if v is None:
            return v
        
        try:
            return validate_url(str(v))
        except ValueError as e:
            raise ValueError(f"URL validation failed: {str(e)}")


class SecureFilenameField(str):
    """Secure filename field with validation"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if v is None:
            return v
        
        try:
            return sanitize_filename(str(v))
        except ValueError as e:
            raise ValueError(f"Filename validation failed: {str(e)}")


# Enhanced models for common API operations

class SecureSourceCreate(SecureBaseModel):
    """Secure model for creating knowledge sources"""
    
    name: str = Field(..., min_length=1, max_length=255, description="Source name")
    description: Optional[str] = Field(None, max_length=2000, description="Source description")
    source_type: str = Field(..., min_length=1, max_length=50, description="Source type")
    url: SecureUrlField = Field(..., description="Source URL")
    config: Optional[Dict[str, Any]] = Field(None, description="Source configuration")
    
    @field_validator('name')
    def validate_name(cls, v):
        return validate_text(v, max_length=255)
    
    @field_validator('description')
    def validate_description(cls, v):
        if v is None:
            return v
        return validate_text(v, max_length=2000, required=False)
    
    @field_validator('source_type')
    def validate_source_type(cls, v):
        # Only allow specific source types
        allowed_types = {
            'website', 'api', 'database', 'file', 'git', 
            'documentation', 'wiki', 'blog', 'forum'
        }
        
        v = validate_text(v, max_length=50)
        if v.lower() not in allowed_types:
            raise ValueError(f"Invalid source type. Allowed: {', '.join(allowed_types)}")
        
        return v.lower()
    
    @field_validator('config')
    def validate_config(cls, v):
        if v is None:
            return v
        
        # Validate configuration object
        if not isinstance(v, dict):
            raise ValueError("Config must be a dictionary")
        
        # Check for suspicious keys
        suspicious_keys = {'eval', 'exec', 'import', '__'}
        for key in v.keys():
            if any(suspicious in str(key).lower() for suspicious in suspicious_keys):
                raise ValueError(f"Suspicious configuration key: {key}")
        
        return v


class SecureSearchRequest(SecureBaseModel):
    """Secure model for search requests"""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    limit: Optional[int] = Field(20, ge=1, le=100, description="Result limit")
    offset: Optional[int] = Field(0, ge=0, description="Result offset")
    include_memories: Optional[bool] = Field(False, description="Include memory search")
    
    @field_validator('query')
    def validate_query(cls, v):
        # Validate search query
        validated = validate_text(v, max_length=1000)
        
        # Check for SQL injection patterns in search
        sql_patterns = [
            r'\bUNION\b.*\bSELECT\b',
            r'\bDROP\b.*\bTABLE\b',
            r';\s*DELETE\b',
            r';\s*UPDATE\b.*\bSET\b'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, validated, re.IGNORECASE):
                raise ValueError("Search query contains potentially dangerous patterns")
        
        return validated
    
    @field_validator('filters')
    def validate_filters(cls, v):
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError("Filters must be a dictionary")
        
        # Validate filter keys and values
        allowed_filter_keys = {
            'source_type', 'date_range', 'author', 'category', 
            'tag', 'language', 'format', 'status'
        }
        
        for key, value in v.items():
            if key not in allowed_filter_keys:
                raise ValueError(f"Invalid filter key: {key}")
            
            # Validate filter values
            if isinstance(value, str):
                if len(value) > 255:
                    raise ValueError(f"Filter value too long for {key}")
                validate_text(value, max_length=255)
        
        return v


class SecureMemoryCreate(SecureBaseModel):
    """Secure model for creating memories"""
    
    content: str = Field(..., min_length=1, max_length=50000, description="Memory content")
    memory_type: str = Field(..., description="Memory type")
    importance: Optional[str] = Field("medium", description="Memory importance")
    context: Optional[str] = Field(None, max_length=10000, description="Memory context")
    session_id: Optional[str] = Field(None, description="Session ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @field_validator('content')
    def validate_content(cls, v):
        return validate_text(v, max_length=50000)
    
    @field_validator('memory_type')
    def validate_memory_type(cls, v):
        allowed_types = {
            'fact', 'decision', 'context', 'preference', 
            'instruction', 'summary', 'note'
        }
        
        v = validate_text(v, max_length=50)
        if v.lower() not in allowed_types:
            raise ValueError(f"Invalid memory type. Allowed: {', '.join(allowed_types)}")
        
        return v.lower()
    
    @field_validator('importance')
    def validate_importance(cls, v):
        if v is None:
            return "medium"
        
        allowed_levels = {'low', 'medium', 'high', 'critical'}
        v = validate_text(v, max_length=10)
        
        if v.lower() not in allowed_levels:
            raise ValueError(f"Invalid importance level. Allowed: {', '.join(allowed_levels)}")
        
        return v.lower()
    
    @field_validator('context')
    def validate_context(cls, v):
        if v is None:
            return v
        return validate_text(v, max_length=10000, required=False)
    
    @field_validator('session_id')
    def validate_session_id(cls, v):
        if v is None:
            return v
        
        # UUID validation
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, v, re.IGNORECASE):
            raise ValueError("Invalid session ID format")
        
        return v.lower()
    
    @field_validator('metadata')
    def validate_metadata(cls, v):
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Limit metadata size
        if len(str(v)) > 10000:
            raise ValueError("Metadata too large")
        
        return v


class SecureUserCreate(SecureBaseModel):
    """Secure model for user creation"""
    
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: SecureEmailField = Field(..., description="Email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    full_name: Optional[str] = Field(None, max_length=255, description="Full name")
    
    @field_validator('username')
    def validate_username(cls, v):
        # Username should only contain alphanumeric characters and underscores
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Username can only contain letters, numbers, and underscores")
        
        # Reserved usernames
        reserved = {
            'admin', 'root', 'system', 'test', 'user', 'guest', 
            'anonymous', 'null', 'undefined', 'api', 'bot'
        }
        
        if v.lower() in reserved:
            raise ValueError("Username is reserved")
        
        return validate_text(v, max_length=50)
    
    @field_validator('password')
    def validate_password(cls, v):
        # Basic password strength validation
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        # Check for common patterns
        if v.lower() in {'password', '12345678', 'qwerty123', 'admin123'}:
            raise ValueError("Password is too common")
        
        # Must contain at least one letter and one number
        if not re.search(r'[a-zA-Z]', v) or not re.search(r'[0-9]', v):
            raise ValueError("Password must contain both letters and numbers")
        
        return v
    
    @field_validator('full_name')
    def validate_full_name(cls, v):
        if v is None:
            return v
        
        # Allow letters, spaces, hyphens, apostrophes
        if not re.match(r"^[a-zA-Z\s\-']+$", v):
            raise ValueError("Full name contains invalid characters")
        
        return validate_text(v, max_length=255, required=False)


class SecureFileUpload(SecureBaseModel):
    """Secure model for file uploads"""
    
    filename: SecureFilenameField = Field(..., description="File name")
    content_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., ge=1, le=100*1024*1024, description="File size in bytes")  # 100MB max
    description: Optional[str] = Field(None, max_length=1000, description="File description")
    
    @field_validator('content_type')
    def validate_content_type(cls, v):
        # Only allow specific MIME types
        allowed_types = {
            'text/plain', 'text/markdown', 'text/csv',
            'application/json', 'application/xml',
            'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'image/jpeg', 'image/png', 'image/gif', 'image/webp'
        }
        
        if v not in allowed_types:
            raise ValueError(f"File type not allowed: {v}")
        
        return v
    
    @field_validator('description')
    def validate_description(cls, v):
        if v is None:
            return v
        return validate_text(v, max_length=1000, required=False)


class SecureAPIKeyCreate(SecureBaseModel):
    """Secure model for API key creation"""
    
    name: str = Field(..., min_length=1, max_length=255, description="API key name")
    permissions: List[str] = Field(..., description="API key permissions")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")
    
    @field_validator('name')
    def validate_name(cls, v):
        return validate_text(v, max_length=255)
    
    @field_validator('permissions')
    def validate_permissions(cls, v):
        if not isinstance(v, list):
            raise ValueError("Permissions must be a list")
        
        allowed_permissions = {
            'read', 'write', 'delete', 'admin', 
            'sources:read', 'sources:write', 'sources:delete',
            'memories:read', 'memories:write', 'memories:delete',
            'search:read', 'analytics:read'
        }
        
        for permission in v:
            if permission not in allowed_permissions:
                raise ValueError(f"Invalid permission: {permission}")
        
        return v


class SecureConfigUpdate(SecureBaseModel):
    """Secure model for configuration updates"""
    
    settings: Dict[str, Any] = Field(..., description="Configuration settings")
    
    @field_validator('settings')
    def validate_settings(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Settings must be a dictionary")
        
        # Validate setting keys
        allowed_settings = {
            'max_concurrent_scrapers', 'scraper_timeout_ms', 'scraper_rate_limit_rps',
            'max_chunk_size', 'chunk_overlap', 'min_chunk_size',
            'rate_limit_requests_per_minute', 'cors_origins',
            'log_level', 'debug_mode', 'api_workers'
        }
        
        for key, value in v.items():
            if key not in allowed_settings:
                raise ValueError(f"Invalid setting key: {key}")
            
            # Type validation for specific settings
            if key in {'max_concurrent_scrapers', 'scraper_timeout_ms', 'max_chunk_size'}:
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Setting {key} must be a positive integer")
            
            elif key in {'scraper_rate_limit_rps'}:
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Setting {key} must be a positive number")
            
            elif key in {'debug_mode'}:
                if not isinstance(value, bool):
                    raise ValueError(f"Setting {key} must be a boolean")
            
            elif key == 'cors_origins':
                if not isinstance(value, list):
                    raise ValueError("cors_origins must be a list")
                
                for origin in value:
                    try:
                        validate_url(origin)
                    except ValueError:
                        raise ValueError(f"Invalid CORS origin: {origin}")
        
        return v