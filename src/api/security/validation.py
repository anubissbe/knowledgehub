"""
Enhanced Input Validation and Sanitization

Provides comprehensive input validation, sanitization, and data integrity
protection for all API endpoints and user-provided data.
"""

import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum
from dataclasses import dataclass
from urllib.parse import urlparse, quote, unquote
import base64
import binascii

from pydantic import BaseModel, Field, validator
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # Maximum security, reject anything suspicious
    MODERATE = "moderate"  # Balanced security and usability
    PERMISSIVE = "permissive"  # Minimal validation, maximum compatibility


class ContentType(str, Enum):
    """Content types for validation"""
    TEXT = "text"
    HTML = "html"
    EMAIL = "email"
    URL = "url"
    FILENAME = "filename"
    JSON = "json"
    SQL_IDENTIFIER = "sql_identifier"
    API_KEY = "api_key"
    UUID = "uuid"
    BASE64 = "base64"
    MARKDOWN = "markdown"


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    sanitized_value: Any
    original_value: Any
    issues: List[str]
    severity: str  # "low", "medium", "high", "critical"
    
    def __post_init__(self):
        if not self.issues:
            self.issues = []


class SecurityValidator:
    """Comprehensive security validation and sanitization engine"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        
        # Dangerous patterns that should always be blocked
        self.critical_patterns = [
            # Script injection
            r'<script[^>]*>.*?</script>',
            r'javascript\s*:',
            r'vbscript\s*:',
            r'data\s*:\s*text/html',
            
            # SQL injection
            r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b.*\b(FROM|INTO|SET|WHERE|TABLE|DATABASE)\b',
            r'(\'\s*OR\s*\'|\'\s*AND\s*\')',
            r'(\bUNION\b.*\bSELECT\b)',
            
            # Command injection (more specific patterns)
            r'(\$\([^)]*\)|`[^`]*`)',
            r'(\|\s*[a-zA-Z]+\s*;|\&\&\s*[a-zA-Z]+|\|\|\s*[a-zA-Z]+)',
            r'(;\s*(cat|ls|pwd|whoami|id|nc|netcat|wget|curl)\s+)',
            
            # Path traversal
            r'(\.\./|\.\.\x5c)',
            r'(%2e%2e%2f|%2e%2e%5c)',
            
            # XML/XXE
            r'<!ENTITY',
            r'<!DOCTYPE.*ENTITY',
            r'SYSTEM\s+[\'"]file:',
            
            # LDAP injection
            r'(\)\(\||\)\(&|\*\)\()',
        ]
        
        # Suspicious patterns (warn but may allow)
        self.suspicious_patterns = [
            r'<[^>]*on\w+\s*=',  # Event handlers
            r'<(iframe|object|embed|applet)',  # Embedded content
            r'\b(eval|setTimeout|setInterval)\s*\(',  # Dynamic execution
            r'(expression\s*\(|@import)',  # CSS injection
        ]
        
        # Compile patterns for performance
        self.compiled_critical = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.critical_patterns]
        self.compiled_suspicious = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.suspicious_patterns]
        
        # Character limits by content type
        self.max_lengths = {
            ContentType.TEXT: 10000,
            ContentType.HTML: 50000,
            ContentType.EMAIL: 254,
            ContentType.URL: 2048,
            ContentType.FILENAME: 255,
            ContentType.JSON: 1000000,  # 1MB
            ContentType.SQL_IDENTIFIER: 128,
            ContentType.API_KEY: 512,
            ContentType.UUID: 36,
            ContentType.BASE64: 100000,
            ContentType.MARKDOWN: 100000
        }
        
        # Allowed characters by content type
        self.allowed_chars = {
            ContentType.SQL_IDENTIFIER: re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$'),
            ContentType.UUID: re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE),
            ContentType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            ContentType.FILENAME: re.compile(r'^[a-zA-Z0-9._-]+$'),
        }
        
        # File extension whitelist
        self.safe_extensions = {
            'text': {'.txt', '.md', '.rst', '.log'},
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'},
            'document': {'.pdf', '.doc', '.docx', '.odt', '.rtf'},
            'data': {'.json', '.xml', '.csv', '.yaml', '.yml'},
            'code': {'.py', '.js', '.html', '.css', '.sql', '.sh'},
        }
        
        logger.info(f"Security validator initialized with {validation_level.value} level")
    
    def validate_input(self, value: Any, content_type: ContentType, 
                      field_name: str = "input", required: bool = True) -> ValidationResult:
        """Comprehensive input validation and sanitization"""
        
        original_value = value
        issues = []
        severity = "low"
        
        try:
            # Handle None/empty values
            if value is None or (isinstance(value, str) and not value.strip()):
                if required:
                    return ValidationResult(
                        is_valid=False,
                        sanitized_value=None,
                        original_value=original_value,
                        issues=[f"{field_name} is required"],
                        severity="medium"
                    )
                return ValidationResult(
                    is_valid=True,
                    sanitized_value=None,
                    original_value=original_value,
                    issues=[],
                    severity="low"
                )
            
            # Convert to string for processing
            if not isinstance(value, str):
                try:
                    value = str(value)
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        sanitized_value=None,
                        original_value=original_value,
                        issues=[f"Cannot convert {field_name} to string: {str(e)}"],
                        severity="high"
                    )
            
            # Check length limits
            max_length = self.max_lengths.get(content_type, 10000)
            if len(value) > max_length:
                issues.append(f"{field_name} exceeds maximum length of {max_length} characters")
                severity = "medium"
                if self.validation_level == ValidationLevel.STRICT:
                    return ValidationResult(
                        is_valid=False,
                        sanitized_value=None,
                        original_value=original_value,
                        issues=issues,
                        severity=severity
                    )
                # Truncate if not strict
                value = value[:max_length]
            
            # Check for critical security patterns
            critical_issues = self._check_critical_patterns(value, field_name)
            if critical_issues:
                issues.extend(critical_issues)
                severity = "critical"
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=None,
                    original_value=original_value,
                    issues=issues,
                    severity=severity
                )
            
            # Check for suspicious patterns
            suspicious_issues = self._check_suspicious_patterns(value, field_name)
            if suspicious_issues:
                issues.extend(suspicious_issues)
                severity = "medium"
                if self.validation_level == ValidationLevel.STRICT:
                    return ValidationResult(
                        is_valid=False,
                        sanitized_value=None,
                        original_value=original_value,
                        issues=issues,
                        severity=severity
                    )
            
            # Content-specific validation and sanitization
            sanitized_value = self._sanitize_by_content_type(value, content_type, field_name)
            if sanitized_value is None:
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=None,
                    original_value=original_value,
                    issues=[f"{field_name} failed content-specific validation"],
                    severity="high"
                )
            
            # Check if sanitization changed the value
            if sanitized_value != original_value:
                issues.append(f"{field_name} was sanitized")
                if severity == "low":
                    severity = "medium"
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=sanitized_value,
                original_value=original_value,
                issues=issues,
                severity=severity
            )
            
        except Exception as e:
            logger.error(f"Validation error for {field_name}: {e}")
            return ValidationResult(
                is_valid=False,
                sanitized_value=None,
                original_value=original_value,
                issues=[f"Validation error: {str(e)}"],
                severity="critical"
            )
    
    def _check_critical_patterns(self, value: str, field_name: str) -> List[str]:
        """Check for critical security patterns"""
        issues = []
        
        for pattern in self.compiled_critical:
            if pattern.search(value):
                issues.append(f"{field_name} contains critical security pattern: {pattern.pattern[:50]}...")
                logger.warning(f"Critical pattern detected in {field_name}: {pattern.pattern}")
        
        return issues
    
    def _check_suspicious_patterns(self, value: str, field_name: str) -> List[str]:
        """Check for suspicious patterns"""
        issues = []
        
        for pattern in self.compiled_suspicious:
            if pattern.search(value):
                issues.append(f"{field_name} contains suspicious pattern: {pattern.pattern[:50]}...")
                logger.info(f"Suspicious pattern detected in {field_name}: {pattern.pattern}")
        
        return issues
    
    def _sanitize_by_content_type(self, value: str, content_type: ContentType, field_name: str) -> Optional[str]:
        """Content-specific sanitization"""
        
        try:
            if content_type == ContentType.TEXT:
                return self._sanitize_text(value)
            
            elif content_type == ContentType.HTML:
                return self._sanitize_html(value)
            
            elif content_type == ContentType.EMAIL:
                return self._validate_email(value)
            
            elif content_type == ContentType.URL:
                return self._validate_url(value)
            
            elif content_type == ContentType.FILENAME:
                return self._sanitize_filename(value)
            
            elif content_type == ContentType.JSON:
                return self._validate_json(value)
            
            elif content_type == ContentType.SQL_IDENTIFIER:
                return self._validate_sql_identifier(value)
            
            elif content_type == ContentType.API_KEY:
                return self._validate_api_key(value)
            
            elif content_type == ContentType.UUID:
                return self._validate_uuid(value)
            
            elif content_type == ContentType.BASE64:
                return self._validate_base64(value)
            
            elif content_type == ContentType.MARKDOWN:
                return self._sanitize_markdown(value)
            
            else:
                return self._sanitize_text(value)
                
        except Exception as e:
            logger.error(f"Content-specific sanitization failed for {field_name}: {e}")
            return None
    
    def _sanitize_text(self, value: str) -> str:
        """Basic text sanitization"""
        # Remove null bytes and control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
        
        # HTML encode dangerous characters
        value = html.escape(value, quote=True)
        
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value).strip()
        
        return value
    
    def _sanitize_html(self, value: str) -> str:
        """HTML sanitization (basic - consider using bleach for production)"""
        # For now, escape all HTML
        # In production, use a library like bleach for proper HTML sanitization
        return html.escape(value, quote=True)
    
    def _validate_email(self, value: str) -> Optional[str]:
        """Email validation"""
        value = value.strip().lower()
        
        if not self.allowed_chars[ContentType.EMAIL].match(value):
            return None
        
        # Additional email checks
        if value.count('@') != 1:
            return None
        
        local, domain = value.split('@')
        if len(local) > 64 or len(domain) > 253:
            return None
        
        return value
    
    def _validate_url(self, value: str) -> Optional[str]:
        """URL validation"""
        value = value.strip()
        
        try:
            parsed = urlparse(value)
            
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return None
            
            # Only allow safe schemes
            if parsed.scheme.lower() not in {'http', 'https', 'ftp', 'ftps'}:
                return None
            
            # Check for suspicious domains
            if any(dangerous in parsed.netloc.lower() for dangerous in ['localhost', '127.0.0.1', '0.0.0.0']):
                if self.validation_level == ValidationLevel.STRICT:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _sanitize_filename(self, value: str) -> Optional[str]:
        """Filename sanitization"""
        value = value.strip()
        
        # Remove dangerous characters
        value = re.sub(r'[<>:"/\\|?*]', '', value)
        
        # Remove null bytes and control characters
        value = ''.join(char for char in value if ord(char) >= 32)
        
        # Check against allowed pattern
        if not self.allowed_chars[ContentType.FILENAME].match(value):
            return None
        
        # Check file extension
        if '.' in value:
            ext = '.' + value.split('.')[-1].lower()
            all_safe_exts = set()
            for exts in self.safe_extensions.values():
                all_safe_exts.update(exts)
            
            if ext not in all_safe_exts and self.validation_level == ValidationLevel.STRICT:
                return None
        
        return value
    
    def _validate_json(self, value: str) -> Optional[str]:
        """JSON validation"""
        try:
            # Parse and re-serialize to normalize
            parsed = json.loads(value)
            return json.dumps(parsed, separators=(',', ':'))
        except (json.JSONDecodeError, TypeError):
            return None
    
    def _validate_sql_identifier(self, value: str) -> Optional[str]:
        """SQL identifier validation"""
        value = value.strip()
        
        if not self.allowed_chars[ContentType.SQL_IDENTIFIER].match(value):
            return None
        
        # Check against SQL reserved words
        sql_reserved = {
            'select', 'insert', 'update', 'delete', 'drop', 'create', 'alter',
            'table', 'database', 'index', 'view', 'procedure', 'function',
            'user', 'grant', 'revoke', 'commit', 'rollback', 'transaction'
        }
        
        if value.lower() in sql_reserved:
            return None
        
        return value
    
    def _validate_api_key(self, value: str) -> Optional[str]:
        """API key validation"""
        value = value.strip()
        
        # Basic API key format checks
        if len(value) < 16 or len(value) > 512:
            return None
        
        # Should contain only safe characters
        if not re.match(r'^[a-zA-Z0-9._-]+$', value):
            return None
        
        return value
    
    def _validate_uuid(self, value: str) -> Optional[str]:
        """UUID validation"""
        value = value.strip().lower()
        
        if not self.allowed_chars[ContentType.UUID].match(value):
            return None
        
        return value
    
    def _validate_base64(self, value: str) -> Optional[str]:
        """Base64 validation"""
        value = value.strip()
        
        try:
            # Try to decode and re-encode
            decoded = base64.b64decode(value, validate=True)
            return base64.b64encode(decoded).decode('ascii')
        except (binascii.Error, ValueError):
            return None
    
    def _sanitize_markdown(self, value: str) -> str:
        """Markdown sanitization"""
        # Basic markdown sanitization
        # Remove dangerous HTML tags while preserving markdown
        
        # Remove script tags
        value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: links
        value = re.sub(r'javascript\s*:', '', value, flags=re.IGNORECASE)
        
        # Remove dangerous event handlers
        value = re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', '', value, flags=re.IGNORECASE)
        
        return value


class RequestValidator:
    """High-level request validation for API endpoints"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validator = SecurityValidator(validation_level)
        self.validation_level = validation_level
    
    def validate_request_data(self, data: Dict[str, Any], 
                            field_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate entire request payload
        
        Args:
            data: Request data to validate
            field_rules: Validation rules per field
                Format: {
                    "field_name": {
                        "content_type": ContentType.TEXT,
                        "required": True,
                        "max_length": 1000
                    }
                }
        
        Returns:
            Validated and sanitized data
            
        Raises:
            HTTPException: If validation fails
        """
        
        validated_data = {}
        all_issues = []
        
        # Validate each field
        for field_name, rules in field_rules.items():
            content_type = rules.get('content_type', ContentType.TEXT)
            required = rules.get('required', False)
            
            value = data.get(field_name)
            
            result = self.validator.validate_input(
                value, content_type, field_name, required
            )
            
            if not result.is_valid:
                if result.severity in ['critical', 'high']:
                    # Immediately reject critical/high severity issues
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Validation failed",
                            "field": field_name,
                            "issues": result.issues,
                            "severity": result.severity
                        }
                    )
                else:
                    all_issues.extend([f"{field_name}: {issue}" for issue in result.issues])
            
            validated_data[field_name] = result.sanitized_value
            
            if result.issues and result.severity in ['medium', 'high']:
                logger.warning(f"Validation issues for {field_name}: {result.issues}")
        
        # Check for any medium severity issues in strict mode
        if all_issues and self.validation_level == ValidationLevel.STRICT:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Validation failed",
                    "issues": all_issues
                }
            )
        
        return validated_data
    
    def validate_query_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """Validate query parameters"""
        validated_params = {}
        
        for param_name, param_value in params.items():
            result = self.validator.validate_input(
                param_value, ContentType.TEXT, param_name, required=False
            )
            
            if not result.is_valid and result.severity in ['critical', 'high']:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid query parameter",
                        "parameter": param_name,
                        "issues": result.issues
                    }
                )
            
            validated_params[param_name] = result.sanitized_value
        
        return validated_params
    
    def validate_headers(self, headers: Dict[str, str], 
                        required_headers: Set[str] = None) -> Dict[str, str]:
        """Validate HTTP headers"""
        validated_headers = {}
        required_headers = required_headers or set()
        
        for header_name, header_value in headers.items():
            # Skip certain headers that are handled by the framework
            if header_name.lower() in {'host', 'user-agent', 'accept-encoding'}:
                validated_headers[header_name] = header_value
                continue
            
            result = self.validator.validate_input(
                header_value, ContentType.TEXT, header_name, required=False
            )
            
            if not result.is_valid and result.severity in ['critical', 'high']:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid header",
                        "header": header_name,
                        "issues": result.issues
                    }
                )
            
            validated_headers[header_name] = result.sanitized_value
        
        # Check required headers
        for required_header in required_headers:
            if required_header not in validated_headers:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Missing required header",
                        "header": required_header
                    }
                )
        
        return validated_headers


# Global validator instances
strict_validator = RequestValidator(ValidationLevel.STRICT)
moderate_validator = RequestValidator(ValidationLevel.MODERATE)
permissive_validator = RequestValidator(ValidationLevel.PERMISSIVE)


# Utility functions for easy integration
def validate_text(text: str, max_length: int = 10000, required: bool = True) -> str:
    """Quick text validation"""
    validator = SecurityValidator()
    result = validator.validate_input(text, ContentType.TEXT, "text", required)
    if not result.is_valid:
        raise ValueError(f"Text validation failed: {result.issues}")
    return result.sanitized_value


def validate_email(email: str) -> str:
    """Quick email validation"""
    validator = SecurityValidator()
    result = validator.validate_input(email, ContentType.EMAIL, "email", True)
    if not result.is_valid:
        raise ValueError(f"Email validation failed: {result.issues}")
    return result.sanitized_value


def validate_url(url: str) -> str:
    """Quick URL validation"""
    validator = SecurityValidator()
    result = validator.validate_input(url, ContentType.URL, "url", True)
    if not result.is_valid:
        raise ValueError(f"URL validation failed: {result.issues}")
    return result.sanitized_value


def sanitize_filename(filename: str) -> str:
    """Quick filename sanitization"""
    validator = SecurityValidator()
    result = validator.validate_input(filename, ContentType.FILENAME, "filename", True)
    if not result.is_valid:
        raise ValueError(f"Filename validation failed: {result.issues}")
    return result.sanitized_value