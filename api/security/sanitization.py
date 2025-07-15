"""Input sanitization and validation utilities for security"""

import html
import re
import bleach
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

# Allowed HTML tags for rich text fields (if any)
ALLOWED_TAGS = []  # Start with no HTML allowed

# Allowed attributes for HTML tags
ALLOWED_ATTRIBUTES = {}

# Patterns for potentially dangerous content
DANGEROUS_PATTERNS = [
    # JavaScript patterns
    r'<script[^>]*>.*?</script>',
    r'javascript:',
    r'on\w+\s*=',
    r'data:text/html',
    r'vbscript:',
    
    # SQL injection patterns
    r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
    r'(\s|^)(or|and)\s+[\w\'"]+\s*=\s*[\w\'"]+',
    r'[\'"](\s)*(union|select|insert|update|delete|drop)',
    
    # Path traversal patterns
    r'\.\./|\.\.\\',
    r'/etc/passwd',
    r'/proc/',
    
    # Command injection patterns
    r'[;&|`$(){}]',
    r'\$\(',
    r'`[^`]*`',
]

class InputSanitizer:
    """Comprehensive input sanitization and validation"""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """
        Sanitize text input to prevent XSS and other attacks
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow safe HTML tags
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '').replace('\r', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if allow_html:
            # Use bleach to clean HTML while preserving safe tags
            text = bleach.clean(
                text,
                tags=ALLOWED_TAGS,
                attributes=ALLOWED_ATTRIBUTES,
                strip=True
            )
        else:
            # Escape all HTML entities
            text = html.escape(text, quote=True)
        
        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potentially dangerous pattern detected: {pattern}")
                # Replace with safe placeholder that doesn't contain the original dangerous text
                text = re.sub(pattern, '[REMOVED]', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """
        Sanitize and validate URL input
        
        Args:
            url: URL to sanitize
            
        Returns:
            Sanitized URL
            
        Raises:
            ValueError: If URL is invalid or dangerous
        """
        if not isinstance(url, str):
            url = str(url)
        
        # Basic URL validation
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
        
        # Check for dangerous patterns in URL
        dangerous_url_patterns = [
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'file:',
            r'ftp:',
        ]
        
        for pattern in dangerous_url_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous URL pattern: {pattern}")
        
        # Basic hostname validation
        if not parsed.netloc:
            raise ValueError("URL must have a valid hostname")
        
        # Prevent localhost/internal network access in production
        internal_patterns = [
            r'localhost',
            r'127\.0\.0\.1',
            r'0\.0\.0\.0',
            r'192\.168\.',
            r'10\.',
            r'172\.(1[6-9]|2[0-9]|3[0-1])\.',
        ]
        
        # Only check in production
        import os
        if os.getenv('APP_ENV') == 'production':
            for pattern in internal_patterns:
                if re.search(pattern, parsed.netloc, re.IGNORECASE):
                    raise ValueError("Internal network URLs not allowed in production")
        
        return url
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            filename = str(filename)
        
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[^\w\.-]', '_', filename)
        
        # Prevent hidden files
        if filename.startswith('.'):
            filename = 'file_' + filename[1:]
        
        # Ensure reasonable length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def sanitize_dict(data: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary values
        
        Args:
            data: Dictionary to sanitize
            max_depth: Maximum recursion depth
            
        Returns:
            Sanitized dictionary
        """
        if max_depth <= 0:
            logger.warning("Maximum sanitization depth reached")
            return {}
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = InputSanitizer.sanitize_text(str(key), max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[clean_key] = InputSanitizer.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = InputSanitizer.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, list):
                sanitized[clean_key] = InputSanitizer.sanitize_list(value, max_depth - 1)
            elif isinstance(value, (int, float, bool)) or value is None:
                sanitized[clean_key] = value
            else:
                # Convert unknown types to string and sanitize
                sanitized[clean_key] = InputSanitizer.sanitize_text(str(value))
        
        return sanitized
    
    @staticmethod
    def sanitize_list(data: List[Any], max_depth: int = 10) -> List[Any]:
        """
        Recursively sanitize list values
        
        Args:
            data: List to sanitize
            max_depth: Maximum recursion depth
            
        Returns:
            Sanitized list
        """
        if max_depth <= 0:
            logger.warning("Maximum sanitization depth reached")
            return []
        
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                sanitized.append(InputSanitizer.sanitize_text(item))
            elif isinstance(item, dict):
                sanitized.append(InputSanitizer.sanitize_dict(item, max_depth - 1))
            elif isinstance(item, list):
                sanitized.append(InputSanitizer.sanitize_list(item, max_depth - 1))
            elif isinstance(item, (int, float, bool)) or item is None:
                sanitized.append(item)
            else:
                sanitized.append(InputSanitizer.sanitize_text(str(item)))
        
        return sanitized


def validate_content_type(content_type: str) -> bool:
    """
    Validate content type header
    
    Args:
        content_type: Content-Type header value
        
    Returns:
        True if content type is allowed
    """
    allowed_types = [
        'application/json',
        'application/x-www-form-urlencoded',
        'multipart/form-data',
        'text/plain',
    ]
    
    # Extract main type (ignore parameters like charset)
    main_type = content_type.split(';')[0].strip().lower()
    
    return main_type in allowed_types


def create_security_headers() -> Dict[str, str]:
    """
    Create security headers for HTTP responses
    
    Returns:
        Dictionary of security headers
    """
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        ),
    }