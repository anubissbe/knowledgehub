"""
Enhanced Input Validation and Sanitization

Extends the existing validation system with advanced pattern detection,
context-aware validation, and performance optimizations.
"""

import re
import json
import time
import asyncio
import ipaddress
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timedelta
import logging

from .validation import (
    SecurityValidator,
    ValidationLevel,
    ContentType,
    ValidationResult,
    RequestValidator
)
from .monitoring import log_security_event, SecurityEventType, ThreatLevel

logger = logging.getLogger(__name__)


class AdvancedThreatType(str, Enum):
    """Advanced threat types for enhanced detection"""
    GRAPHQL_INJECTION = "graphql_injection"
    SSRF_ATTEMPT = "ssrf_attempt"
    XXE_INJECTION = "xxe_injection"
    TEMPLATE_INJECTION = "template_injection"
    NOSQL_INJECTION = "nosql_injection"
    LDAP_ADVANCED = "ldap_advanced"
    UNICODE_BYPASS = "unicode_bypass"
    POLYGLOT_PAYLOAD = "polyglot_payload"


@dataclass
class ValidationContext:
    """Context information for validation decisions"""
    user_role: Optional[str] = None
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    source_country: Optional[str] = None
    request_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    is_authenticated: bool = False
    risk_score: float = 0.0


class EnhancedSecurityValidator(SecurityValidator):
    """Enhanced security validator with advanced pattern detection"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        super().__init__(validation_level)
        
        # Advanced attack patterns
        self.advanced_patterns = {
            AdvancedThreatType.GRAPHQL_INJECTION: [
                r'\{[^}]*__schema[^}]*\}',
                r'mutation\s*\{[^}]*\}',
                r'query\s+IntrospectionQuery',
                r'__typename',
                r'fragment\s+\w+\s+on\s+\w+',
            ],
            AdvancedThreatType.SSRF_ATTEMPT: [
                r'(gopher|dict|ftp|sftp|jar|netdoc)://',
                r'(metadata\.google|169\.254\.169\.254|localhost|127\.0\.0\.1)',
                r'file:///proc/self',
                r'http://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+',
                r'@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+',
            ],
            AdvancedThreatType.XXE_INJECTION: [
                r'<!ENTITY\s+\S+\s+(SYSTEM|PUBLIC)',
                r'<!DOCTYPE[^>]*\[',
                r'xmlns:xi\s*=',
                r'<xi:include',
                r'SYSTEM\s+"file:',
            ],
            AdvancedThreatType.TEMPLATE_INJECTION: [
                r'\{\{[^}]*\}\}',
                r'\{%[^%]*%\}',
                r'#set\s*\(',
                r'\$\{[^}]*\}',
                r'<%[^%]*%>',
                r'<#[^#]*#>',
                r'freemarker\.',
                r'velocity\.',
            ],
            AdvancedThreatType.NOSQL_INJECTION: [
                r'\$where\s*:',
                r'\$ne\s*:',
                r'\$gt\s*:',
                r'\$regex\s*:',
                r'"\s*:\s*\{\s*"\$',
                r'mapReduce\s*:',
                r'function\s*\(\s*\)\s*\{',
            ],
            AdvancedThreatType.LDAP_ADVANCED: [
                r'\(\|\(',
                r'\(&\(',
                r'\)\)\(',
                r'objectClass=\*',
                r'cn=\*\)\(',
            ],
            AdvancedThreatType.UNICODE_BYPASS: [
                r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
                r'%c0%ae',  # URL encoded path traversal
                r'%252e%252e',  # Double URL encoding
                r'\\u0027',  # Unicode apostrophe
                r'\\x3c',  # Hex encoded <
            ],
            AdvancedThreatType.POLYGLOT_PAYLOAD: [
                r'javascript:.*<svg',
                r'"><script>.*</script>',
                r"'><script>.*</script>",
                r'javascript:/*-/*`/*\\\`/*\'/*"',
            ]
        }
        
        # Compile advanced patterns
        self.compiled_advanced = {}
        for threat_type, patterns in self.advanced_patterns.items():
            self.compiled_advanced[threat_type] = [
                re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns
            ]
        
        # Context-based thresholds
        self.context_thresholds = {
            'high_risk_countries': {'CN', 'RU', 'KP', 'IR'},
            'max_requests_per_minute': 60,
            'suspicious_user_agents': {
                'sqlmap', 'nikto', 'nmap', 'burp', 'zap', 'acunetix',
                'nessus', 'metasploit', 'havij', 'wget', 'curl'
            },
            'business_hours': (8, 18),  # 8 AM to 6 PM
        }
        
        # Performance optimization: LRU cache for validation results
        self._validation_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info(f"Enhanced security validator initialized with {validation_level.value} level")
    
    def validate_with_context(self, value: Any, content_type: ContentType,
                            field_name: str, context: ValidationContext,
                            required: bool = True) -> ValidationResult:
        """Validate input with context awareness"""
        
        # Check cache first
        cache_key = self._get_cache_key(value, content_type, field_name)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Perform base validation
        result = self.validate_input(value, content_type, field_name, required)
        
        # If base validation passed, perform context-aware checks
        if result.is_valid:
            context_issues = self._validate_context(value, field_name, context)
            if context_issues:
                result.issues.extend(context_issues)
                result.is_valid = False
                result.severity = "high"
        
        # Check advanced patterns
        if result.is_valid and isinstance(value, str):
            advanced_issues = self._check_advanced_patterns(value, field_name)
            if advanced_issues:
                result.issues.extend(advanced_issues)
                result.is_valid = False
                result.severity = "critical"
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        return result
    
    def _validate_context(self, value: Any, field_name: str,
                         context: ValidationContext) -> List[str]:
        """Perform context-aware validation"""
        issues = []
        
        # Check user role restrictions
        if context.user_role == 'guest' and isinstance(value, str):
            # Stricter validation for guest users
            if len(value) > 500:
                issues.append(f"{field_name}: Guest users limited to 500 characters")
            
            # No file paths for guests
            if re.search(r'[/\\]', value):
                issues.append(f"{field_name}: Path characters not allowed for guest users")
        
        # Check geographic restrictions
        if context.source_country in self.context_thresholds['high_risk_countries']:
            # Enhanced validation for high-risk countries
            if isinstance(value, str) and re.search(r'[<>]', value):
                issues.append(f"{field_name}: Special characters restricted for your region")
        
        # Check request rate
        if context.request_count > self.context_thresholds['max_requests_per_minute']:
            issues.append(f"{field_name}: Rate limit exceeded")
        
        # Check suspicious user agents
        if context.user_agent:
            user_agent_lower = context.user_agent.lower()
            for suspicious in self.context_thresholds['suspicious_user_agents']:
                if suspicious in user_agent_lower:
                    issues.append(f"{field_name}: Suspicious user agent detected")
                    break
        
        # Time-based validation
        current_hour = datetime.now().hour
        business_start, business_end = self.context_thresholds['business_hours']
        
        if not (business_start <= current_hour < business_end):
            # After hours - extra validation
            if context.user_role != 'admin' and isinstance(value, str):
                if re.search(r'(DROP|DELETE|TRUNCATE)', value, re.IGNORECASE):
                    issues.append(f"{field_name}: Destructive operations not allowed after hours")
        
        return issues
    
    def _check_advanced_patterns(self, value: str, field_name: str) -> List[str]:
        """Check for advanced attack patterns"""
        issues = []
        
        for threat_type, patterns in self.compiled_advanced.items():
            for pattern in patterns:
                if pattern.search(value):
                    issues.append(
                        f"{field_name} contains {threat_type.value} pattern: {pattern.pattern[:50]}..."
                    )
                    logger.warning(
                        f"Advanced threat detected - Type: {threat_type.value}, "
                        f"Field: {field_name}, Pattern: {pattern.pattern}"
                    )
                    break  # One match per threat type is enough
        
        return issues
    
    def _get_cache_key(self, value: Any, content_type: ContentType,
                      field_name: str) -> str:
        """Generate cache key for validation result"""
        # Create a hash of the value for the cache key
        import hashlib
        value_str = str(value)[:1000]  # Limit to prevent huge keys
        value_hash = hashlib.md5(value_str.encode()).hexdigest()
        return f"{content_type.value}:{field_name}:{value_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[ValidationResult]:
        """Get cached validation result if available"""
        if cache_key in self._validation_cache:
            cached_data = self._validation_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self._cache_ttl:
                return cached_data['result']
            else:
                # Expired
                del self._validation_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: ValidationResult):
        """Cache validation result"""
        self._validation_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Limit cache size
        if len(self._validation_cache) > 10000:
            # Remove oldest entries
            sorted_keys = sorted(
                self._validation_cache.keys(),
                key=lambda k: self._validation_cache[k]['timestamp']
            )
            for key in sorted_keys[:5000]:  # Remove half
                del self._validation_cache[key]


class ContextAwareRequestValidator(RequestValidator):
    """Request validator with context awareness"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        super().__init__(validation_level)
        self.enhanced_validator = EnhancedSecurityValidator(validation_level)
    
    async def validate_request_with_context(
        self,
        data: Dict[str, Any],
        field_rules: Dict[str, Dict[str, Any]],
        context: ValidationContext
    ) -> Dict[str, Any]:
        """Validate request data with context information"""
        
        validated_data = {}
        all_issues = []
        high_severity_found = False
        
        # Validate each field with context
        for field_name, rules in field_rules.items():
            content_type = rules.get('content_type', ContentType.TEXT)
            required = rules.get('required', False)
            
            value = data.get(field_name)
            
            result = self.enhanced_validator.validate_with_context(
                value, content_type, field_name, context, required
            )
            
            if not result.is_valid:
                if result.severity in ['critical', 'high']:
                    high_severity_found = True
                    
                    # Log security event
                    await log_security_event(
                        SecurityEventType.INJECTION_ATTEMPT,
                        ThreatLevel.HIGH,
                        context.source_ip or "unknown",
                        context.user_agent or "",
                        context.endpoint or "",
                        context.method or "GET",
                        f"Advanced validation failed for {field_name}: {result.issues}"
                    )
                
                all_issues.extend([f"{field_name}: {issue}" for issue in result.issues])
            
            validated_data[field_name] = result.sanitized_value
        
        # Reject if high severity issues found
        if high_severity_found:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Security validation failed",
                    "issues": all_issues,
                    "severity": "high"
                }
            )
        
        # In strict mode, reject any issues
        if all_issues and self.validation_level == ValidationLevel.STRICT:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Validation failed",
                    "issues": all_issues
                }
            )
        
        return validated_data


class ValidationMetrics:
    """Track validation metrics for monitoring"""
    
    def __init__(self):
        self.total_validations = 0
        self.validation_failures = {}
        self.pattern_matches = {}
        self.validation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_validation(self, field_name: str, content_type: str,
                         success: bool, duration: float, issues: List[str] = None):
        """Record validation metrics"""
        self.total_validations += 1
        self.validation_times.append(duration)
        
        if not success and issues:
            for issue in issues:
                key = f"{field_name}:{issue[:50]}"
                self.validation_failures[key] = self.validation_failures.get(key, 0) + 1
    
    def record_pattern_match(self, pattern_type: str):
        """Record pattern match"""
        self.pattern_matches[pattern_type] = self.pattern_matches.get(pattern_type, 0) + 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_time = sum(self.validation_times) / len(self.validation_times) if self.validation_times else 0
        
        return {
            "total_validations": self.total_validations,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "average_validation_time_ms": avg_time * 1000,
            "top_validation_failures": sorted(
                self.validation_failures.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "pattern_match_counts": self.pattern_matches
        }


# Global instances
validation_metrics = ValidationMetrics()


# Helper functions for async validation
async def validate_parallel(values: List[Tuple[Any, ContentType, str]], 
                          context: ValidationContext = None) -> List[ValidationResult]:
    """Validate multiple values in parallel"""
    validator = EnhancedSecurityValidator()
    
    async def validate_one(value, content_type, field_name):
        return validator.validate_with_context(
            value, content_type, field_name, context or ValidationContext()
        )
    
    tasks = [
        validate_one(value, content_type, field_name)
        for value, content_type, field_name in values
    ]
    
    return await asyncio.gather(*tasks)


@lru_cache(maxsize=1000)
def is_private_ip(ip_str: str) -> bool:
    """Check if IP address is private (cached)"""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False


def detect_encoding_bypass(value: str) -> bool:
    """Detect various encoding bypass attempts"""
    # Check for multiple encoding
    if '%25' in value:  # Double URL encoding
        return True
    
    # Check for Unicode/UTF-8 bypass attempts
    suspicious_unicode = [
        '\u0027',  # Unicode apostrophe
        '\u003c',  # Unicode <
        '\u003e',  # Unicode >
        '\uff1c',  # Fullwidth <
        '\uff1e',  # Fullwidth >
    ]
    
    return any(char in value for char in suspicious_unicode)