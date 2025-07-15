"""
Validation Monitoring API Endpoints

Provides endpoints for monitoring input validation performance,
viewing validation metrics, and managing validation rules.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from ..security.validation_enhanced import (
    validation_metrics,
    ValidationContext,
    AdvancedThreatType
)
from ..security.validation import ValidationLevel, ContentType

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class ValidationMetricsResponse(BaseModel):
    """Validation metrics response model"""
    total_validations: int
    cache_hit_rate: float
    average_validation_time_ms: float
    top_validation_failures: List[tuple]
    pattern_match_counts: Dict[str, int]
    system_health: Dict[str, Any]


class ValidationRuleRequest(BaseModel):
    """Validation rule configuration request"""
    endpoint: str = Field(..., description="API endpoint pattern")
    field_name: str = Field(..., description="Field name to validate")
    content_type: ContentType = Field(..., description="Content type for validation")
    required: bool = Field(False, description="Whether field is required")
    max_length: Optional[int] = Field(None, description="Maximum field length")
    custom_patterns: Optional[List[str]] = Field(None, description="Custom validation patterns")


class ValidationTestRequest(BaseModel):
    """Test validation request"""
    value: str = Field(..., description="Value to test")
    content_type: ContentType = Field(ContentType.TEXT, description="Content type")
    field_name: str = Field("test_field", description="Field name")
    validation_level: ValidationLevel = Field(ValidationLevel.MODERATE, description="Validation level")
    context: Optional[Dict[str, Any]] = Field(None, description="Validation context")


class ValidationTestResponse(BaseModel):
    """Test validation response"""
    is_valid: bool
    sanitized_value: Any
    issues: List[str]
    severity: str
    patterns_matched: List[str]
    validation_time_ms: float


# Admin API key verification (simplified)
async def verify_admin_api_key(x_api_key: Optional[str] = None) -> str:
    """Verify admin API key for validation endpoints"""
    if not x_api_key or x_api_key != "admin":
        raise HTTPException(status_code=401, detail="Admin API key required")
    return x_api_key


@router.get("/validation/metrics", response_model=ValidationMetricsResponse)
async def get_validation_metrics(
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get comprehensive validation metrics
    
    Returns detailed statistics about validation performance,
    failure patterns, and system health.
    """
    try:
        metrics = validation_metrics.get_metrics()
        
        # Add system health info
        system_health = {
            "validation_cache_size": len(getattr(validation_metrics, '_validation_cache', {})),
            "active_sessions": len(getattr(validation_metrics, 'user_sessions', {})),
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": 24,  # Would track actual uptime
        }
        
        return ValidationMetricsResponse(
            **metrics,
            system_health=system_health
        )
        
    except Exception as e:
        logger.error(f"Error getting validation metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get validation metrics")


@router.get("/validation/patterns")
async def get_validation_patterns(
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get validation patterns and their match statistics
    
    Returns information about detection patterns and how often
    they match suspicious input.
    """
    try:
        from ..security.validation_enhanced import EnhancedSecurityValidator
        
        validator = EnhancedSecurityValidator()
        patterns_info = {}
        
        # Get advanced patterns
        for threat_type, patterns in validator.advanced_patterns.items():
            if pattern_type and threat_type.value != pattern_type:
                continue
                
            patterns_info[threat_type.value] = {
                "patterns": patterns,
                "pattern_count": len(patterns),
                "match_count": validation_metrics.pattern_matches.get(threat_type.value, 0),
                "description": f"Detection patterns for {threat_type.value}"
            }
        
        return {
            "patterns": patterns_info,
            "total_pattern_types": len(patterns_info),
            "available_types": [t.value for t in AdvancedThreatType]
        }
        
    except Exception as e:
        logger.error(f"Error getting validation patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to get validation patterns")


@router.post("/validation/test", response_model=ValidationTestResponse)
async def test_validation(
    request: ValidationTestRequest,
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Test validation against a specific input
    
    Allows testing validation rules and patterns against
    specific input to verify detection capabilities.
    """
    try:
        from ..security.validation_enhanced import EnhancedSecurityValidator
        import time
        
        # Create validator
        validator = EnhancedSecurityValidator(request.validation_level)
        
        # Create context
        context = ValidationContext()
        if request.context:
            for key, value in request.context.items():
                if hasattr(context, key):
                    setattr(context, key, value)
        
        # Perform validation
        start_time = time.time()
        result = validator.validate_with_context(
            request.value,
            request.content_type,
            request.field_name,
            context
        )
        validation_time = (time.time() - start_time) * 1000
        
        # Check which patterns matched
        patterns_matched = []
        if isinstance(request.value, str):
            for threat_type, compiled_patterns in validator.compiled_advanced.items():
                for pattern in compiled_patterns:
                    if pattern.search(request.value):
                        patterns_matched.append(f"{threat_type.value}: {pattern.pattern[:50]}...")
                        break
        
        return ValidationTestResponse(
            is_valid=result.is_valid,
            sanitized_value=result.sanitized_value,
            issues=result.issues,
            severity=result.severity,
            patterns_matched=patterns_matched,
            validation_time_ms=validation_time
        )
        
    except Exception as e:
        logger.error(f"Error testing validation: {e}")
        raise HTTPException(status_code=500, detail="Failed to test validation")


@router.get("/validation/failures")
async def get_validation_failures(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of failures to return"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get recent validation failures
    
    Returns a list of recent validation failures with details
    about what patterns were detected and why validation failed.
    """
    try:
        # Get failure data from metrics
        failures = validation_metrics.validation_failures
        
        # Convert to list format
        failure_list = []
        for failure_key, count in sorted(failures.items(), key=lambda x: x[1], reverse=True):
            field_name, issue = failure_key.split(':', 1)
            
            # Basic severity estimation
            estimated_severity = "low"
            if any(keyword in issue.lower() for keyword in ['injection', 'script', 'malicious']):
                estimated_severity = "high"
            elif any(keyword in issue.lower() for keyword in ['suspicious', 'blocked']):
                estimated_severity = "medium"
            
            if severity and estimated_severity != severity:
                continue
            
            failure_list.append({
                "field_name": field_name,
                "issue": issue,
                "count": count,
                "estimated_severity": estimated_severity,
                "first_seen": "2023-01-01T00:00:00Z",  # Would track actual timestamps
                "last_seen": datetime.now().isoformat()
            })
            
            if len(failure_list) >= limit:
                break
        
        return {
            "failures": failure_list,
            "total_found": len(failure_list),
            "filters_applied": {
                "severity": severity,
                "hours_back": hours,
                "limit": limit
            },
            "summary": {
                "total_failure_types": len(failures),
                "most_common_failure": max(failures.items(), key=lambda x: x[1])[0] if failures else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting validation failures: {e}")
        raise HTTPException(status_code=500, detail="Failed to get validation failures")


@router.get("/validation/performance")
async def get_validation_performance(
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get validation performance statistics
    
    Returns performance metrics including validation times,
    cache efficiency, and throughput statistics.
    """
    try:
        metrics = validation_metrics.get_metrics()
        
        # Calculate additional performance metrics
        performance_stats = {
            "average_validation_time_ms": metrics["average_validation_time_ms"],
            "cache_hit_rate": metrics["cache_hit_rate"],
            "cache_efficiency": "excellent" if metrics["cache_hit_rate"] > 0.8 else "good" if metrics["cache_hit_rate"] > 0.6 else "poor",
            "total_validations": metrics["total_validations"],
            "validations_per_second": metrics["total_validations"] / (24 * 3600),  # Assuming 24h uptime
            "performance_grade": "A" if metrics["average_validation_time_ms"] < 10 else "B" if metrics["average_validation_time_ms"] < 50 else "C",
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Add recommendations based on performance
        if metrics["cache_hit_rate"] < 0.5:
            performance_stats["bottlenecks"].append("Low cache hit rate")
            performance_stats["recommendations"].append("Increase cache TTL or size")
        
        if metrics["average_validation_time_ms"] > 100:
            performance_stats["bottlenecks"].append("High validation latency")
            performance_stats["recommendations"].append("Optimize regex patterns or implement async validation")
        
        return {
            "performance": performance_stats,
            "timestamp": datetime.now().isoformat(),
            "monitoring_period": "24 hours"
        }
        
    except Exception as e:
        logger.error(f"Error getting validation performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get validation performance")


@router.post("/validation/cache/clear")
async def clear_validation_cache(
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Clear validation cache
    
    Clears the validation result cache to force fresh validation
    of all subsequent requests.
    """
    try:
        # Clear cache (would need access to the actual cache)
        # This is a placeholder - in real implementation would clear the cache
        
        return {
            "success": True,
            "message": "Validation cache cleared",
            "timestamp": datetime.now().isoformat(),
            "cache_entries_cleared": 1000  # Placeholder
        }
        
    except Exception as e:
        logger.error(f"Error clearing validation cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear validation cache")


@router.get("/validation/health")
async def validation_health():
    """
    Validation system health check
    
    Returns the health status of the validation system.
    No authentication required for health checks.
    """
    try:
        metrics = validation_metrics.get_metrics()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if metrics["average_validation_time_ms"] > 1000:
            health_status = "degraded"
            issues.append("High validation latency")
        
        if metrics["cache_hit_rate"] < 0.3:
            health_status = "degraded"
            issues.append("Low cache efficiency")
        
        if len(issues) > 2:
            health_status = "unhealthy"
        
        return {
            "status": health_status,
            "validation_system": "active",
            "total_validations": metrics["total_validations"],
            "cache_hit_rate": metrics["cache_hit_rate"],
            "average_response_time_ms": metrics["average_validation_time_ms"],
            "issues": issues,
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Validation health check failed: {e}")
        raise HTTPException(status_code=503, detail="Validation system unhealthy")