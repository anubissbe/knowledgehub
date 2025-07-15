"""
Prometheus Metrics Endpoint

Exposes application metrics in Prometheus format.
"""

from fastapi import APIRouter, Response
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """
    Expose metrics in Prometheus format
    
    Returns all collected metrics including:
    - Security events and threats
    - Authentication attempts
    - Blocked requests
    - Performance metrics
    """
    try:
        # Generate metrics in Prometheus format
        metrics_data = generate_latest(REGISTRY)
        
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
        
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {str(e)}",
            media_type="text/plain",
            status_code=500
        )


@router.get("/metrics/health")
async def metrics_health():
    """Health check for metrics endpoint"""
    return {
        "status": "healthy",
        "endpoint": "/metrics",
        "format": "prometheus"
    }