"""
Security Metrics for Prometheus Integration

Provides Prometheus metrics for security monitoring and alerting.
"""

from prometheus_client import Counter, Gauge, Histogram, Info
from typing import Dict, Any, Optional
import time
from datetime import datetime

# Security Event Counters
security_events_total = Counter(
    'knowledgehub_security_events_total',
    'Total number of security events',
    ['event_type', 'threat_level', 'blocked']
)

authentication_attempts_total = Counter(
    'knowledgehub_auth_attempts_total',
    'Total authentication attempts',
    ['status']
)

blocked_requests_total = Counter(
    'knowledgehub_blocked_requests_total',
    'Total blocked security requests',
    ['reason', 'endpoint']
)

# Threat Metrics
active_threats_gauge = Gauge(
    'knowledgehub_active_threats',
    'Number of currently active threats',
    ['threat_type']
)

blocked_ips_gauge = Gauge(
    'knowledgehub_blocked_ips_total',
    'Number of currently blocked IP addresses'
)

suspicious_ips_gauge = Gauge(
    'knowledgehub_suspicious_ips_total',
    'Number of currently suspicious IP addresses'
)

# Performance Metrics
security_check_duration = Histogram(
    'knowledgehub_security_check_duration_seconds',
    'Time spent performing security checks',
    ['check_type']
)

# System Info
security_system_info = Info(
    'knowledgehub_security_system',
    'Security system information'
)

# Initialize system info
security_system_info.info({
    'version': '1.0.0',
    'monitoring_enabled': 'true',
    'auto_blocking_enabled': 'true'
})


class SecurityMetricsCollector:
    """Collects and updates security metrics for Prometheus"""
    
    def __init__(self, security_monitor):
        self.monitor = security_monitor
        self.last_update = time.time()
    
    def update_metrics(self):
        """Update all security metrics"""
        try:
            # Get current stats
            stats = self.monitor.get_security_stats()
            
            # Update gauges
            blocked_ips_gauge.set(stats['blocked_ips_count'])
            suspicious_ips_gauge.set(stats['suspicious_ips_count'])
            
            # Update threat gauges by type
            for event_type, count in stats['event_types_24h'].items():
                active_threats_gauge.labels(threat_type=event_type).set(count)
            
            self.last_update = time.time()
            
        except Exception as e:
            print(f"Error updating security metrics: {e}")
    
    def record_security_event(self, event_type: str, threat_level: str, blocked: bool):
        """Record a security event in metrics"""
        security_events_total.labels(
            event_type=event_type,
            threat_level=threat_level,
            blocked=str(blocked).lower()
        ).inc()
    
    def record_auth_attempt(self, success: bool):
        """Record authentication attempt"""
        status = 'success' if success else 'failure'
        authentication_attempts_total.labels(status=status).inc()
    
    def record_blocked_request(self, reason: str, endpoint: str):
        """Record a blocked request"""
        blocked_requests_total.labels(reason=reason, endpoint=endpoint).inc()
    
    def observe_security_check(self, check_type: str, duration: float):
        """Record security check duration"""
        security_check_duration.labels(check_type=check_type).observe(duration)


# Global metrics collector instance will be initialized with security monitor
metrics_collector = None


def init_metrics_collector(security_monitor):
    """Initialize the global metrics collector"""
    global metrics_collector
    metrics_collector = SecurityMetricsCollector(security_monitor)
    return metrics_collector


def get_metrics_collector():
    """Get the global metrics collector instance"""
    return metrics_collector


# Convenience functions for recording metrics
def record_security_event(event_type: str, threat_level: str, blocked: bool = False):
    """Record a security event"""
    if metrics_collector:
        metrics_collector.record_security_event(event_type, threat_level, blocked)


def record_auth_attempt(success: bool):
    """Record authentication attempt"""
    if metrics_collector:
        metrics_collector.record_auth_attempt(success)


def record_blocked_request(reason: str, endpoint: str):
    """Record blocked request"""
    if metrics_collector:
        metrics_collector.record_blocked_request(reason, endpoint)


def observe_security_check(check_type: str, duration: float):
    """Record security check duration"""
    if metrics_collector:
        metrics_collector.observe_security_check(check_type, duration)