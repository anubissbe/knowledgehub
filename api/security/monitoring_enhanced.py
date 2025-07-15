"""
Enhanced Security Monitoring with Metrics and Alerting Integration

This module extends the existing security monitoring with Prometheus metrics
and automated alerting capabilities.
"""

import asyncio
import time
from datetime import datetime
from typing import Optional

from .monitoring import (
    SecurityMonitor as BaseSecurityMonitor,
    SecurityEvent,
    SecurityEventType,
    ThreatLevel,
    logger
)
from .metrics import (
    init_metrics_collector,
    record_security_event,
    record_auth_attempt,
    record_blocked_request,
    observe_security_check
)
from .alerting import init_alerting_system, get_alerting_system


class EnhancedSecurityMonitor(BaseSecurityMonitor):
    """Enhanced security monitor with metrics and alerting"""
    
    def __init__(self, log_directory: str = "/app/logs/security"):
        super().__init__(log_directory)
        
        # Initialize metrics collector
        self.metrics_collector = init_metrics_collector(self)
        
        # Initialize alerting system
        self.alerting_system = init_alerting_system()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Enhanced security monitoring system initialized with metrics and alerting")
    
    def _start_background_tasks(self):
        """Start background tasks for metrics and alerting"""
        # Schedule periodic metrics updates
        asyncio.create_task(self._periodic_metrics_update())
        
        # Schedule periodic alert checks
        asyncio.create_task(self._periodic_alert_check())
    
    async def _periodic_metrics_update(self):
        """Periodically update Prometheus metrics"""
        while True:
            try:
                # Update metrics every 30 seconds
                await asyncio.sleep(30)
                self.metrics_collector.update_metrics()
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
    
    async def _periodic_alert_check(self):
        """Periodically check for alert conditions"""
        while True:
            try:
                # Check alerts every minute
                await asyncio.sleep(60)
                stats = self.get_security_stats()
                await self.alerting_system.check_alerts(stats)
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
    
    async def log_event(self, event: SecurityEvent) -> None:
        """Log event with metrics recording"""
        start_time = time.time()
        
        # Call parent method
        await super().log_event(event)
        
        # Record metrics
        record_security_event(
            event.event_type.value,
            event.threat_level.value,
            event.blocked
        )
        
        # Record specific metrics based on event type
        if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
            record_auth_attempt(False)
        elif event.event_type == SecurityEventType.AUTHENTICATION_SUCCESS:
            record_auth_attempt(True)
        
        if event.blocked:
            record_blocked_request(
                event.event_type.value,
                event.endpoint
            )
        
        # Record check duration
        duration = time.time() - start_time
        observe_security_check("event_logging", duration)
        
        # Check for immediate alerts on critical events
        if event.threat_level == ThreatLevel.CRITICAL:
            stats = self.get_security_stats()
            await self.alerting_system.check_alerts(stats)
    
    async def _analyze_event(self, event: SecurityEvent) -> None:
        """Analyze event with metrics tracking"""
        start_time = time.time()
        
        # Call parent method
        await super()._analyze_event(event)
        
        # Record analysis duration
        duration = time.time() - start_time
        observe_security_check("event_analysis", duration)
    
    def get_enhanced_stats(self) -> dict:
        """Get enhanced statistics including metrics and alerts"""
        base_stats = self.get_security_stats()
        
        # Add metrics info
        base_stats["metrics"] = {
            "collector_active": self.metrics_collector is not None,
            "last_update": getattr(self.metrics_collector, 'last_update', None)
        }
        
        # Add alerting info
        if self.alerting_system:
            base_stats["alerting"] = self.alerting_system.get_alert_stats()
        
        return base_stats


# Create enhanced global instance
enhanced_security_monitor = None


def init_enhanced_monitoring(log_directory: Optional[str] = None):
    """Initialize enhanced security monitoring"""
    global enhanced_security_monitor
    log_dir = log_directory or "/app/logs/security"
    enhanced_security_monitor = EnhancedSecurityMonitor(log_dir)
    return enhanced_security_monitor


def get_enhanced_monitor():
    """Get enhanced security monitor instance"""
    global enhanced_security_monitor
    if not enhanced_security_monitor:
        init_enhanced_monitoring()
    return enhanced_security_monitor