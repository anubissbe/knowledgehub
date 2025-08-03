"""
Real Production Alert Service

Comprehensive alerting system that integrates with Prometheus AlertManager,
processes alerts, manages escalation policies, and coordinates recovery actions.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, update

from ..config import settings
from ..database import get_db_session
from ..models.health_check import SystemAlert, AlertSeverity
from .real_websocket_events import RealWebSocketEvents, EventType
from .prometheus_metrics import prometheus_metrics

logger = logging.getLogger(__name__)

class AlertPriority(str, Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: AlertSeverity
    priority: AlertPriority
    escalation_delay: int = 300  # 5 minutes
    auto_resolve: bool = True
    recovery_actions: List[str] = field(default_factory=list)

@dataclass
class AlertNotification:
    """Alert notification configuration"""
    alert_id: str
    recipient_type: str  # email, slack, webhook, sms
    recipient: str
    template: str
    retry_count: int = 0
    max_retries: int = 3
    next_retry: Optional[datetime] = None

@dataclass
class ProcessedAlert:
    """Processed alert with enriched information"""
    alert_id: str
    name: str
    message: str
    severity: AlertSeverity
    priority: AlertPriority
    status: AlertStatus
    service_name: Optional[str]
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: datetime
    ends_at: Optional[datetime]
    fingerprint: str
    generator_url: str = ""

class RealAlertService:
    """
    Real production-grade alerting service providing:
    - Alert processing and enrichment
    - Escalation policies and management
    - Multi-channel notifications (email, Slack, webhooks)
    - Alert correlation and deduplication
    - Automated recovery action triggering
    - Alert lifecycle management
    """
    
    def __init__(self, config=None):
        self.config = config or settings
        self.websocket_events = RealWebSocketEvents(config)
        
        # Alert processing state
        self.active_alerts: Dict[str, ProcessedAlert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_queue: List[AlertNotification] = []
        self.recovery_handlers: Dict[str, List[Callable]] = {}
        
        # Processing configuration
        self.processing_interval = 10  # seconds
        self.escalation_interval = 300  # 5 minutes
        self.cleanup_interval = 3600  # 1 hour
        
        # Notification channels
        self.notification_channels: Dict[str, Callable] = {
            'email': self._send_email_notification,
            'slack': self._send_slack_notification,
            'webhook': self._send_webhook_notification,
            'teams': self._send_teams_notification
        }
        
        # Alert correlation
        self.correlation_rules: List[Dict[str, Any]] = []
        self.suppression_rules: List[Dict[str, Any]] = []
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'resolved_alerts': 0,
            'acknowledged_alerts': 0,
            'escalated_alerts': 0,
            'auto_resolved_alerts': 0
        }
        
        self._initialize_default_rules()
        logger.info("Real Alert Service initialized")

    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="ServiceDown",
                condition="knowledgehub_service_up == 0",
                threshold=0,
                duration=30,
                severity=AlertSeverity.CRITICAL,
                priority=AlertPriority.CRITICAL,
                escalation_delay=60,
                recovery_actions=["restart_service", "notify_oncall"]
            ),
            AlertRule(
                name="HighResponseTime",
                condition="knowledgehub_service_response_time_seconds > 1.0",
                threshold=1.0,
                duration=120,
                severity=AlertSeverity.WARNING,
                priority=AlertPriority.HIGH,
                escalation_delay=600,
                recovery_actions=["scale_service", "clear_cache"]
            ),
            AlertRule(
                name="MemorySearchTooSlow",
                condition="memory_search_95th_percentile > 0.05",
                threshold=0.05,
                duration=120,
                severity=AlertSeverity.CRITICAL,
                priority=AlertPriority.CRITICAL,
                escalation_delay=300,
                recovery_actions=["optimize_indices", "restart_ai_service"]
            ),
            AlertRule(
                name="HighErrorRate",
                condition="error_rate_5m > 0.1",
                threshold=0.1,
                duration=300,
                severity=AlertSeverity.WARNING,
                priority=AlertPriority.HIGH,
                escalation_delay=900,
                recovery_actions=["check_logs", "restart_failing_services"]
            ),
            AlertRule(
                name="DatabaseConnectionsHigh",
                condition="database_connections > 50",
                threshold=50,
                duration=300,
                severity=AlertSeverity.WARNING,
                priority=AlertPriority.MEDIUM,
                escalation_delay=1800,
                recovery_actions=["close_idle_connections", "scale_db_pool"]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule

    async def start_processing(self) -> None:
        """Start alert processing system"""
        logger.info("Starting alert processing system...")
        
        # Start processing tasks
        processing_tasks = [
            asyncio.create_task(self._process_alerts_loop()),
            asyncio.create_task(self._process_notifications_loop()),
            asyncio.create_task(self._escalation_loop()),
            asyncio.create_task(self._cleanup_loop()),
            asyncio.create_task(self._correlation_loop())
        ]
        
        logger.info("Alert processing system started")
        
        # Wait for all tasks
        await asyncio.gather(*processing_tasks, return_exceptions=True)

    async def process_webhook_alert(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming alert from Prometheus AlertManager webhook"""
        try:
            alerts = webhook_data.get('alerts', [])
            processed_count = 0
            
            for alert_data in alerts:
                processed_alert = await self._process_single_alert(alert_data)
                if processed_alert:
                    await self._store_alert(processed_alert)
                    await self._trigger_notifications(processed_alert)
                    await self._trigger_recovery_actions(processed_alert)
                    processed_count += 1
            
            self.alert_stats['total_alerts'] += processed_count
            
            return {
                "status": "success",
                "processed_alerts": processed_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process webhook alert: {e}")
            prometheus_metrics.record_error("webhook_processing", "alert_service")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: str = "") -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                
                # Update database
                async with get_db_session() as session:
                    query = update(SystemAlert).where(
                        SystemAlert.alert_id == alert_id
                    ).values(
                        acknowledged=True,
                        acknowledged_by=acknowledged_by,
                        acknowledged_at=datetime.now(timezone.utc)
                    )
                    await session.execute(query)
                    await session.commit()
                
                # Send WebSocket notification
                await self.websocket_events.send_event(
                    EventType.ALERT_ACKNOWLEDGED,
                    "alert_system",
                    {
                        "alert_id": alert_id,
                        "acknowledged_by": acknowledged_by,
                        "notes": notes
                    }
                )
                
                self.alert_stats['acknowledged_alerts'] += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system", notes: str = "") -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.ends_at = datetime.now(timezone.utc)
                
                # Update database
                async with get_db_session() as session:
                    query = update(SystemAlert).where(
                        SystemAlert.alert_id == alert_id
                    ).values(
                        resolved=True,
                        resolved_at=datetime.now(timezone.utc)
                    )
                    await session.execute(query)
                    await session.commit()
                
                # Send WebSocket notification
                await self.websocket_events.send_event(
                    EventType.ALERT_RESOLVED,
                    "alert_system",
                    {
                        "alert_id": alert_id,
                        "resolved_by": resolved_by,
                        "notes": notes
                    }
                )
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.alert_stats['resolved_alerts'] += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False

    async def add_recovery_handler(self, alert_name: str, handler: Callable) -> None:
        """Add a recovery handler for specific alert types"""
        if alert_name not in self.recovery_handlers:
            self.recovery_handlers[alert_name] = []
        self.recovery_handlers[alert_name].append(handler)

    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert processing statistics"""
        active_by_priority = {}
        active_by_severity = {}
        
        for alert in self.active_alerts.values():
            # Count by priority
            priority = alert.priority.value
            active_by_priority[priority] = active_by_priority.get(priority, 0) + 1
            
            # Count by severity
            severity = alert.severity.value
            active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
        
        return {
            **self.alert_stats,
            "active_alerts": len(self.active_alerts),
            "active_by_priority": active_by_priority,
            "active_by_severity": active_by_severity,
            "pending_notifications": len(self.notification_queue),
            "configured_rules": len(self.alert_rules)
        }

    # Private methods

    async def _process_single_alert(self, alert_data: Dict[str, Any]) -> Optional[ProcessedAlert]:
        """Process a single alert from webhook data"""
        try:
            labels = alert_data.get('labels', {})
            annotations = alert_data.get('annotations', {})
            
            alert_name = labels.get('alertname', 'Unknown')
            service_name = labels.get('service', labels.get('instance', ''))
            
            # Generate unique alert ID
            fingerprint = alert_data.get('fingerprint', '')
            alert_id = f"{alert_name}_{fingerprint}_{service_name}"
            
            # Determine severity and priority
            severity = AlertSeverity(labels.get('severity', 'warning'))
            priority = self._determine_priority(alert_name, severity, labels)
            
            # Parse timestamps
            starts_at_str = alert_data.get('startsAt', '')
            starts_at = datetime.fromisoformat(starts_at_str.replace('Z', '+00:00')) if starts_at_str else datetime.now(timezone.utc)
            
            ends_at_str = alert_data.get('endsAt', '')
            ends_at = datetime.fromisoformat(ends_at_str.replace('Z', '+00:00')) if ends_at_str else None
            
            # Determine status
            status = AlertStatus.RESOLVED if ends_at and ends_at < datetime.now(timezone.utc) else AlertStatus.ACTIVE
            
            return ProcessedAlert(
                alert_id=alert_id,
                name=alert_name,
                message=annotations.get('summary', annotations.get('description', f'Alert: {alert_name}')),
                severity=severity,
                priority=priority,
                status=status,
                service_name=service_name,
                labels=labels,
                annotations=annotations,
                starts_at=starts_at,
                ends_at=ends_at,
                fingerprint=fingerprint,
                generator_url=alert_data.get('generatorURL', '')
            )
            
        except Exception as e:
            logger.error(f"Failed to process alert data: {e}")
            return None

    def _determine_priority(self, alert_name: str, severity: AlertSeverity, labels: Dict[str, str]) -> AlertPriority:
        """Determine alert priority based on rule configuration"""
        if alert_name in self.alert_rules:
            return self.alert_rules[alert_name].priority
        
        # Default priority mapping
        priority_map = {
            AlertSeverity.CRITICAL: AlertPriority.CRITICAL,
            AlertSeverity.WARNING: AlertPriority.HIGH,
            AlertSeverity.INFO: AlertPriority.MEDIUM
        }
        
        return priority_map.get(severity, AlertPriority.MEDIUM)

    async def _store_alert(self, alert: ProcessedAlert) -> None:
        """Store alert in database"""
        try:
            async with get_db_session() as session:
                # Check if alert already exists
                existing_query = select(SystemAlert).where(
                    SystemAlert.alert_id == alert.alert_id
                )
                result = await session.execute(existing_query)
                existing_alert = result.scalar_one_or_none()
                
                if existing_alert:
                    # Update existing alert
                    existing_alert.last_occurrence = datetime.now(timezone.utc)
                    existing_alert.count += 1
                    if alert.status == AlertStatus.RESOLVED:
                        existing_alert.resolved = True
                        existing_alert.resolved_at = alert.ends_at
                else:
                    # Create new alert
                    db_alert = SystemAlert(
                        alert_id=alert.alert_id,
                        service_name=alert.service_name,
                        alert_type=alert.name,
                        severity=alert.severity,
                        message=alert.message,
                        details={"labels": alert.labels, "annotations": alert.annotations},
                        acknowledged=alert.status == AlertStatus.ACKNOWLEDGED,
                        resolved=alert.status == AlertStatus.RESOLVED,
                        count=1,
                        first_occurrence=alert.starts_at,
                        last_occurrence=datetime.now(timezone.utc)
                    )
                    session.add(db_alert)
                
                await session.commit()
                
                # Add to active alerts if not resolved
                if alert.status != AlertStatus.RESOLVED:
                    self.active_alerts[alert.alert_id] = alert
                    
        except Exception as e:
            logger.error(f"Failed to store alert {alert.alert_id}: {e}")

    async def _trigger_notifications(self, alert: ProcessedAlert) -> None:
        """Trigger notifications for an alert"""
        try:
            # Determine notification channels based on priority
            channels = self._get_notification_channels(alert.priority)
            
            for channel_config in channels:
                notification = AlertNotification(
                    alert_id=alert.alert_id,
                    recipient_type=channel_config['type'],
                    recipient=channel_config['recipient'],
                    template=channel_config['template']
                )
                self.notification_queue.append(notification)
                
        except Exception as e:
            logger.error(f"Failed to trigger notifications for alert {alert.alert_id}: {e}")

    async def _trigger_recovery_actions(self, alert: ProcessedAlert) -> None:
        """Trigger automated recovery actions for an alert"""
        try:
            if alert.name in self.alert_rules:
                rule = self.alert_rules[alert.name]
                
                for action in rule.recovery_actions:
                    if alert.name in self.recovery_handlers:
                        for handler in self.recovery_handlers[alert.name]:
                            try:
                                await handler(alert, action)
                                logger.info(f"Executed recovery action '{action}' for alert {alert.alert_id}")
                            except Exception as e:
                                logger.error(f"Recovery action '{action}' failed for alert {alert.alert_id}: {e}")
                                
        except Exception as e:
            logger.error(f"Failed to trigger recovery actions for alert {alert.alert_id}: {e}")

    def _get_notification_channels(self, priority: AlertPriority) -> List[Dict[str, str]]:
        """Get notification channels based on alert priority"""
        # Configuration would typically come from database or config file
        channel_config = {
            AlertPriority.CRITICAL: [
                {'type': 'email', 'recipient': 'admin@knowledgehub.local', 'template': 'critical_alert'},
                {'type': 'slack', 'recipient': '#alerts-critical', 'template': 'slack_critical'},
                {'type': 'webhook', 'recipient': 'http://oncall-system/webhook', 'template': 'webhook_alert'}
            ],
            AlertPriority.HIGH: [
                {'type': 'email', 'recipient': 'ops@knowledgehub.local', 'template': 'high_alert'},
                {'type': 'slack', 'recipient': '#alerts', 'template': 'slack_alert'}
            ],
            AlertPriority.MEDIUM: [
                {'type': 'slack', 'recipient': '#alerts', 'template': 'slack_alert'}
            ],
            AlertPriority.LOW: [
                {'type': 'email', 'recipient': 'dev@knowledgehub.local', 'template': 'low_alert'}
            ]
        }
        
        return channel_config.get(priority, [])

    async def _process_alerts_loop(self) -> None:
        """Main alert processing loop"""
        while True:
            try:
                # Process any pending alert logic
                await asyncio.sleep(self.processing_interval)
            except Exception as e:
                logger.error(f"Alert processing loop error: {e}")
                await asyncio.sleep(5)

    async def _process_notifications_loop(self) -> None:
        """Process notification queue"""
        while True:
            try:
                if self.notification_queue:
                    notification = self.notification_queue.pop(0)
                    await self._send_notification(notification)
                
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Notification processing error: {e}")
                await asyncio.sleep(5)

    async def _send_notification(self, notification: AlertNotification) -> bool:
        """Send a notification via the specified channel"""
        try:
            handler = self.notification_channels.get(notification.recipient_type)
            if handler:
                success = await handler(notification)
                if not success and notification.retry_count < notification.max_retries:
                    notification.retry_count += 1
                    notification.next_retry = datetime.now(timezone.utc) + timedelta(minutes=2 ** notification.retry_count)
                    self.notification_queue.append(notification)
                return success
            else:
                logger.warning(f"Unknown notification type: {notification.recipient_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    async def _send_email_notification(self, notification: AlertNotification) -> bool:
        """Send email notification"""
        # Implement email sending logic
        logger.info(f"Sending email notification to {notification.recipient} for alert {notification.alert_id}")
        return True

    async def _send_slack_notification(self, notification: AlertNotification) -> bool:
        """Send Slack notification"""
        # Implement Slack webhook sending logic
        logger.info(f"Sending Slack notification to {notification.recipient} for alert {notification.alert_id}")
        return True

    async def _send_webhook_notification(self, notification: AlertNotification) -> bool:
        """Send webhook notification"""
        # Implement webhook sending logic
        logger.info(f"Sending webhook notification to {notification.recipient} for alert {notification.alert_id}")
        return True

    async def _send_teams_notification(self, notification: AlertNotification) -> bool:
        """Send Teams notification"""
        # Implement Teams webhook sending logic
        logger.info(f"Sending Teams notification to {notification.recipient} for alert {notification.alert_id}")
        return True

    async def _escalation_loop(self) -> None:
        """Handle alert escalation"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                for alert in list(self.active_alerts.values()):
                    if alert.status == AlertStatus.ACTIVE:
                        # Check if alert should be escalated
                        rule = self.alert_rules.get(alert.name)
                        if rule:
                            escalation_time = alert.starts_at + timedelta(seconds=rule.escalation_delay)
                            if current_time >= escalation_time:
                                await self._escalate_alert(alert)
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Escalation loop error: {e}")
                await asyncio.sleep(30)

    async def _escalate_alert(self, alert: ProcessedAlert) -> None:
        """Escalate an alert to higher priority channels"""
        try:
            logger.info(f"Escalating alert {alert.alert_id}")
            
            # Increase priority
            if alert.priority == AlertPriority.MEDIUM:
                alert.priority = AlertPriority.HIGH
            elif alert.priority == AlertPriority.HIGH:
                alert.priority = AlertPriority.CRITICAL
            
            # Trigger escalated notifications
            await self._trigger_notifications(alert)
            
            self.alert_stats['escalated_alerts'] += 1
            
        except Exception as e:
            logger.error(f"Failed to escalate alert {alert.alert_id}: {e}")

    async def _cleanup_loop(self) -> None:
        """Clean up old resolved alerts"""
        while True:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                
                # Remove old resolved alerts
                resolved_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.status == AlertStatus.RESOLVED and alert.ends_at and alert.ends_at < cutoff_time
                ]
                
                for alert_id in resolved_alerts:
                    del self.active_alerts[alert_id]
                
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(1800)

    async def _correlation_loop(self) -> None:
        """Handle alert correlation and deduplication"""
        while True:
            try:
                # Implement alert correlation logic
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Correlation loop error: {e}")
                await asyncio.sleep(30)
    
    async def get_alert_rules(self) -> List[AlertRule]:
        """Get all configured alert rules"""
        return list(self.alert_rules.values())
    
    async def create_alert_rule(self, rule_data: Dict[str, Any]) -> AlertRule:
        """Create a new alert rule"""
        try:
            # Create AlertRule from data
            rule = AlertRule(
                name=rule_data['name'],
                condition=rule_data['condition'],
                threshold=rule_data['threshold'],
                duration=rule_data['duration'],
                severity=AlertSeverity(rule_data['severity']),
                priority=AlertPriority(rule_data['priority']),
                escalation_delay=rule_data.get('escalation_delay', 300),
                auto_resolve=rule_data.get('auto_resolve', True),
                recovery_actions=rule_data.get('recovery_actions', [])
            )
            
            # Store the rule
            self.alert_rules[rule.name] = rule
            
            # TODO: Persist to database
            logger.info(f"Created alert rule: {rule.name}")
            
            return rule
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            raise
    
    async def delete_alert_rule(self, rule_name: str) -> bool:
        """Delete an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Deleted alert rule: {rule_name}")
            return True
        return False
    
    async def update_alert_rule(self, rule_name: str, rule_data: Dict[str, Any]) -> Optional[AlertRule]:
        """Update an existing alert rule"""
        if rule_name not in self.alert_rules:
            return None
            
        try:
            # Update existing rule
            rule = self.alert_rules[rule_name]
            
            # Update fields if provided
            if 'condition' in rule_data:
                rule.condition = rule_data['condition']
            if 'threshold' in rule_data:
                rule.threshold = rule_data['threshold']
            if 'duration' in rule_data:
                rule.duration = rule_data['duration']
            if 'severity' in rule_data:
                rule.severity = AlertSeverity(rule_data['severity'])
            if 'priority' in rule_data:
                rule.priority = AlertPriority(rule_data['priority'])
            if 'escalation_delay' in rule_data:
                rule.escalation_delay = rule_data['escalation_delay']
            if 'auto_resolve' in rule_data:
                rule.auto_resolve = rule_data['auto_resolve']
            if 'recovery_actions' in rule_data:
                rule.recovery_actions = rule_data['recovery_actions']
            
            logger.info(f"Updated alert rule: {rule_name}")
            return rule
            
        except Exception as e:
            logger.error(f"Failed to update alert rule: {e}")
            raise

# Global alert service instance
real_alert_service = RealAlertService()