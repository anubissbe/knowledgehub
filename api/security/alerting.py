"""
Security Alerting System

Provides automated security alert management and notification.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import aiohttp
from collections import defaultdict

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert notification channels"""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRule:
    """Defines when and how to trigger alerts"""
    name: str
    description: str
    condition: str  # e.g., "auth_failures > 10"
    threshold: int
    time_window_minutes: int
    severity: AlertSeverity
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents a triggered alert"""
    id: str
    rule_name: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    data: Dict[str, Any]
    channels: List[AlertChannel]
    acknowledged: bool = False
    resolved: bool = False


class SecurityAlertingSystem:
    """Manages security alerts and notifications"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, int] = defaultdict(int)
        
        # Notification configuration
        self.webhook_urls = self.config.get('webhook_urls', {})
        self.email_config = self.config.get('email', {})
        self.slack_config = self.config.get('slack', {})
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
        logger.info("Security alerting system initialized")
    
    def _initialize_default_rules(self):
        """Set up default security alert rules"""
        default_rules = [
            AlertRule(
                name="high_auth_failure_rate",
                description="High rate of authentication failures",
                condition="auth_failures",
                threshold=10,
                time_window_minutes=60,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK]
            ),
            AlertRule(
                name="critical_threat_detected",
                description="Critical security threat detected",
                condition="critical_threats",
                threshold=1,
                time_window_minutes=5,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK, AlertChannel.PAGERDUTY],
                cooldown_minutes=15
            ),
            AlertRule(
                name="dos_attack_suspected",
                description="Potential DoS attack detected",
                condition="dos_attempts",
                threshold=50,
                time_window_minutes=5,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=30
            ),
            AlertRule(
                name="injection_attempts",
                description="Multiple injection attempts detected",
                condition="injection_attempts",
                threshold=5,
                time_window_minutes=30,
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK]
            ),
            AlertRule(
                name="new_blocked_ips",
                description="New IPs have been blocked",
                condition="blocked_ips",
                threshold=5,
                time_window_minutes=60,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG]
            ),
            AlertRule(
                name="security_scan_detected",
                description="Security scanning activity detected",
                condition="security_scans",
                threshold=3,
                time_window_minutes=15,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK]
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Alert rule added/updated: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Alert rule removed: {rule_name}")
    
    async def check_alerts(self, security_stats: Dict[str, Any]):
        """Check if any alert conditions are met"""
        current_time = datetime.now()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule_name in self.last_alert_times:
                time_since_last = current_time - self.last_alert_times[rule_name]
                if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                    continue
            
            # Check condition
            should_alert = self._evaluate_condition(rule, security_stats)
            
            if should_alert:
                await self._trigger_alert(rule, security_stats)
    
    def _evaluate_condition(self, rule: AlertRule, stats: Dict[str, Any]) -> bool:
        """Evaluate if alert condition is met"""
        try:
            # Map conditions to stats
            condition_map = {
                "auth_failures": stats.get("event_types_24h", {}).get("auth_failure", 0),
                "critical_threats": stats.get("threat_levels_24h", {}).get("critical", 0),
                "dos_attempts": stats.get("event_types_24h", {}).get("dos_attempt", 0),
                "injection_attempts": stats.get("event_types_24h", {}).get("injection_attempt", 0),
                "blocked_ips": stats.get("blocked_ips_count", 0),
                "security_scans": stats.get("event_types_24h", {}).get("security_scan", 0)
            }
            
            value = condition_map.get(rule.condition, 0)
            return value >= rule.threshold
            
        except Exception as e:
            logger.error(f"Error evaluating condition for rule {rule.name}: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, stats: Dict[str, Any]):
        """Trigger an alert based on rule"""
        alert_id = f"{rule.name}_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            title=f"Security Alert: {rule.description}",
            description=self._generate_alert_description(rule, stats),
            timestamp=datetime.now(),
            data=stats,
            channels=rule.channels
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.name] = datetime.now()
        self.alert_counts[rule.name] += 1
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"Alert triggered: {alert.title}")
    
    def _generate_alert_description(self, rule: AlertRule, stats: Dict[str, Any]) -> str:
        """Generate detailed alert description"""
        desc_parts = [
            f"Alert rule '{rule.name}' triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Threshold: {rule.threshold} events in {rule.time_window_minutes} minutes",
            f"Current stats: {json.dumps(stats.get('event_types_24h', {}), indent=2)}"
        ]
        
        if stats.get('top_threatening_ips'):
            desc_parts.append(f"Top threats: {stats['top_threatening_ips'][:3]}")
        
        return "\n".join(desc_parts)
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        tasks = []
        
        for channel in alert.channels:
            if channel == AlertChannel.LOG:
                self._log_alert(alert)
            elif channel == AlertChannel.WEBHOOK:
                tasks.append(self._send_webhook(alert))
            elif channel == AlertChannel.SLACK:
                tasks.append(self._send_slack(alert))
            elif channel == AlertChannel.EMAIL:
                tasks.append(self._send_email(alert))
            elif channel == AlertChannel.TEAMS:
                tasks.append(self._send_teams(alert))
            elif channel == AlertChannel.PAGERDUTY:
                tasks.append(self._send_pagerduty(alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _log_alert(self, alert: Alert):
        """Log alert to system logs"""
        log_level = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }
        
        log_func = log_level.get(alert.severity, logger.warning)
        log_func(f"SECURITY ALERT: {alert.title} - {alert.description}")
    
    async def _send_webhook(self, alert: Alert):
        """Send alert via webhook"""
        if not self.webhook_urls:
            return
        
        webhook_url = self.webhook_urls.get(alert.severity.value, 
                                           self.webhook_urls.get('default'))
        if not webhook_url:
            return
        
        payload = {
            "alert_id": alert.id,
            "severity": alert.severity.value,
            "title": alert.title,
            "description": alert.description,
            "timestamp": alert.timestamp.isoformat(),
            "rule": alert.rule_name,
            "data": alert.data
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Webhook notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    async def _send_slack(self, alert: Alert):
        """Send alert to Slack"""
        if not self.slack_config.get('webhook_url'):
            return
        
        # Color based on severity
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#d00000"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "#ff9900"),
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                ],
                "footer": "KnowledgeHub Security",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_config['webhook_url'],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Slack notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_email(self, alert: Alert):
        """Send alert via email (placeholder)"""
        # Email implementation would depend on email service configuration
        logger.info(f"Email alert would be sent: {alert.title}")
    
    async def _send_teams(self, alert: Alert):
        """Send alert to Microsoft Teams (placeholder)"""
        logger.info(f"Teams alert would be sent: {alert.title}")
    
    async def _send_pagerduty(self, alert: Alert):
        """Send alert to PagerDuty (placeholder)"""
        logger.info(f"PagerDuty alert would be sent: {alert.title}")
    
    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.acknowledged = True
            # Move to history only
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "active_alerts": len(self.active_alerts),
            "total_alerts_triggered": sum(self.alert_counts.values()),
            "alerts_by_rule": dict(self.alert_counts),
            "recent_alerts": [
                {
                    "id": alert.id,
                    "rule": alert.rule_name,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ]
        }


# Global alerting system instance
alerting_system = None


def init_alerting_system(config: Optional[Dict[str, Any]] = None):
    """Initialize the global alerting system"""
    global alerting_system
    alerting_system = SecurityAlertingSystem(config)
    return alerting_system


def get_alerting_system():
    """Get the global alerting system instance"""
    return alerting_system