
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import smtplib
from email.mime.text import MIMEText

class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertingSystem:
    """Comprehensive alerting system"""
    
    def __init__(self):
        self.alert_rules = {
            "high_cpu": {
                "condition": lambda m: m.get("cpu_usage", 0) > 80,
                "severity": AlertSeverity.HIGH,
                "message": "CPU usage above 80%"
            },
            "high_memory": {
                "condition": lambda m: m.get("memory_usage", 0) > 90,
                "severity": AlertSeverity.CRITICAL,
                "message": "Memory usage above 90%"
            },
            "slow_response": {
                "condition": lambda m: m.get("avg_response_time", 0) > 1.0,
                "severity": AlertSeverity.MEDIUM,
                "message": "Average response time above 1 second"
            },
            "error_rate": {
                "condition": lambda m: m.get("error_rate", 0) > 0.05,
                "severity": AlertSeverity.HIGH,
                "message": "Error rate above 5%"
            },
            "database_connections": {
                "condition": lambda m: m.get("db_connections", 0) > 90,
                "severity": AlertSeverity.HIGH,
                "message": "Database connection pool nearly exhausted"
            }
        }
        
        self.alert_history = []
        self.alert_cooldown = {}  # Prevent alert spam
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if rule["condition"](metrics):
                # Check cooldown
                if rule_name in self.alert_cooldown:
                    last_alert = self.alert_cooldown[rule_name]
                    if datetime.now() - last_alert < timedelta(minutes=5):
                        continue
                
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "rule": rule_name,
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "metrics": metrics
                }
                
                alerts.append(alert)
                self.alert_history.append(alert)
                self.alert_cooldown[rule_name] = datetime.now()
                
                # Send alert
                await self.send_alert(alert)
        
        return alerts
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels"""
        # Log alert
        logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")
        
        # Send to monitoring dashboard
        # await self.send_to_dashboard(alert)
        
        # Send email for critical alerts
        if alert["severity"] in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            # await self.send_email_alert(alert)
            pass
        
        # Send to Slack/Discord webhook
        # await self.send_webhook_alert(alert)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        recent = datetime.now() - timedelta(hours=1)
        recent_alerts = [
            a for a in self.alert_history
            if datetime.fromisoformat(a["timestamp"]) > recent
        ]
        
        return {
            "total_alerts": len(recent_alerts),
            "by_severity": {
                severity: len([a for a in recent_alerts if a["severity"] == severity])
                for severity in AlertSeverity
            },
            "recent_alerts": recent_alerts[-10:]  # Last 10 alerts
        }

alerting = AlertingSystem()
