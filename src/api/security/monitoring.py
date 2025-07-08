"""
Security Monitoring and Logging System

Provides comprehensive security monitoring, threat detection, audit logging,
and alerting capabilities for the KnowledgeHub API.
"""

import logging
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncio
from pathlib import Path

from pydantic import BaseModel
# aiofiles not available, using standard file operations

logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Security event types for classification"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHORIZATION_FAILURE = "authz_failure"
    CORS_VIOLATION = "cors_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_REQUEST = "suspicious_request"
    INJECTION_ATTEMPT = "injection_attempt"
    DOS_ATTEMPT = "dos_attempt"
    MALFORMED_REQUEST = "malformed_request"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS_ANOMALY = "data_access_anomaly"
    API_ABUSE = "api_abuse"
    SECURITY_SCAN = "security_scan"
    BRUTE_FORCE = "brute_force"
    SESSION_HIJACK = "session_hijack"


class ThreatLevel(str, Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    timestamp: datetime
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_agent: str
    endpoint: str
    method: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    origin: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = None
    blocked: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SecurityMonitor:
    """Central security monitoring and alerting system"""
    
    def __init__(self, log_directory: str = "/app/logs/security"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Event tracking
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.ip_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Set[str] = set()
        
        # Attack pattern detection
        self.failed_auth_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.endpoint_access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        
        # Alerting thresholds
        self.config = {
            "max_failed_auth_per_hour": 10,
            "max_requests_per_minute": 120,
            "max_endpoints_per_minute": 20,
            "suspicious_user_agents": [
                "sqlmap", "nikto", "nmap", "burp", "nessus", "openvas",
                "dirbuster", "gobuster", "wfuzz", "hydra", "metasploit"
            ],
            "blocked_request_threshold": 50,  # Auto-block after 50 blocked requests
            "alert_cooldown_minutes": 15  # Minimum time between alerts for same IP
        }
        
        # Alert tracking
        self.last_alerts: Dict[str, datetime] = {}
        
        # Setup specialized loggers
        self._setup_loggers()
        
        logger.info("Security monitoring system initialized")
    
    def _setup_loggers(self):
        """Setup specialized security loggers"""
        # Security events logger
        self.security_logger = logging.getLogger("security.events")
        security_handler = logging.FileHandler(self.log_directory / "security_events.log")
        security_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.security_logger.addHandler(security_handler)
        self.security_logger.setLevel(logging.INFO)
        
        # Audit logger
        self.audit_logger = logging.getLogger("security.audit")
        audit_handler = logging.FileHandler(self.log_directory / "audit.log")
        audit_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
        
        # Threat intelligence logger
        self.threat_logger = logging.getLogger("security.threats")
        threat_handler = logging.FileHandler(self.log_directory / "threats.log")
        threat_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.threat_logger.addHandler(threat_handler)
        self.threat_logger.setLevel(logging.WARNING)
    
    async def log_event(self, event: SecurityEvent) -> None:
        """Log a security event and perform analysis"""
        try:
            # Add to memory tracking
            self.events.append(event)
            self.ip_events[event.source_ip].append(event)
            
            # Log to appropriate logger
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()
            
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.threat_logger.warning(f"HIGH/CRITICAL THREAT: {json.dumps(event_dict)}")
            else:
                self.security_logger.info(json.dumps(event_dict))
            
            # Audit logging for important events
            if event.event_type in [SecurityEventType.AUTHENTICATION_SUCCESS, 
                                   SecurityEventType.AUTHORIZATION_FAILURE,
                                   SecurityEventType.PRIVILEGE_ESCALATION]:
                self.audit_logger.info(f"AUDIT: {event.event_type.value} - IP: {event.source_ip} - Endpoint: {event.endpoint}")
            
            # Perform real-time analysis
            await self._analyze_event(event)
            
            # Write to JSON log for structured analysis
            self._write_json_log(event)
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    async def _analyze_event(self, event: SecurityEvent) -> None:
        """Perform real-time security analysis"""
        
        # Check for brute force attacks
        if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
            await self._check_brute_force(event)
        
        # Check for DoS patterns
        await self._check_dos_patterns(event)
        
        # Check for suspicious scanning behavior
        await self._check_scanning_behavior(event)
        
        # Check for injection attempts
        await self._check_injection_patterns(event)
        
        # Auto-blocking logic
        await self._check_auto_block(event)
    
    async def _check_brute_force(self, event: SecurityEvent) -> None:
        """Detect brute force authentication attempts"""
        ip = event.source_ip
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Track failed attempts
        self.failed_auth_attempts[ip].append(now)
        
        # Count recent failures
        recent_failures = [
            attempt for attempt in self.failed_auth_attempts[ip]
            if attempt > hour_ago
        ]
        
        if len(recent_failures) >= self.config["max_failed_auth_per_hour"]:
            threat_event = SecurityEvent(
                timestamp=now,
                event_type=SecurityEventType.BRUTE_FORCE,
                threat_level=ThreatLevel.HIGH,
                source_ip=ip,
                user_agent=event.user_agent,
                endpoint=event.endpoint,
                method=event.method,
                description=f"Brute force detected: {len(recent_failures)} failed attempts in 1 hour",
                blocked=True
            )
            
            await self.log_event(threat_event)
            await self._trigger_alert(threat_event)
            self.suspicious_ips.add(ip)
    
    async def _check_dos_patterns(self, event: SecurityEvent) -> None:
        """Detect potential DoS attacks"""
        ip = event.source_ip
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Count recent requests from this IP
        recent_requests = [
            e for e in self.ip_events[ip]
            if e.timestamp > minute_ago
        ]
        
        if len(recent_requests) > self.config["max_requests_per_minute"]:
            threat_event = SecurityEvent(
                timestamp=now,
                event_type=SecurityEventType.DOS_ATTEMPT,
                threat_level=ThreatLevel.HIGH,
                source_ip=ip,
                user_agent=event.user_agent,
                endpoint=event.endpoint,
                method=event.method,
                description=f"DoS pattern detected: {len(recent_requests)} requests in 1 minute",
                blocked=True
            )
            
            await self.log_event(threat_event)
            await self._trigger_alert(threat_event)
            self.suspicious_ips.add(ip)
    
    async def _check_scanning_behavior(self, event: SecurityEvent) -> None:
        """Detect security scanning behavior"""
        ip = event.source_ip
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Track endpoint access patterns
        self.endpoint_access_patterns[ip].append((now, event.endpoint))
        
        # Count unique endpoints accessed recently
        recent_endpoints = set([
            endpoint for timestamp, endpoint in self.endpoint_access_patterns[ip]
            if timestamp > minute_ago
        ])
        
        if len(recent_endpoints) > self.config["max_endpoints_per_minute"]:
            threat_event = SecurityEvent(
                timestamp=now,
                event_type=SecurityEventType.SECURITY_SCAN,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=ip,
                user_agent=event.user_agent,
                endpoint=event.endpoint,
                method=event.method,
                description=f"Scanning behavior detected: {len(recent_endpoints)} unique endpoints in 1 minute",
                blocked=False
            )
            
            await self.log_event(threat_event)
            self.suspicious_ips.add(ip)
    
    async def _check_injection_patterns(self, event: SecurityEvent) -> None:
        """Detect injection attempt patterns"""
        # Check user agent for known attack tools
        user_agent_lower = event.user_agent.lower()
        for suspicious_agent in self.config["suspicious_user_agents"]:
            if suspicious_agent in user_agent_lower:
                threat_event = SecurityEvent(
                    timestamp=datetime.now(),
                    event_type=SecurityEventType.INJECTION_ATTEMPT,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=event.source_ip,
                    user_agent=event.user_agent,
                    endpoint=event.endpoint,
                    method=event.method,
                    description=f"Suspicious user agent detected: {suspicious_agent}",
                    blocked=True
                )
                
                await self.log_event(threat_event)
                await self._trigger_alert(threat_event)
                self.suspicious_ips.add(event.source_ip)
                break
    
    async def _check_auto_block(self, event: SecurityEvent) -> None:
        """Auto-block IPs with excessive blocked requests"""
        if event.blocked and event.event_type != SecurityEventType.API_ABUSE:  # Prevent recursion
            ip = event.source_ip
            blocked_count = sum(1 for e in self.ip_events[ip] if e.blocked)
            
            if blocked_count >= self.config["blocked_request_threshold"] and ip not in self.blocked_ips:
                self.blocked_ips.add(ip)
                
                # Log the blocking event but don't re-process it to avoid recursion
                logger.critical(f"IP {ip} auto-blocked after {blocked_count} blocked requests")
                await self._trigger_alert(event)
    
    async def _trigger_alert(self, event: SecurityEvent) -> None:
        """Trigger security alerts for high-priority events"""
        alert_key = f"{event.source_ip}_{event.event_type.value}"
        now = datetime.now()
        cooldown = timedelta(minutes=self.config["alert_cooldown_minutes"])
        
        # Check if we're in cooldown period
        if alert_key in self.last_alerts:
            if now - self.last_alerts[alert_key] < cooldown:
                return
        
        self.last_alerts[alert_key] = now
        
        # Log alert
        alert_message = f"SECURITY ALERT: {event.event_type.value} from {event.source_ip} - {event.description}"
        self.threat_logger.critical(alert_message)
        
        # In production, this would integrate with:
        # - Email/SMS alerting
        # - Slack/Teams notifications
        # - SIEM systems (Splunk, ELK, etc.)
        # - Incident response tools (PagerDuty, Opsgenie)
        
        logger.critical(f"SECURITY ALERT TRIGGERED: {alert_message}")
    
    def _write_json_log(self, event: SecurityEvent) -> None:
        """Write structured JSON log for analysis tools"""
        try:
            log_file = self.log_directory / f"security_events_{datetime.now().strftime('%Y%m%d')}.json"
            
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing JSON log: {e}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get current security statistics"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        recent_events = [e for e in self.events if e.timestamp > hour_ago]
        daily_events = [e for e in self.events if e.timestamp > day_ago]
        
        # Event type counts
        event_type_counts = defaultdict(int)
        threat_level_counts = defaultdict(int)
        
        for event in daily_events:
            event_type_counts[event.event_type.value] += 1
            threat_level_counts[event.threat_level.value] += 1
        
        return {
            "monitoring_status": "active",
            "total_events_last_hour": len(recent_events),
            "total_events_last_24h": len(daily_events),
            "blocked_ips_count": len(self.blocked_ips),
            "suspicious_ips_count": len(self.suspicious_ips),
            "event_types_24h": dict(event_type_counts),
            "threat_levels_24h": dict(threat_level_counts),
            "top_threatening_ips": self._get_top_threatening_ips(),
            "system_health": {
                "log_directory": str(self.log_directory),
                "events_in_memory": len(self.events),
                "tracking_ips": len(self.ip_events)
            }
        }
    
    def _get_top_threatening_ips(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top threatening IP addresses"""
        ip_threat_scores = defaultdict(int)
        
        for event in self.events:
            score = 1
            if event.threat_level == ThreatLevel.CRITICAL:
                score = 10
            elif event.threat_level == ThreatLevel.HIGH:
                score = 5
            elif event.threat_level == ThreatLevel.MEDIUM:
                score = 2
            
            ip_threat_scores[event.source_ip] += score
        
        # Sort by threat score
        sorted_ips = sorted(ip_threat_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "ip": ip,
                "threat_score": score,
                "is_blocked": ip in self.blocked_ips,
                "is_suspicious": ip in self.suspicious_ips,
                "event_count": len(self.ip_events[ip])
            }
            for ip, score in sorted_ips[:limit]
        ]
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is blocked"""
        return ip in self.blocked_ips
    
    def is_ip_suspicious(self, ip: str) -> bool:
        """Check if an IP is suspicious"""
        return ip in self.suspicious_ips
    
    def block_ip(self, ip: str, reason: str = "Manual block") -> None:
        """Manually block an IP address"""
        self.blocked_ips.add(ip)
        logger.warning(f"IP {ip} manually blocked: {reason}")
    
    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        self.suspicious_ips.discard(ip)
        logger.info(f"IP {ip} unblocked")
    
    async def cleanup_old_events(self, days_to_keep: int = 30) -> None:
        """Clean up old events and logs"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean memory events
        self.events = deque([e for e in self.events if e.timestamp > cutoff_date], maxlen=10000)
        
        # Clean IP events
        for ip in list(self.ip_events.keys()):
            self.ip_events[ip] = deque([e for e in self.ip_events[ip] if e.timestamp > cutoff_date], maxlen=100)
            if not self.ip_events[ip]:
                del self.ip_events[ip]
        
        # Archive old log files
        self._archive_old_logs(days_to_keep)
        
        logger.info(f"Cleaned up security events older than {days_to_keep} days")
    
    def _archive_old_logs(self, days_to_keep: int) -> None:
        """Archive old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            archive_dir = self.log_directory / "archive"
            archive_dir.mkdir(exist_ok=True)
            
            for log_file in self.log_directory.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    archive_path = archive_dir / f"{log_file.stem}_{int(cutoff_date.timestamp())}.log"
                    log_file.rename(archive_path)
                    logger.info(f"Archived old log file: {log_file} -> {archive_path}")
                    
        except Exception as e:
            logger.error(f"Error archiving old logs: {e}")


# Global security monitor instance
security_monitor = SecurityMonitor()


# Convenience functions for easy integration
async def log_security_event(
    event_type: SecurityEventType,
    threat_level: ThreatLevel,
    source_ip: str,
    user_agent: str,
    endpoint: str,
    method: str,
    description: str = "",
    **kwargs
) -> None:
    """Convenience function to log security events"""
    event = SecurityEvent(
        timestamp=datetime.now(),
        event_type=event_type,
        threat_level=threat_level,
        source_ip=source_ip,
        user_agent=user_agent,
        endpoint=endpoint,
        method=method,
        description=description,
        **kwargs
    )
    await security_monitor.log_event(event)


async def log_auth_failure(source_ip: str, user_agent: str, endpoint: str, user_id: str = None) -> None:
    """Log authentication failure"""
    await log_security_event(
        SecurityEventType.AUTHENTICATION_FAILURE,
        ThreatLevel.MEDIUM,
        source_ip,
        user_agent,
        endpoint,
        "POST",
        f"Authentication failed for user: {user_id or 'unknown'}",
        user_id=user_id
    )


async def log_cors_violation(source_ip: str, user_agent: str, endpoint: str, origin: str) -> None:
    """Log CORS violation"""
    await log_security_event(
        SecurityEventType.CORS_VIOLATION,
        ThreatLevel.MEDIUM,
        source_ip,
        user_agent,
        endpoint,
        "OPTIONS",
        f"CORS violation from origin: {origin}",
        origin=origin,
        blocked=True
    )


async def log_rate_limit_exceeded(source_ip: str, user_agent: str, endpoint: str) -> None:
    """Log rate limit exceeded"""
    await log_security_event(
        SecurityEventType.RATE_LIMIT_EXCEEDED,
        ThreatLevel.LOW,
        source_ip,
        user_agent,
        endpoint,
        "GET",
        "Rate limit exceeded",
        blocked=True
    )


async def log_suspicious_request(source_ip: str, user_agent: str, endpoint: str, reason: str) -> None:
    """Log suspicious request"""
    await log_security_event(
        SecurityEventType.SUSPICIOUS_REQUEST,
        ThreatLevel.MEDIUM,
        source_ip,
        user_agent,
        endpoint,
        "GET",
        f"Suspicious request: {reason}",
        blocked=True
    )