"""
Security Dashboard Data Provider

Provides aggregated security data for dashboards and reporting.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

from .monitoring import security_monitor, SecurityEventType, ThreatLevel

logger = logging.getLogger(__name__)


class SecurityDashboard:
    """Security dashboard data aggregator"""
    
    def __init__(self):
        self.monitor = security_monitor
    
    def get_dashboard_data(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent events
        recent_events = [e for e in self.monitor.events if e.timestamp > cutoff_time]
        
        return {
            "overview": self._get_overview_stats(recent_events),
            "threat_analysis": self._get_threat_analysis(recent_events),
            "geographic_data": self._get_geographic_analysis(recent_events),
            "timeline_data": self._get_timeline_data(recent_events, hours_back),
            "top_threats": self._get_top_threats(),
            "security_health": self._get_security_health(),
            "recommendations": self._get_security_recommendations(recent_events)
        }
    
    def _get_overview_stats(self, events: List) -> Dict[str, Any]:
        """Get overview statistics"""
        total_events = len(events)
        blocked_events = sum(1 for e in events if e.blocked)
        unique_ips = len(set(e.source_ip for e in events))
        
        # Threat level distribution
        threat_counts = Counter(e.threat_level.value for e in events)
        
        return {
            "total_events": total_events,
            "blocked_events": blocked_events,
            "unique_source_ips": unique_ips,
            "block_rate": (blocked_events / total_events * 100) if total_events > 0 else 0,
            "threat_distribution": dict(threat_counts),
            "currently_blocked_ips": len(self.monitor.blocked_ips),
            "suspicious_ips": len(self.monitor.suspicious_ips)
        }
    
    def _get_threat_analysis(self, events: List) -> Dict[str, Any]:
        """Analyze threat patterns"""
        # Event type analysis
        event_type_counts = Counter(e.event_type.value for e in events)
        
        # Attack pattern trends
        injection_attempts = sum(1 for e in events if e.event_type == SecurityEventType.INJECTION_ATTEMPT)
        dos_attempts = sum(1 for e in events if e.event_type == SecurityEventType.DOS_ATTEMPT)
        auth_failures = sum(1 for e in events if e.event_type == SecurityEventType.AUTHENTICATION_FAILURE)
        
        # Most targeted endpoints
        endpoint_targets = Counter(e.endpoint for e in events if e.blocked)
        
        return {
            "event_types": dict(event_type_counts),
            "attack_patterns": {
                "injection_attempts": injection_attempts,
                "dos_attempts": dos_attempts,
                "authentication_failures": auth_failures,
                "brute_force_attacks": sum(1 for e in events if e.event_type == SecurityEventType.BRUTE_FORCE)
            },
            "targeted_endpoints": dict(endpoint_targets.most_common(10)),
            "attack_success_rate": self._calculate_attack_success_rate(events)
        }
    
    def _get_geographic_analysis(self, events: List) -> Dict[str, Any]:
        """Analyze geographic distribution of threats"""
        # Group by IP ranges (simplified geographic analysis)
        ip_ranges = defaultdict(int)
        
        for event in events:
            ip = event.source_ip
            if ip.startswith('192.168.'):
                ip_ranges['Local Network'] += 1
            elif ip.startswith('10.'):
                ip_ranges['Private Network'] += 1
            elif ip.startswith('127.'):
                ip_ranges['Localhost'] += 1
            else:
                # Simplified public IP categorization
                first_octet = ip.split('.')[0] if '.' in ip else '0'
                try:
                    octet_val = int(first_octet)
                    if octet_val < 64:
                        ip_ranges['Americas'] += 1
                    elif octet_val < 128:
                        ip_ranges['Europe'] += 1
                    elif octet_val < 192:
                        ip_ranges['Asia-Pacific'] += 1
                    else:
                        ip_ranges['Other'] += 1
                except ValueError:
                    ip_ranges['Unknown'] += 1
        
        return {
            "regions": dict(ip_ranges),
            "unique_countries": len(ip_ranges),  # Simplified
            "international_threats": sum(v for k, v in ip_ranges.items() 
                                       if k not in ['Local Network', 'Private Network', 'Localhost'])
        }
    
    def _get_timeline_data(self, events: List, hours_back: int) -> Dict[str, Any]:
        """Generate timeline data for charts"""
        # Create hourly buckets
        timeline = {}
        now = datetime.now()
        
        for i in range(hours_back):
            hour_start = now - timedelta(hours=i+1)
            hour_key = hour_start.strftime('%Y-%m-%d %H:00')
            timeline[hour_key] = {
                'total_events': 0,
                'blocked_events': 0,
                'threat_levels': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
            }
        
        # Populate timeline with event data
        for event in events:
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:00')
            if hour_key in timeline:
                timeline[hour_key]['total_events'] += 1
                if event.blocked:
                    timeline[hour_key]['blocked_events'] += 1
                timeline[hour_key]['threat_levels'][event.threat_level.value] += 1
        
        # Convert to list format for charting
        timeline_list = []
        for hour_key in sorted(timeline.keys()):
            data = timeline[hour_key]
            timeline_list.append({
                'timestamp': hour_key,
                'total_events': data['total_events'],
                'blocked_events': data['blocked_events'],
                'critical_events': data['threat_levels']['critical'],
                'high_events': data['threat_levels']['high']
            })
        
        return {
            "hourly_timeline": timeline_list,
            "peak_hour": max(timeline.items(), key=lambda x: x[1]['total_events'])[0] if timeline else None,
            "quietest_hour": min(timeline.items(), key=lambda x: x[1]['total_events'])[0] if timeline else None
        }
    
    def _get_top_threats(self) -> List[Dict[str, Any]]:
        """Get top threatening IPs with detailed analysis"""
        return self.monitor._get_top_threatening_ips(limit=20)
    
    def _get_security_health(self) -> Dict[str, Any]:
        """Assess overall security health"""
        stats = self.monitor.get_security_stats()
        
        # Calculate health score (0-100)
        health_score = 100
        
        # Deduct for high threat activity
        if stats['total_events_last_hour'] > 100:
            health_score -= 20
        elif stats['total_events_last_hour'] > 50:
            health_score -= 10
        
        # Deduct for blocked IPs
        if stats['blocked_ips_count'] > 10:
            health_score -= 15
        elif stats['blocked_ips_count'] > 5:
            health_score -= 5
        
        # Deduct for critical/high threat events
        critical_events = stats['threat_levels_24h'].get('critical', 0)
        high_events = stats['threat_levels_24h'].get('high', 0)
        
        health_score -= (critical_events * 5) + (high_events * 2)
        health_score = max(0, health_score)  # Ensure non-negative
        
        # Determine health status
        if health_score >= 90:
            status = "EXCELLENT"
            color = "green"
        elif health_score >= 75:
            status = "GOOD"
            color = "green"
        elif health_score >= 60:
            status = "FAIR"
            color = "yellow"
        elif health_score >= 40:
            status = "POOR"
            color = "orange"
        else:
            status = "CRITICAL"
            color = "red"
        
        return {
            "health_score": health_score,
            "status": status,
            "status_color": color,
            "monitoring_uptime": "99.9%",  # Placeholder
            "last_incident": self._get_last_critical_incident(),
            "security_posture": self._assess_security_posture(stats)
        }
    
    def _get_security_recommendations(self, events: List) -> List[Dict[str, Any]]:
        """Generate security recommendations based on recent activity"""
        recommendations = []
        
        # Analyze patterns for recommendations
        injection_attempts = sum(1 for e in events if e.event_type == SecurityEventType.INJECTION_ATTEMPT)
        dos_attempts = sum(1 for e in events if e.event_type == SecurityEventType.DOS_ATTEMPT)
        auth_failures = sum(1 for e in events if e.event_type == SecurityEventType.AUTHENTICATION_FAILURE)
        
        if injection_attempts > 10:
            recommendations.append({
                "priority": "HIGH",
                "category": "Input Validation",
                "title": "Increase Input Validation",
                "description": f"Detected {injection_attempts} injection attempts. Consider implementing stricter input validation and WAF rules.",
                "action": "Review and strengthen input sanitization procedures"
            })
        
        if dos_attempts > 5:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Rate Limiting",
                "title": "Enhance Rate Limiting",
                "description": f"Detected {dos_attempts} DoS attempts. Consider implementing more aggressive rate limiting.",
                "action": "Review rate limiting thresholds and implement adaptive rate limiting"
            })
        
        if auth_failures > 50:
            recommendations.append({
                "priority": "HIGH",
                "category": "Authentication",
                "title": "Strengthen Authentication",
                "description": f"High number of authentication failures ({auth_failures}). Consider implementing account lockouts and CAPTCHA.",
                "action": "Implement progressive authentication delays and account lockout policies"
            })
        
        if len(self.monitor.blocked_ips) > 20:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "IP Management",
                "title": "Review Blocked IPs",
                "description": f"Large number of blocked IPs ({len(self.monitor.blocked_ips)}). Review and clean up if necessary.",
                "action": "Audit blocked IP list and implement automatic cleanup procedures"
            })
        
        # Add general recommendations if no specific issues
        if not recommendations:
            recommendations.append({
                "priority": "LOW",
                "category": "Maintenance",
                "title": "Security System Healthy",
                "description": "No immediate security concerns detected. Continue regular monitoring.",
                "action": "Maintain current security posture and monitoring practices"
            })
        
        return recommendations
    
    def _calculate_attack_success_rate(self, events: List) -> float:
        """Calculate what percentage of attacks were successful (not blocked)"""
        attack_events = [e for e in events if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
        if not attack_events:
            return 0.0
        
        successful_attacks = sum(1 for e in attack_events if not e.blocked)
        return (successful_attacks / len(attack_events)) * 100
    
    def _get_last_critical_incident(self) -> Optional[str]:
        """Get timestamp of last critical security incident"""
        for event in reversed(self.monitor.events):
            if event.threat_level == ThreatLevel.CRITICAL:
                return event.timestamp.isoformat()
        return None
    
    def _assess_security_posture(self, stats: Dict) -> str:
        """Assess overall security posture"""
        critical_count = stats['threat_levels_24h'].get('critical', 0)
        high_count = stats['threat_levels_24h'].get('high', 0)
        
        if critical_count > 5 or high_count > 20:
            return "DEFENSIVE - High threat activity detected"
        elif critical_count > 0 or high_count > 5:
            return "VIGILANT - Moderate threat activity"
        else:
            return "SECURE - Low threat environment"


# Global dashboard instance
security_dashboard = SecurityDashboard()