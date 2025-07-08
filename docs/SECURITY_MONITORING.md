# Security Monitoring and Logging System

## Overview

The KnowledgeHub API implements a comprehensive security monitoring and logging system that provides real-time threat detection, automated response, and detailed security analytics. This system enhances the API's security posture by detecting, logging, and responding to various types of security threats.

## Architecture Components

### 1. Security Monitoring Core (`/src/api/security/monitoring.py`)

**Central Security Monitor**:
- Real-time event collection and analysis
- Threat pattern detection and classification
- Automated IP blocking and alerting
- Comprehensive security statistics

**Key Features**:
- **15 Security Event Types**: Authentication failures, injection attempts, DoS attacks, etc.
- **4 Threat Levels**: Low, Medium, High, Critical
- **Attack Pattern Detection**: Brute force, DoS, scanning, injection attempts
- **Automated Response**: IP blocking, rate limiting, alerting
- **Memory-Efficient**: Bounded collections with configurable limits

### 2. Security Monitoring Middleware (`/src/api/middleware/security_monitoring.py`)

**Real-Time Request Analysis**:
- Pre-request security validation
- Malicious payload detection
- Post-request security analysis
- Automatic threat blocking

**Detection Capabilities**:
- **User Agent Analysis**: Detects known attack tools (sqlmap, nikto, nmap, etc.)
- **Injection Pattern Detection**: SQL, XSS, Command, Path Traversal, LDAP, NoSQL, XXE
- **Suspicious Path Detection**: Admin paths, backup files, configuration files
- **Header Validation**: Injection attempts, abnormal sizes
- **Request Body Analysis**: Malicious payloads, oversized requests

### 3. Security Management API (`/src/api/routes/security_monitoring.py`)

**Administrative Endpoints**:
```
GET  /api/security/monitoring/health          # System health check
GET  /api/security/monitoring/stats           # Security statistics  
GET  /api/security/monitoring/events          # Recent security events
GET  /api/security/monitoring/threats/{ip}    # IP threat intelligence
GET  /api/security/monitoring/dashboard       # Comprehensive dashboard data
POST /api/security/monitoring/events          # Manual event logging
POST /api/security/monitoring/block-ip        # Block IP address
POST /api/security/monitoring/unblock-ip      # Unblock IP address
GET  /api/security/monitoring/blocked-ips     # List blocked IPs
POST /api/security/monitoring/cleanup         # Clean old events
```

### 4. Security Dashboard (`/src/api/security/dashboard.py`)

**Analytics and Reporting**:
- Real-time threat analysis
- Geographic threat distribution
- Timeline visualization data
- Security health assessment
- Automated recommendations

## Security Event Types

### Authentication & Authorization
- `AUTHENTICATION_FAILURE` - Failed login attempts
- `AUTHENTICATION_SUCCESS` - Successful authentications
- `AUTHORIZATION_FAILURE` - Access denied events
- `PRIVILEGE_ESCALATION` - Attempted privilege escalation

### Attack Patterns
- `INJECTION_ATTEMPT` - SQL, XSS, Command injection attempts
- `DOS_ATTEMPT` - Denial of service patterns
- `BRUTE_FORCE` - Brute force authentication attacks
- `SECURITY_SCAN` - Automated security scanning
- `API_ABUSE` - API misuse and abuse

### Protocol Violations
- `CORS_VIOLATION` - Cross-origin policy violations
- `RATE_LIMIT_EXCEEDED` - Rate limiting triggers
- `MALFORMED_REQUEST` - Invalid request formats

### Data & Session Security
- `DATA_ACCESS_ANOMALY` - Unusual data access patterns
- `SESSION_HIJACK` - Session hijacking attempts
- `SUSPICIOUS_REQUEST` - General suspicious activity

## Threat Detection Patterns

### 1. Injection Attack Detection
```python
# SQL Injection Patterns
r"(\bUNION\b.*\bSELECT\b|\bSELECT\b.*\bFROM\b.*\bWHERE\b)"
r"(\'\s*OR\s*\'\d+\'\s*=\s*\'\d+|\'\s*OR\s*\d+\s*=\s*\d+)"

# XSS Patterns  
r"(<script[^>]*>.*?</script>|javascript\s*:|vbscript\s*:)"
r"(onload\s*=|onerror\s*=|onclick\s*=|onmouseover\s*=)"

# Command Injection
r"(\|\s*nc\s+|\|\s*netcat\s+|\|\s*telnet\s+)"
r"(\$\(.*\)|`.*`|;.*whoami|;.*id|;.*passwd)"
```

### 2. Malicious User Agents
```python
malicious_agents = [
    "sqlmap", "nikto", "nmap", "dirbuster", "gobuster", "wfuzz",
    "burp", "nessus", "openvas", "acunetix", "metasploit", "hydra"
]
```

### 3. Suspicious Paths
```python
suspicious_paths = [
    r"\.php$", r"\.asp$", r"\.jsp$", r"\.cgi$",
    r"/admin", r"/login", r"/config", r"/backup",
    r"\.git/", r"\.env", r"\.sql$", r"\.bak$"
]
```

## Automated Response Mechanisms

### 1. Real-Time Blocking
- **Malicious User Agents**: Immediate blocking of known attack tools
- **Injection Attempts**: Block requests containing malicious patterns
- **Suspicious Paths**: Block access to sensitive file paths
- **Header Injection**: Block requests with malformed headers

### 2. Behavior-Based Blocking
- **Brute Force**: Block after 10 failed auth attempts per hour
- **DoS Protection**: Block after >120 requests per minute
- **Scanning Detection**: Flag after >20 unique endpoints per minute
- **Auto-Block**: Permanent blocking after 50 blocked requests

### 3. Rate Limiting
- **Preflight Requests**: Maximum 30 OPTIONS requests per minute
- **Request Frequency**: Monitor request patterns per IP
- **Endpoint Diversity**: Track scanning behavior

## Security Logging

### 1. Structured Logging
**Log Formats**:
- **Security Events**: `/app/logs/security/security_events.log`
- **Audit Trail**: `/app/logs/security/audit.log`
- **Threat Intelligence**: `/app/logs/security/threats.log`
- **JSON Analytics**: `/app/logs/security/security_events_YYYYMMDD.json`

**Log Rotation**:
- Daily log rotation
- 30-day retention by default
- Automatic archival of old logs

### 2. Event Structure
```json
{
  "timestamp": "2025-07-08T14:23:28.661258",
  "event_type": "injection_attempt",
  "threat_level": "high",
  "source_ip": "192.168.1.100",
  "user_agent": "sqlmap/1.0",
  "endpoint": "/api/data",
  "method": "GET",
  "user_id": null,
  "session_id": null,
  "origin": null,
  "description": "Suspicious user agent detected: sqlmap",
  "metadata": {},
  "blocked": true
}
```

## Dashboard and Analytics

### 1. Overview Statistics
- Total events (last hour/24h)
- Blocked events and success rate
- Unique source IPs
- Threat level distribution
- Currently blocked/suspicious IPs

### 2. Threat Analysis
- Attack pattern trends
- Most targeted endpoints
- Event type distribution
- Attack success rates

### 3. Geographic Analysis
- Regional threat distribution
- International vs. local threats
- IP range analysis

### 4. Timeline Data
- Hourly event buckets
- Peak activity periods
- Trend analysis

### 5. Security Health Assessment
```python
# Health Score Calculation (0-100)
health_score = 100
- (high_activity_penalty)
- (blocked_ips_penalty) 
- (critical_events * 5)
- (high_events * 2)

Status Levels:
- 90-100: EXCELLENT (Green)
- 75-89:  GOOD (Green)  
- 60-74:  FAIR (Yellow)
- 40-59:  POOR (Orange)
- 0-39:   CRITICAL (Red)
```

## Configuration and Tuning

### 1. Detection Thresholds
```python
config = {
    "max_failed_auth_per_hour": 10,
    "max_requests_per_minute": 120,
    "max_endpoints_per_minute": 20,
    "blocked_request_threshold": 50,
    "alert_cooldown_minutes": 15
}
```

### 2. Memory Management
- **Event Buffer**: 10,000 events in memory
- **IP Tracking**: 100 events per IP
- **Preflight Tracking**: 20 requests per IP
- **Automatic Cleanup**: 30-day retention

### 3. Performance Optimization
- **Pattern Compilation**: Pre-compiled regex patterns
- **Bounded Collections**: Memory-efficient data structures
- **Async Processing**: Non-blocking security checks
- **Lazy Evaluation**: On-demand analysis

## Integration Points

### 1. Middleware Integration
```python
# Automatic integration in FastAPI middleware stack
app.add_middleware(SecurityMonitoringMiddleware, environment=settings.APP_ENV)
```

### 2. Event Logging Integration
```python
# Manual event logging from application code
await log_security_event(
    SecurityEventType.AUTHENTICATION_FAILURE,
    ThreatLevel.MEDIUM,
    source_ip, user_agent, endpoint, method,
    "Custom security event description"
)
```

### 3. Monitoring Integration
- **Prometheus Metrics**: Security event counters
- **Health Checks**: System monitoring integration
- **Alerting**: External alert system hooks

## Testing and Validation

### 1. Attack Simulation
```bash
# Test malicious user agent detection
curl -H "User-Agent: sqlmap/1.0" http://localhost:3000/health
# Should return: 403 Forbidden - Request blocked

# Test injection pattern detection  
curl "http://localhost:3000/api/data?id=1' OR '1'='1" 
# Should return: 403 Forbidden - Injection attempt detected

# Test scanning behavior
for i in {1..25}; do curl "http://localhost:3000/api/endpoint$i" & done
# Should trigger: Security scanning detection
```

### 2. API Testing
```bash
# Get security statistics
curl -H "X-API-Key: admin" http://localhost:3000/api/security/monitoring/stats

# Check threat intelligence for IP
curl -H "X-API-Key: admin" http://localhost:3000/api/security/monitoring/threats/192.168.1.100

# Get security dashboard
curl -H "X-API-Key: admin" http://localhost:3000/api/security/monitoring/dashboard
```

### 3. Health Monitoring
```bash
# System health check (no auth required)
curl http://localhost:3000/api/security/monitoring/health

# Expected healthy response:
{
  "status": "healthy",
  "monitoring": "active", 
  "events_tracked": 0,
  "ips_tracked": 0,
  "blocked_ips": 0,
  "version": "1.0.0"
}
```

## Security Recommendations

### 1. Automated Security Actions
- **Immediate Blocking**: High-confidence threats (known attack tools)
- **Progressive Penalties**: Escalating responses for repeated violations
- **Temporary Blocks**: Time-based blocking for lower-confidence threats
- **Allowlist Management**: Automatic cleanup of outdated blocks

### 2. Alert Integration
```python
# Production integrations (currently placeholders)
async def _send_to_security_monitoring(self, event_data):
    # Integrate with:
    # - Email/SMS alerting systems
    # - Slack/Teams notifications  
    # - SIEM systems (Splunk, ELK)
    # - Incident response (PagerDuty, Opsgenie)
    pass
```

### 3. Continuous Improvement
- **Pattern Updates**: Regular updates to detection patterns
- **Threshold Tuning**: Adjustment based on false positive rates
- **New Threat Types**: Addition of emerging attack patterns
- **Performance Monitoring**: Optimization based on usage patterns

## Compliance and Standards

### 1. Security Frameworks
- **OWASP Top 10**: Coverage of major web application risks
- **NIST Cybersecurity Framework**: Detect, Respond, Recover
- **ISO 27001**: Information security management
- **PCI DSS**: Payment card industry standards (where applicable)

### 2. Privacy Considerations
- **Data Minimization**: Only essential security data collected
- **Retention Limits**: Automatic cleanup of old security events
- **Access Controls**: Admin-only access to security data
- **Anonymization**: No personal data in security logs

### 3. Audit Requirements
- **Complete Audit Trail**: All security events logged
- **Tamper Evidence**: Structured, timestamped logs
- **Retention Policy**: Configurable retention periods
- **Export Capabilities**: Security data export for compliance

## Maintenance and Operations

### 1. Regular Maintenance
- **Log Rotation**: Daily rotation with automatic archival
- **Event Cleanup**: Automated cleanup of old events
- **Performance Review**: Regular assessment of detection accuracy
- **Pattern Updates**: Quarterly review of detection patterns

### 2. Monitoring Health
- **System Health**: Regular health checks
- **False Positive Rate**: Monitor and tune detection thresholds
- **Response Time**: Ensure minimal performance impact
- **Alert Fatigue**: Balance between security and usability

### 3. Incident Response
- **Escalation Procedures**: Clear escalation paths for threats
- **Investigation Tools**: Detailed event analysis capabilities
- **Response Coordination**: Integration with incident response teams
- **Post-Incident Analysis**: Learning from security events

The security monitoring system provides comprehensive protection while maintaining usability and performance, establishing KnowledgeHub as a security-first application with enterprise-grade threat detection and response capabilities.