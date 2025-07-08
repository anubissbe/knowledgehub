# KnowledgeHub Security Quick Reference Card

## 🚨 Emergency Contacts
- **Security Hotline**: +1-XXX-XXX-XXXX (24/7)
- **Security Email**: security@knowledgehub.com
- **Incident Response**: incident-response@knowledgehub.com

## 🔐 Authentication & Authorization

### API Key Header
```bash
curl -H "X-API-Key: knhub_your-api-key-here" https://api.knowledgehub.com/endpoint
```

### Bearer Token
```bash
curl -H "Authorization: Bearer your-jwt-token" https://api.knowledgehub.com/endpoint
```

### Session Cookie
```javascript
// Secure cookie settings
{
  httpOnly: true,
  secure: true,
  sameSite: 'strict',
  maxAge: 1800000  // 30 minutes
}
```

## 🛡️ Input Validation Checklist

### Always Validate
- [ ] Data type (string, number, boolean)
- [ ] Length (min/max)
- [ ] Format (regex patterns)
- [ ] Range (numeric bounds)
- [ ] Whitelist values (enums)

### Common Patterns
```python
# Email
r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# Username (3-30 chars, alphanumeric + _ -)
r'^[a-zA-Z0-9_-]{3,30}$'

# Strong Password
r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

# UUID
r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

# Safe Filename
r'^[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$'
```

## 🚫 Security Anti-Patterns (NEVER DO)

### ❌ SQL Injection
```python
# WRONG
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ CORRECT
query = "SELECT * FROM users WHERE id = :user_id"
db.execute(query, {"user_id": user_id})
```

### ❌ XSS
```python
# WRONG
return f"<h1>Welcome {username}</h1>"

# ✅ CORRECT
from markupsafe import escape
return f"<h1>Welcome {escape(username)}</h1>"
```

### ❌ Hardcoded Secrets
```python
# WRONG
API_KEY = "sk-1234567890abcdef"

# ✅ CORRECT
API_KEY = os.environ.get("API_KEY")
# or
API_KEY = vault_client.get_secret("api_key")
```

### ❌ Weak Random
```python
# WRONG
import random
token = random.randint(100000, 999999)

# ✅ CORRECT
import secrets
token = secrets.token_urlsafe(32)
```

## 🔒 Security Headers

### Essential Headers
```python
{
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
}
```

### Content Security Policy
```
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';
```

## 🚦 Rate Limiting

### Standard Limits
- **API Endpoints**: 60 requests/minute
- **Search**: 30 requests/minute
- **AI Analysis**: 10 requests/minute
- **File Upload**: 5 requests/minute

### Response Headers
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1625097600
Retry-After: 30
```

## 🔍 Security Monitoring

### What to Log
- ✅ Failed authentication attempts
- ✅ Permission denied events
- ✅ Rate limit violations
- ✅ Suspicious patterns
- ✅ Data access (who, what, when)
- ❌ Passwords or tokens
- ❌ Full credit card numbers
- ❌ Personal sensitive data

### Log Format
```json
{
  "timestamp": "2025-07-08T10:30:00Z",
  "event_type": "auth.login.failure",
  "severity": "WARNING",
  "user_id": "user123",
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "details": {
    "reason": "invalid_password"
  }
}
```

## 🆘 Incident Response Steps

### 1. Detect
- Monitor alerts
- Check logs
- User reports

### 2. Contain
- Block attacker IP
- Disable compromised accounts
- Isolate affected systems

### 3. Investigate
- Analyze logs
- Identify attack vector
- Assess damage

### 4. Recover
- Apply patches
- Reset credentials
- Restore from backup

### 5. Learn
- Document incident
- Update procedures
- Train team

## 🛠️ Security Tools

### Static Analysis
```bash
# Python security scan
bandit -r src/

# Dependency check
safety check

# Secret scanning
trufflehog --regex --entropy=False .
```

### Dynamic Testing
```bash
# OWASP ZAP scan
docker run -t owasp/zap2docker-stable zap-baseline.py -t https://api.knowledgehub.com

# SSL/TLS test
nmap --script ssl-enum-ciphers -p 443 knowledgehub.com
```

## 📋 Pre-Deployment Checklist

- [ ] All inputs validated
- [ ] SQL queries parameterized
- [ ] XSS prevention in place
- [ ] Authentication required
- [ ] Authorization checked
- [ ] Rate limiting configured
- [ ] Security headers set
- [ ] HTTPS enforced
- [ ] Secrets in Vault
- [ ] Logging configured
- [ ] Error messages sanitized
- [ ] Dependencies updated

## 🔑 Secure Coding Functions

### Password Hashing
```python
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Hash
hashed = pwd_context.hash(password)

# Verify
valid = pwd_context.verify(password, hashed)
```

### Token Generation
```python
import secrets
# URL-safe token
token = secrets.token_urlsafe(32)

# Hex token
hex_token = secrets.token_hex(16)
```

### Input Sanitization
```python
# HTML
from markupsafe import escape
safe_html = escape(user_input)

# SQL identifiers
def sanitize_table_name(name: str) -> str:
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        raise ValueError("Invalid table name")
    return name
```

### CORS Configuration
```python
# Production
origins = [
    "https://app.knowledgehub.com",
    "https://www.knowledgehub.com"
]

# Development
origins = [
    "http://localhost:3000",
    "http://localhost:5173"
]
```

## 📱 Response Codes

### Security-Related HTTP Codes
- **400**: Bad Request (invalid input)
- **401**: Unauthorized (authentication required)
- **403**: Forbidden (insufficient permissions)
- **429**: Too Many Requests (rate limited)
- **503**: Service Unavailable (under attack)

### Standard Error Response
```json
{
  "error": "Invalid request",
  "message": "The provided input is invalid",
  "code": "INVALID_INPUT",
  "timestamp": "2025-07-08T10:30:00Z",
  "request_id": "req_123456"
}
```

---

**Quick Reference Version**: 1.0.0  
**Last Updated**: July 8, 2025  
**Print**: Landscape mode recommended

**Remember**: When in doubt, choose the more secure option!