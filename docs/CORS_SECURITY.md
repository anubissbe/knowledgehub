# CORS Security Configuration

## Overview

The KnowledgeHub API implements secure Cross-Origin Resource Sharing (CORS) to replace the previous wildcard (`*`) configuration that was identified as a security vulnerability. This implementation provides:

- **Environment-specific origin allowlists** instead of wildcard permissions
- **Enhanced security middleware** with attack detection
- **Security header injection** for additional protection
- **Origin validation and monitoring** capabilities
- **Management API** for CORS configuration oversight

## Security Improvements

### Before (Insecure)
```javascript
// Previous configuration allowed all origins
allow_origins: ["*"]  // ❌ Security vulnerability
```

### After (Secure)
```javascript
// New configuration uses specific origin allowlists
development: [
  "http://localhost:3000",
  "http://localhost:3102", 
  "http://192.168.1.25:3000",
  // ... specific origins only
]
production: [
  "https://knowledgehub.example.com",
  "https://api.knowledgehub.example.com"
  // ... HTTPS-only in production
]
```

## Implementation Components

### 1. CORS Configuration (`/src/api/cors_config.py`)

**Key Features:**
- Environment-specific origin lists (development, staging, production)
- Strict security headers configuration
- Origin validation functions
- Support for localhost, local network, and secure HTTPS origins

**Configuration Structure:**
```python
class CORSSecurityConfig:
    allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    allowed_headers = ["Accept", "Content-Type", "Authorization", "X-API-Key"]
    exposed_headers = ["Content-Length", "X-Rate-Limit-Remaining"]
    max_age = 86400  # 24 hours preflight cache
```

### 2. Enhanced Security Middleware (`/src/api/middleware/cors_security.py`)

**Security Features:**
- Origin format validation (RFC 6454 compliance)
- Suspicious pattern detection (XSS, injection attempts)
- Rate limiting for preflight requests (30/minute)
- Attack pattern recognition and logging
- Dynamic Content Security Policy based on origin type

**Attack Detection Patterns:**
- Script injection attempts (`<script`, `javascript:`)
- Protocol-based attacks (`file://`, `ftp://`, `ldap://`)
- Path traversal attempts (`../`)
- Data URL exploits (`data:text/html`)

### 3. Management API (`/src/api/routes/cors_security.py`)

**Available Endpoints:**
```
GET  /api/security/cors/config           # Get current CORS configuration
GET  /api/security/cors/security/stats   # Get security statistics
GET  /api/security/cors/health           # CORS system health check
GET  /api/security/cors/origins/validate/{origin}  # Validate specific origin
POST /api/security/cors/origins/block    # Block suspicious origin
POST /api/security/cors/origins/unblock  # Unblock origin
POST /api/security/cors/security/test    # Test CORS request simulation
```

## Environment Configuration

### Development Environment
- **Origins**: localhost, 127.0.0.1, local network IPs
- **Protocols**: HTTP allowed for development ease
- **Strict Mode**: Disabled (warnings only)
- **CSP**: Relaxed to allow development tools

### Production Environment  
- **Origins**: Specific HTTPS domains only
- **Protocols**: HTTPS required
- **Strict Mode**: Enabled (blocks unauthorized requests)
- **CSP**: Strict security policy

### Staging Environment
- **Origins**: Combination of development and production origins
- **Protocols**: Both HTTP and HTTPS supported
- **Strict Mode**: Enabled with comprehensive logging

## Security Headers Applied

### Standard Security Headers
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Resource-Policy: cross-origin
```

### Dynamic Content Security Policy
- **Localhost/Development**: Relaxed CSP with `unsafe-inline` and `unsafe-eval`
- **Production**: Strict CSP with `'self'` only

## Monitoring and Logging

### Security Event Logging
All suspicious activities are logged with structured data:
```json
{
  "timestamp": "2025-07-08T13:54:00Z",
  "event_type": "suspicious_origin_pattern",
  "origin": "http://malicious-site.com",
  "method": "GET",
  "path": "/api/data",
  "user_agent": "...",
  "ip_address": "192.168.1.100"
}
```

### Attack Detection
- **DoS Detection**: >50 requests/minute from single origin
- **Scanning Detection**: >20 different paths in short time
- **Pattern Detection**: Malicious patterns in origin URLs
- **Rate Limiting**: Preflight request throttling

## Testing and Verification

### Manual Testing
```bash
# Test allowed origin
curl -I -H "Origin: http://localhost:3000" http://localhost:3000/health
# Should return: access-control-allow-origin: http://localhost:3000

# Test blocked origin  
curl -I -H "Origin: http://evil.com" http://localhost:3000/health
# Should NOT return: access-control-allow-origin header
```

### API Testing
```bash
# Get CORS configuration
curl -H "X-API-Key: admin" http://localhost:3000/api/security/cors/config

# Test origin validation
curl -H "X-API-Key: admin" http://localhost:3000/api/security/cors/origins/validate/http://localhost:3000

# Get security statistics
curl -H "X-API-Key: admin" http://localhost:3000/api/security/cors/security/stats
```

## Migration Impact

### Breaking Changes
- **Wildcard origins no longer supported** - Frontend applications must be explicitly allowlisted
- **Enhanced security validation** - Some previously allowed requests may now be blocked

### Compatibility
- **Existing legitimate clients** continue to work if origins are properly configured
- **Development environments** remain functional with localhost allowlisting
- **Production deployments** require HTTPS origin configuration

## Configuration Management

### Adding New Origins
1. Update `cors_config.py` with new origin in appropriate environment list
2. Restart API service to apply changes
3. Verify using the validation endpoint

### Emergency Origin Blocking
```bash
# Block suspicious origin immediately
curl -X POST -H "X-API-Key: admin" http://localhost:3000/api/security/cors/origins/block \
  -d '{"origin": "http://malicious-site.com", "reason": "Security incident"}'
```

## Performance Impact

- **Minimal overhead**: Origin validation is O(1) with efficient string matching
- **Caching enabled**: Preflight responses cached for 24 hours
- **Rate limiting**: Prevents DoS via excessive preflight requests
- **Memory efficient**: Bounded collections for tracking

## Compliance and Standards

- **RFC 6454**: Origin concept compliance
- **OWASP CORS**: Security best practices implementation
- **W3C CORS**: Standard-compliant preflight handling
- **Security Headers**: Industry-standard protective headers

## Maintenance

### Regular Tasks
1. **Review security logs** for suspicious activity patterns
2. **Update origin allowlists** as new environments are added
3. **Monitor attack detection** metrics and tune thresholds
4. **Validate configuration** after deployments

### Troubleshooting
- Check origin format (must include protocol)
- Verify environment-specific configuration
- Review security logs for blocked requests
- Use validation endpoint for testing