# Security Headers and CSRF Protection

## Overview

The Security Headers and CSRF Protection system provides comprehensive HTTP security headers and Cross-Site Request Forgery (CSRF) protection for the KnowledgeHub API. This system implements multiple layers of security controls to protect against common web vulnerabilities and attacks.

## Features

### 1. HTTP Security Headers

The system implements comprehensive HTTP security headers with three configurable security levels:

#### Security Levels

- **STRICT**: Maximum security with restrictive policies
- **MODERATE**: Balanced security and compatibility (default)
- **PERMISSIVE**: Minimal security for maximum compatibility

#### Applied Security Headers

**Core Headers (All Levels):**
- `X-Content-Type-Options: nosniff` - Prevents MIME type sniffing
- `X-Frame-Options: DENY` - Prevents clickjacking attacks
- `X-XSS-Protection: 1; mode=block` - Enables browser XSS protection
- `Referrer-Policy: strict-origin-when-cross-origin` - Controls referrer information
- `Server: KnowledgeHub` - Custom server identification
- `Cache-Control: no-cache, no-store, must-revalidate` - Prevents caching of sensitive content

**Level-Specific Headers:**

**STRICT Mode:**
- `Strict-Transport-Security: max-age=31536000; includeSubDomains; preload`
- `Expect-CT: max-age=86400, enforce`
- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Resource-Policy: same-origin`
- `X-DNS-Prefetch-Control: off`

**MODERATE Mode:**
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Cross-Origin-Embedder-Policy: unsafe-none`
- `Cross-Origin-Opener-Policy: same-origin-allow-popups`
- `Cross-Origin-Resource-Policy: cross-origin`

**PERMISSIVE Mode:**
- `Strict-Transport-Security: max-age=86400`
- `Cross-Origin-Resource-Policy: cross-origin`

### 2. Content Security Policy (CSP)

Dynamic Content Security Policy implementation with nonce support:

#### CSP Directives by Security Level

**STRICT Mode:**
```
default-src 'self';
script-src 'self';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https:;
font-src 'self';
connect-src 'self';
media-src 'self';
object-src 'none';
child-src 'none';
worker-src 'self';
frame-ancestors 'none';
form-action 'self';
base-uri 'self';
manifest-src 'self';
upgrade-insecure-requests;
block-all-mixed-content;
```

**MODERATE Mode:**
```
default-src 'self';
script-src 'self' 'unsafe-inline';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https:;
font-src 'self' https:;
connect-src 'self' https:;
media-src 'self' https:;
object-src 'none';
child-src 'self';
worker-src 'self';
frame-ancestors 'self';
form-action 'self';
base-uri 'self';
manifest-src 'self';
```

**PERMISSIVE Mode:**
```
default-src 'self' 'unsafe-inline' 'unsafe-eval';
script-src 'self' 'unsafe-inline' 'unsafe-eval';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https: http:;
font-src 'self' https: http:;
connect-src 'self' https: http: ws: wss:;
media-src 'self' https: http:;
object-src 'self';
child-src 'self';
worker-src 'self';
frame-ancestors 'self';
form-action 'self';
base-uri 'self';
```

#### CSP Nonce Support

- Dynamic nonce generation for inline scripts
- Nonce is automatically added to CSP header when available
- Nonce is exposed via `X-CSP-Nonce` header for client-side use

### 3. Feature/Permissions Policy

Controls browser features and API access:

#### Feature Policy by Security Level

**STRICT Mode:**
```
accelerometer='none',
camera='none',
geolocation='none',
gyroscope='none',
magnetometer='none',
microphone='none',
payment='none',
usb='none',
interest-cohort=(),
browsing-topics=()
```

**MODERATE Mode:**
```
accelerometer='self',
camera='none',
geolocation='none',
gyroscope='self',
magnetometer='none',
microphone='none',
payment='none',
usb='none',
interest-cohort=(),
browsing-topics=()
```

**PERMISSIVE Mode:**
```
interest-cohort=(),
browsing-topics=()
```

### 4. CSRF Protection

Comprehensive Cross-Site Request Forgery protection system:

#### CSRF Token Management

- **Token Generation**: Secure random tokens with timestamp and session binding
- **Token Validation**: Multi-factor validation including session ID and expiration
- **Token Lifetime**: Configurable token expiration (default: 1 hour)
- **Token Storage**: In-memory storage with automatic cleanup

#### CSRF Configuration

```python
CSRFConfig(
    enabled=True,
    cookie_name="csrf_token",
    header_name="X-CSRF-Token",
    token_length=32,
    token_lifetime=3600,
    same_site="Strict",
    secure=True,
    httponly=True,
    require_referer=True,
    trusted_origins={"http://localhost:3000", "https://api.example.com"}
)
```

#### CSRF Protection Logic

- **Token Binding**: Tokens are bound to session ID (IP + User-Agent)
- **Referer Validation**: Optional referer header validation
- **Trusted Origins**: Configurable list of trusted origins
- **Automatic Cleanup**: Periodic cleanup of expired tokens

#### CSRF Exemptions

The following requests are exempt from CSRF protection:
- GET, HEAD, OPTIONS requests
- Requests with valid API key authentication
- JSON API requests with `X-Requested-With: XMLHttpRequest` header
- Configured exempt paths (health checks, documentation, etc.)

## Implementation

### 1. Middleware Integration

The security headers system is implemented as FastAPI middleware:

```python
from .middleware.security_headers import SecurityHeadersMiddleware
from .security.headers import SecurityHeaderLevel

# Configure security level based on environment
security_level = SecurityHeaderLevel.STRICT if settings.APP_ENV == "production" else SecurityHeaderLevel.MODERATE

# Add middleware to application
app.add_middleware(
    SecurityHeadersMiddleware,
    security_level=security_level,
    csrf_enabled=True,
    environment=settings.APP_ENV
)
```

### 2. Core Components

#### SecurityHeadersManager

Central manager for all security headers functionality:

```python
class SecurityHeadersManager:
    def __init__(self, level: SecurityHeaderLevel = SecurityHeaderLevel.MODERATE,
                 csrf_config: Optional[CSRFConfig] = None):
        self.level = level
        self.csrf_config = csrf_config or CSRFConfig()
        self.csrf_tokens = {}
        self.security_headers = self._get_security_headers()
        self.csp_config = self._get_csp_config()
        self.feature_policy = self._get_feature_policy()
```

#### SecurityHeadersMiddleware

FastAPI middleware for applying security headers:

```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, 
                 security_level: SecurityHeaderLevel = SecurityHeaderLevel.MODERATE,
                 csrf_enabled: bool = True,
                 environment: str = "development"):
        super().__init__(app)
        self.environment = environment
        self.security_level = security_level
        # Initialize CSRF configuration and headers manager
```

### 3. API Endpoints

The system provides comprehensive API endpoints for security headers management:

#### Health Check
- `GET /api/security/headers/health` - Health status (no auth required)

#### Security Headers Status
- `GET /api/security/headers/status` - Current configuration and status

#### CSRF Token Management
- `POST /api/security/headers/csrf/token` - Generate new CSRF token
- `POST /api/security/headers/csrf/validate` - Validate CSRF token
- `GET /api/security/headers/csrf/stats` - CSRF protection statistics
- `POST /api/security/headers/csrf/cleanup` - Clean up expired tokens

#### Trusted Origins Management
- `POST /api/security/headers/csrf/trusted-origins` - Add trusted origin
- `DELETE /api/security/headers/csrf/trusted-origins` - Remove trusted origin
- `GET /api/security/headers/csrf/trusted-origins` - List trusted origins

#### Testing and Debugging
- `GET /api/security/headers/test` - Test security headers configuration

## Configuration

### Environment-Based Configuration

The system automatically configures security levels based on environment:

```python
# Production environment
if settings.APP_ENV == "production":
    security_level = SecurityHeaderLevel.STRICT
    csrf_config = CSRFConfig(
        secure=True,
        same_site="Strict",
        require_referer=True,
        trusted_origins={"https://knowledgehub.example.com"}
    )

# Development environment
else:
    security_level = SecurityHeaderLevel.MODERATE
    csrf_config = CSRFConfig(
        secure=False,
        same_site="Lax",
        require_referer=False,
        trusted_origins={"http://localhost:3000", "http://localhost:3102"}
    )
```

### Trusted Origins Configuration

Environment-specific trusted origins for CSRF protection:

**Development:**
- `http://localhost:3000`
- `http://localhost:3102`
- `http://localhost:5173`
- `http://127.0.0.1:3000`
- `http://192.168.1.24:3000`

**Production:**
- `https://knowledgehub.example.com`
- `https://api.knowledgehub.example.com`
- `https://app.knowledgehub.example.com`

## Security Features

### 1. Attack Prevention

The system protects against:
- **XSS (Cross-Site Scripting)**: CSP, X-XSS-Protection
- **Clickjacking**: X-Frame-Options, frame-ancestors CSP
- **MIME Type Sniffing**: X-Content-Type-Options
- **CSRF**: Token-based protection with session binding
- **Information Disclosure**: Custom server headers, cache control
- **Mixed Content**: CSP upgrade-insecure-requests
- **Tracking**: Feature policy disabling cohort/topics

### 2. Session Security

- **Session Binding**: CSRF tokens bound to IP + User-Agent
- **Token Expiration**: Configurable token lifetime
- **Automatic Cleanup**: Periodic removal of expired tokens
- **One-Time Use**: Optional token consumption on validation

### 3. Monitoring and Logging

- **Security Event Logging**: CSRF violations logged as security events
- **Performance Metrics**: Token generation and validation metrics
- **Health Monitoring**: System health checks and status reporting
- **Statistics**: Detailed CSRF protection statistics

## Testing

### 1. Security Headers Validation

Test that security headers are properly applied:

```bash
# Test security headers on health endpoint
curl -I http://localhost:3000/health

# Expected headers:
# X-Content-Type-Options: nosniff
# X-Frame-Options: DENY
# X-XSS-Protection: 1; mode=block
# Strict-Transport-Security: max-age=31536000; includeSubDomains
# Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' 'nonce-...'
# Permissions-Policy: accelerometer='self', camera='none', ...
```

### 2. CSRF Token Testing

Test CSRF token generation and validation:

```bash
# Generate CSRF token
curl -X POST -H "X-API-Key: admin" http://localhost:3000/api/security/headers/csrf/token

# Validate CSRF token
curl -X POST -H "X-API-Key: admin" \
  -H "Content-Type: application/json" \
  -d '{"csrf_token": "token_value"}' \
  http://localhost:3000/api/security/headers/csrf/validate
```

### 3. Health Check Testing

Test system health and configuration:

```bash
# Health check (no auth required)
curl http://localhost:3000/api/security/headers/health

# Configuration status
curl -H "X-API-Key: admin" http://localhost:3000/api/security/headers/status

# Test headers configuration
curl -H "X-API-Key: admin" http://localhost:3000/api/security/headers/test
```

## Integration with Other Security Systems

### 1. Security Monitoring Integration

- CSRF violations are logged as security events
- Integration with threat detection system
- Automatic IP blocking for repeated violations

### 2. CORS Security Integration

- Coordinated with CORS security middleware
- Shared origin validation logic
- Consistent security policy enforcement

### 3. Authentication Integration

- API key authentication bypass for CSRF
- Session-based authentication support
- Coordinated security header application

## Performance Considerations

### 1. Token Management

- In-memory token storage for performance
- Periodic cleanup to prevent memory leaks
- Configurable cleanup intervals

### 2. Header Application

- Efficient header application using middleware
- Minimal performance impact on responses
- Cached configuration for repeated use

### 3. Monitoring Overhead

- Lightweight security event logging
- Asynchronous event processing
- Minimal impact on request processing

## Troubleshooting

### Common Issues

1. **CSRF Token Validation Failures**
   - Check token expiration
   - Verify session binding
   - Validate trusted origins

2. **CSP Violations**
   - Check nonce implementation
   - Verify script/style sources
   - Review CSP policy configuration

3. **Performance Issues**
   - Monitor token cleanup frequency
   - Check memory usage
   - Review security event volume

### Debug Information

Enable debug logging to troubleshoot issues:

```python
# Enable debug logging
logging.getLogger("api.security.headers").setLevel(logging.DEBUG)
logging.getLogger("api.middleware.security_headers").setLevel(logging.DEBUG)
```

### Health Check Endpoints

Use health check endpoints to verify system status:

```bash
# System health
curl http://localhost:3000/api/security/headers/health

# CSRF statistics
curl -H "X-API-Key: admin" http://localhost:3000/api/security/headers/csrf/stats
```

## Future Enhancements

### 1. Enhanced Token Security

- JWT-based CSRF tokens
- Cryptographic token signing
- Token rotation strategies

### 2. Advanced CSP Features

- CSP reporting endpoint
- Dynamic CSP policy updates
- CSP violation analysis

### 3. Additional Security Headers

- Network Error Logging (NEL)
- Certificate Transparency monitoring
- Custom security headers

### 4. Performance Optimization

- Redis-based token storage
- Distributed token management
- Advanced caching strategies

## Security Best Practices

1. **Always use HTTPS in production**
2. **Regularly update security configurations**
3. **Monitor security event logs**
4. **Test security headers configuration**
5. **Keep trusted origins list minimal**
6. **Implement proper token cleanup**
7. **Use strict security levels in production**
8. **Regular security audits and testing**

---

**Status**: ✅ **IMPLEMENTED AND FUNCTIONAL**

The Security Headers and CSRF Protection system has been successfully implemented with comprehensive HTTP security headers, CSRF protection, and management APIs. The system provides robust protection against common web vulnerabilities while maintaining flexibility for different deployment environments.

**Key Features Implemented:**
- ✅ HTTP Security Headers (3 security levels)
- ✅ Content Security Policy with nonce support
- ✅ Feature/Permissions Policy
- ✅ CSRF token generation and validation
- ✅ Trusted origins management
- ✅ Security event logging
- ✅ Health monitoring and statistics
- ✅ Management API endpoints
- ✅ Environment-based configuration
- ✅ Automatic token cleanup
- ✅ Session-based token binding

**Last Updated**: July 8, 2025  
**Version**: 1.0.0  
**Environment**: Development/Production Ready