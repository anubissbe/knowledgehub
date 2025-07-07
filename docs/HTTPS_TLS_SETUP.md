# HTTPS/TLS Security Implementation

## Overview

KnowledgeHub has been configured with comprehensive HTTPS/TLS security using Traefik reverse proxy.

## Configuration

### Traefik Setup
- **HTTP Port**: 8080 (redirects to HTTPS)
- **HTTPS Port**: 8443 (TLS termination)
- **Dashboard**: 8082 (API access)

### SSL Certificates
- **Development**: Self-signed certificates in `traefik/certs/`
- **Production**: Configure Let's Encrypt in `traefik/traefik.yml`

### Security Headers
All HTTPS responses include comprehensive security headers:

- `Content-Security-Policy`: Strict CSP preventing XSS
- `Strict-Transport-Security`: HSTS with preload
- `X-Frame-Options`: DENY (prevents clickjacking)
- `X-Content-Type-Options`: nosniff
- `X-XSS-Protection`: 1; mode=block
- `Referrer-Policy`: strict-origin-when-cross-origin
- `Permissions-Policy`: Restrictive permissions

## Access Points

### Web UI
- **HTTPS**: https://localhost:8443/
- **HTTP**: http://localhost:8080/ (redirects to HTTPS)

### API
- **HTTPS**: https://api.localhost:8443/
- **HTTP**: http://api.localhost:8080/ (redirects to HTTPS)

### Traefik Dashboard
- **HTTP**: http://localhost:8082/

## Files Structure

```
traefik/
├── traefik.yml           # Main configuration
├── dynamic/
│   ├── certs.yml        # Certificate configuration
│   └── middlewares.yml  # Security middleware
└── certs/
    ├── tls.crt          # SSL certificate
    └── tls.key          # SSL private key
```

## Environment Variables

```bash
# HTTPS/TLS Configuration
TRAEFIK_DASHBOARD_USER=admin
TRAEFIK_DASHBOARD_PASSWORD=secure_dashboard_password_change_me
TLS_CERT_EMAIL=admin@knowledgehub.local

# CORS Configuration (includes HTTPS origins)
CORS_ORIGINS=["https://localhost:8443", "https://api.localhost:8443", "http://localhost:8080", "http://localhost:3001", "http://localhost:3000"]
```

## Verification

### Test HTTPS Access
```bash
# Web UI
curl -k https://localhost:8443/

# Check security headers
curl -k -s -D /tmp/headers https://localhost:8443/ > /dev/null && cat /tmp/headers
```

### Verify Traefik Configuration
```bash
# Check services
curl -s http://localhost:8082/api/http/services

# Check routers
curl -s http://localhost:8082/api/http/routers
```

## Production Setup

For production deployment:

1. **Update SSL Certificates**:
   - Replace self-signed certificates with proper CA certificates
   - Or configure Let's Encrypt automatic certificates

2. **Update Domain Names**:
   - Replace `localhost` with actual domain names
   - Update CORS origins accordingly

3. **Security Enhancements**:
   - Disable Traefik API (`api.insecure: false`)
   - Use strong dashboard credentials
   - Configure proper certificate email for Let's Encrypt

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Use alternative ports if 80/443 are occupied
2. **Certificate Errors**: Regenerate self-signed certificates if needed
3. **Router Discovery**: Check Docker network configuration in `traefik.yml`

### Generate New Self-Signed Certificates
```bash
cd traefik/certs
openssl req -x509 -newkey rsa:4096 -keyout tls.key -out tls.crt -days 365 -nodes \
  -subj "/C=US/ST=Development/L=Local/O=KnowledgeHub/OU=Development/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,DNS:*.localhost,DNS:127.0.0.1,DNS:knowledgehub.local,DNS:*.knowledgehub.local,IP:127.0.0.1,IP:::1"
```

## Security Features Implemented

✅ **TLS/HTTPS Encryption**: All traffic encrypted in transit  
✅ **Security Headers**: Comprehensive security header suite  
✅ **HTTP to HTTPS Redirect**: Automatic HTTPS enforcement  
✅ **HSTS**: Strict Transport Security with preload  
✅ **CSP**: Content Security Policy preventing XSS  
✅ **Clickjacking Protection**: X-Frame-Options DENY  
✅ **MIME Sniffing Protection**: X-Content-Type-Options nosniff  
✅ **Referrer Policy**: Strict origin policy  
✅ **Permissions Policy**: Restrictive feature permissions  

This implementation provides enterprise-grade HTTPS/TLS security for the KnowledgeHub platform.