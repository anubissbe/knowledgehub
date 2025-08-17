
from fastapi import Request
from fastapi.responses import Response
import hashlib
import secrets

class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # Security headers
                    headers[b"x-content-type-options"] = b"nosniff"
                    headers[b"x-frame-options"] = b"DENY"
                    headers[b"x-xss-protection"] = b"1; mode=block"
                    headers[b"strict-transport-security"] = b"max-age=31536000; includeSubDomains"
                    headers[b"referrer-policy"] = b"strict-origin-when-cross-origin"
                    headers[b"permissions-policy"] = b"geolocation=(), microphone=(), camera=()"
                    
                    # CSP with nonce for scripts
                    nonce = secrets.token_urlsafe(16)
                    csp = f"default-src 'self'; script-src 'self' 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'"
                    headers[b"content-security-policy"] = csp.encode()
                    
                    message["headers"] = list(headers.items())
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)
