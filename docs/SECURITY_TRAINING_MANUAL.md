# KnowledgeHub Security Training Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Security Fundamentals](#security-fundamentals)
3. [Hands-On Exercises](#hands-on-exercises)
4. [Security Scenarios](#security-scenarios)
5. [Best Practices Guide](#best-practices-guide)
6. [Quick Reference](#quick-reference)
7. [Assessment](#assessment)

## Introduction

Welcome to the KnowledgeHub Security Training Manual. This hands-on guide will help you understand and implement security best practices in the KnowledgeHub system.

### Training Objectives

By the end of this training, you will be able to:
- Identify common security vulnerabilities
- Implement secure coding practices
- Use KnowledgeHub security features effectively
- Respond to security incidents
- Maintain security compliance

### Prerequisites

- Basic understanding of Python and FastAPI
- Access to KnowledgeHub development environment
- Completion of general KnowledgeHub onboarding

## Security Fundamentals

### The CIA Triad

The foundation of information security rests on three principles:

#### 1. **Confidentiality**
Ensuring information is accessible only to authorized users.

**Example in KnowledgeHub**:
```python
# API endpoints protected by authentication
@router.get("/api/documents/{doc_id}")
@require_auth  # Ensures only authenticated users can access
async def get_document(doc_id: str, user: User = Depends(get_current_user)):
    # Check if user has permission to view this document
    if not await has_document_access(user.id, doc_id):
        raise HTTPException(403, "Access denied")
    return await fetch_document(doc_id)
```

#### 2. **Integrity**
Ensuring information remains accurate and unmodified.

**Example in KnowledgeHub**:
```python
# Input validation to maintain data integrity
class DocumentUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., max_length=50000)
    
    @validator('content')
    def validate_content(cls, v):
        # Ensure content doesn't contain malicious scripts
        return sanitize_html(v)
```

#### 3. **Availability**
Ensuring information is accessible when needed.

**Example in KnowledgeHub**:
```python
# Rate limiting to prevent DoS attacks
@router.get("/api/search")
@rate_limit(requests_per_minute=60)  # Prevent service overload
async def search(query: str):
    return await perform_search(query)
```

### Common Vulnerabilities

#### 1. SQL Injection

**Vulnerable Code** ❌:
```python
# NEVER DO THIS
query = f"SELECT * FROM users WHERE username = '{username}'"
db.execute(query)
```

**Secure Code** ✅:
```python
# ALWAYS DO THIS
query = "SELECT * FROM users WHERE username = :username"
db.execute(query, {"username": username})
```

#### 2. Cross-Site Scripting (XSS)

**Vulnerable Code** ❌:
```python
# NEVER DO THIS
@app.get("/profile")
async def profile(name: str):
    return f"<h1>Welcome {name}</h1>"  # name could contain <script> tags
```

**Secure Code** ✅:
```python
# ALWAYS DO THIS
from markupsafe import escape

@app.get("/profile")
async def profile(name: str):
    return f"<h1>Welcome {escape(name)}</h1>"
```

#### 3. Broken Authentication

**Vulnerable Code** ❌:
```python
# NEVER DO THIS
@app.post("/login")
async def login(username: str, password: str):
    if username == "admin" and password == "admin123":  # Hardcoded credentials
        return {"token": "secret-token"}
```

**Secure Code** ✅:
```python
# ALWAYS DO THIS
@app.post("/login")
async def login(credentials: LoginCredentials):
    user = await get_user_by_username(credentials.username)
    if not user or not verify_password(credentials.password, user.hashed_password):
        # Same error for both cases to prevent username enumeration
        raise HTTPException(401, "Invalid credentials")
    
    # Generate secure token
    token = create_access_token(user.id)
    return {"token": token}
```

## Hands-On Exercises

### Exercise 1: Input Validation

**Task**: Implement secure input validation for a user registration endpoint.

**Starting Code**:
```python
@router.post("/register")
async def register(username: str, email: str, password: str):
    # TODO: Add validation
    user = await create_user(username, email, password)
    return {"user_id": user.id}
```

**Solution**:
```python
from pydantic import BaseModel, EmailStr, validator
import re

class UserRegistration(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        
        # Check against reserved names
        if v.lower() in ['admin', 'root', 'system']:
            raise ValueError('Username is reserved')
        
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one number')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char in '!@#$%^&*()_+-=' for char in v):
            raise ValueError('Password must contain at least one special character')
        return v

@router.post("/register")
async def register(user_data: UserRegistration):
    # Check if username already exists
    if await username_exists(user_data.username):
        raise HTTPException(400, "Username already taken")
    
    # Check if email already exists
    if await email_exists(user_data.email):
        raise HTTPException(400, "Email already registered")
    
    # Hash password before storing
    hashed_password = hash_password(user_data.password)
    
    user = await create_user(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    
    return {"user_id": user.id, "message": "Registration successful"}
```

### Exercise 2: Implementing Rate Limiting

**Task**: Add rate limiting to protect an API endpoint from abuse.

**Starting Code**:
```python
@router.post("/api/analyze")
async def analyze_document(content: str):
    # TODO: Add rate limiting
    result = await expensive_ai_analysis(content)
    return {"analysis": result}
```

**Solution**:
```python
from src.api.security.rate_limiting import RateLimiter, RateLimitConfig

# Configure rate limiter
rate_limiter = RateLimiter(
    config=RateLimitConfig(
        requests_per_minute=10,
        requests_per_hour=100,
        burst_size=5
    )
)

@router.post("/api/analyze")
async def analyze_document(
    content: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    # Check rate limit
    client_id = request.client.host
    allowed, retry_after = await rate_limiter.check_rate_limit(client_id)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Validate content length
    if len(content) > 10000:
        raise HTTPException(400, "Content too long")
    
    # Log the request for monitoring
    background_tasks.add_task(
        log_api_usage,
        client_id=client_id,
        endpoint="/api/analyze",
        timestamp=datetime.utcnow()
    )
    
    # Perform analysis
    result = await expensive_ai_analysis(content)
    return {"analysis": result}
```

### Exercise 3: Secure Session Management

**Task**: Implement secure session handling with proper timeout and regeneration.

**Starting Code**:
```python
sessions = {}  # In-memory session storage

@router.post("/login")
async def login(username: str, password: str):
    # TODO: Implement secure session management
    if verify_credentials(username, password):
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"username": username}
        return {"session_id": session_id}
```

**Solution**:
```python
from datetime import datetime, timedelta
import secrets
import redis

# Use Redis for distributed session storage
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class SessionManager:
    def __init__(self, timeout_minutes: int = 30):
        self.timeout = timedelta(minutes=timeout_minutes)
        self.redis = redis_client
    
    def create_session(self, user_id: str, user_data: dict) -> str:
        # Generate cryptographically secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Store session data
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            **user_data
        }
        
        # Store in Redis with expiration
        self.redis.setex(
            f"session:{session_id}",
            self.timeout,
            json.dumps(session_data)
        )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        data = self.redis.get(f"session:{session_id}")
        if not data:
            return None
        
        session = json.loads(data)
        
        # Check if session is still valid
        last_activity = datetime.fromisoformat(session['last_activity'])
        if datetime.utcnow() - last_activity > self.timeout:
            self.destroy_session(session_id)
            return None
        
        # Update last activity
        session['last_activity'] = datetime.utcnow().isoformat()
        self.redis.setex(
            f"session:{session_id}",
            self.timeout,
            json.dumps(session)
        )
        
        return session
    
    def regenerate_session_id(self, old_session_id: str) -> Optional[str]:
        # Get existing session
        session = self.get_session(old_session_id)
        if not session:
            return None
        
        # Create new session with same data
        new_session_id = self.create_session(session['user_id'], session)
        
        # Destroy old session
        self.destroy_session(old_session_id)
        
        return new_session_id
    
    def destroy_session(self, session_id: str):
        self.redis.delete(f"session:{session_id}")

session_manager = SessionManager()

@router.post("/login")
async def login(credentials: LoginCredentials, response: Response):
    # Verify credentials
    user = await authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    
    # Create secure session
    session_id = session_manager.create_session(
        user_id=user.id,
        user_data={
            "username": user.username,
            "roles": user.roles
        }
    )
    
    # Set secure cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,  # HTTPS only
        samesite="strict",
        max_age=1800  # 30 minutes
    )
    
    return {"message": "Login successful"}

@router.post("/logout")
async def logout(session_id: str = Cookie(None)):
    if session_id:
        session_manager.destroy_session(session_id)
    return {"message": "Logged out successfully"}
```

### Exercise 4: API Key Management

**Task**: Implement secure API key generation and validation.

**Solution**:
```python
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta

class APIKeyManager:
    def __init__(self, db: Session):
        self.db = db
        self.key_prefix = "knhub_"
        self.key_length = 32
    
    def generate_api_key(self, user_id: str, name: str, expires_in_days: int = 90) -> dict:
        # Generate secure random key
        raw_key = secrets.token_urlsafe(self.key_length)
        api_key = f"{self.key_prefix}{raw_key}"
        
        # Create HMAC hash of the key
        key_hash = hmac.new(
            settings.SECRET_KEY.encode(),
            api_key.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Store in database
        api_key_record = APIKey(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
            last_used_at=None,
            is_active=True
        )
        
        self.db.add(api_key_record)
        self.db.commit()
        
        return {
            "api_key": api_key,  # Only returned once
            "key_id": str(api_key_record.id),
            "expires_at": api_key_record.expires_at.isoformat()
        }
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        # Validate format
        if not api_key.startswith(self.key_prefix):
            return None
        
        # Create hash
        key_hash = hmac.new(
            settings.SECRET_KEY.encode(),
            api_key.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Look up in database
        api_key_record = self.db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True,
            APIKey.expires_at > datetime.utcnow()
        ).first()
        
        if api_key_record:
            # Update last used
            api_key_record.last_used_at = datetime.utcnow()
            self.db.commit()
        
        return api_key_record
    
    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        api_key = self.db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == user_id
        ).first()
        
        if api_key:
            api_key.is_active = False
            api_key.revoked_at = datetime.utcnow()
            self.db.commit()
            return True
        
        return False

# Usage in endpoint
@router.post("/api-keys")
@require_auth
async def create_api_key(
    key_request: APIKeyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    manager = APIKeyManager(db)
    
    # Check if user has permission to create API keys
    if not current_user.can_create_api_keys:
        raise HTTPException(403, "Not authorized to create API keys")
    
    # Generate new API key
    result = manager.generate_api_key(
        user_id=current_user.id,
        name=key_request.name,
        expires_in_days=key_request.expires_in_days or 90
    )
    
    return {
        "api_key": result["api_key"],
        "key_id": result["key_id"],
        "expires_at": result["expires_at"],
        "warning": "Store this API key securely. It will not be shown again."
    }
```

## Security Scenarios

### Scenario 1: Suspicious Login Activity

**Situation**: Multiple failed login attempts detected from the same IP address.

**Detection Code**:
```python
class LoginMonitor:
    def __init__(self):
        self.failed_attempts = {}  # IP -> list of attempts
        self.max_attempts = 5
        self.window_minutes = 15
    
    async def check_login_attempt(self, ip: str, success: bool) -> Optional[SecurityAlert]:
        now = datetime.utcnow()
        
        if not success:
            # Track failed attempt
            if ip not in self.failed_attempts:
                self.failed_attempts[ip] = []
            
            self.failed_attempts[ip].append(now)
            
            # Clean old attempts
            cutoff = now - timedelta(minutes=self.window_minutes)
            self.failed_attempts[ip] = [
                t for t in self.failed_attempts[ip] if t > cutoff
            ]
            
            # Check if threshold exceeded
            if len(self.failed_attempts[ip]) >= self.max_attempts:
                return SecurityAlert(
                    type=AlertType.BRUTE_FORCE,
                    severity=Severity.HIGH,
                    source_ip=ip,
                    message=f"Multiple failed login attempts from {ip}",
                    recommended_action="Block IP temporarily"
                )
        else:
            # Clear failed attempts on success
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
        
        return None
```

**Response Actions**:
```python
async def handle_brute_force_alert(alert: SecurityAlert):
    # 1. Block IP temporarily
    await ip_blocker.block_ip(
        ip=alert.source_ip,
        duration_minutes=30,
        reason="Brute force detection"
    )
    
    # 2. Log security event
    await security_logger.log_event(
        event_type=SecurityEventType.BRUTE_FORCE_ATTEMPT,
        details=alert.dict(),
        severity=alert.severity
    )
    
    # 3. Send alert to security team
    await send_security_alert(
        to=SECURITY_TEAM_EMAIL,
        subject="Brute Force Attack Detected",
        alert=alert
    )
    
    # 4. Increase monitoring for this IP
    await monitoring.add_watch_list(alert.source_ip)
```

### Scenario 2: Data Exfiltration Attempt

**Situation**: Unusual data access patterns detected.

**Detection Code**:
```python
class DataAccessMonitor:
    def __init__(self):
        self.access_patterns = {}
        self.thresholds = {
            "documents_per_minute": 20,
            "total_size_mb": 100,
            "unique_resources": 50
        }
    
    async def monitor_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        size_bytes: int
    ) -> Optional[SecurityAlert]:
        now = datetime.utcnow()
        user_key = f"{user_id}:{now.minute}"
        
        if user_key not in self.access_patterns:
            self.access_patterns[user_key] = {
                "count": 0,
                "size_bytes": 0,
                "resources": set()
            }
        
        pattern = self.access_patterns[user_key]
        pattern["count"] += 1
        pattern["size_bytes"] += size_bytes
        pattern["resources"].add(resource_id)
        
        # Check thresholds
        if (pattern["count"] > self.thresholds["documents_per_minute"] or
            pattern["size_bytes"] / 1024 / 1024 > self.thresholds["total_size_mb"] or
            len(pattern["resources"]) > self.thresholds["unique_resources"]):
            
            return SecurityAlert(
                type=AlertType.DATA_EXFILTRATION,
                severity=Severity.CRITICAL,
                user_id=user_id,
                message="Potential data exfiltration detected",
                details={
                    "documents_accessed": pattern["count"],
                    "total_size_mb": pattern["size_bytes"] / 1024 / 1024,
                    "unique_resources": len(pattern["resources"])
                }
            )
        
        return None
```

### Scenario 3: Malicious File Upload

**Situation**: User attempts to upload a potentially malicious file.

**Detection Code**:
```python
import magic
import hashlib

class FileSecurityScanner:
    def __init__(self):
        self.blocked_extensions = {
            '.exe', '.dll', '.scr', '.bat', '.cmd', '.com',
            '.pif', '.vbs', '.js', '.jar', '.zip', '.rar'
        }
        self.allowed_mime_types = {
            'application/pdf',
            'image/jpeg',
            'image/png',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    async def scan_file(self, file_path: str, original_name: str) -> ScanResult:
        # Check file extension
        ext = os.path.splitext(original_name)[1].lower()
        if ext in self.blocked_extensions:
            return ScanResult(
                safe=False,
                reason=f"Blocked file extension: {ext}"
            )
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            return ScanResult(
                safe=False,
                reason=f"File too large: {file_size} bytes"
            )
        
        # Check MIME type
        mime = magic.from_file(file_path, mime=True)
        if mime not in self.allowed_mime_types:
            return ScanResult(
                safe=False,
                reason=f"Disallowed MIME type: {mime}"
            )
        
        # Check for embedded threats
        if await self.check_embedded_threats(file_path):
            return ScanResult(
                safe=False,
                reason="Embedded threats detected"
            )
        
        # Calculate file hash for tracking
        file_hash = self.calculate_file_hash(file_path)
        
        return ScanResult(
            safe=True,
            file_hash=file_hash,
            mime_type=mime,
            size=file_size
        )
    
    async def check_embedded_threats(self, file_path: str) -> bool:
        # Check for suspicious patterns in file content
        suspicious_patterns = [
            b'<script',
            b'javascript:',
            b'eval(',
            b'ActiveXObject',
            b'.exe',
            b'cmd.exe'
        ]
        
        with open(file_path, 'rb') as f:
            content = f.read(1024 * 1024)  # Read first 1MB
            
            for pattern in suspicious_patterns:
                if pattern in content:
                    return True
        
        return False
```

## Best Practices Guide

### Secure Coding Checklist

#### Before Writing Code
- [ ] Understand the security requirements
- [ ] Review similar secure implementations
- [ ] Plan input validation strategy
- [ ] Identify sensitive data flows

#### While Writing Code
- [ ] Validate all inputs
- [ ] Use parameterized queries
- [ ] Encode all outputs
- [ ] Implement proper error handling
- [ ] Add security logging

#### After Writing Code
- [ ] Run security linters
- [ ] Perform code review
- [ ] Write security tests
- [ ] Update documentation
- [ ] Check for secrets in code

### Common Security Patterns

#### 1. Defense in Depth
```python
@router.post("/sensitive-operation")
@rate_limit(10)  # Layer 1: Rate limiting
@require_auth    # Layer 2: Authentication
async def sensitive_operation(
    request: SecureRequest,  # Layer 3: Input validation
    user: User = Depends(get_current_user)
):
    # Layer 4: Authorization
    if not user.has_permission("sensitive_operation"):
        raise HTTPException(403, "Insufficient permissions")
    
    # Layer 5: Audit logging
    await audit_log(
        user_id=user.id,
        action="sensitive_operation",
        details=request.dict()
    )
    
    # Perform operation
    result = await perform_sensitive_operation(request)
    
    # Layer 6: Output sanitization
    return sanitize_response(result)
```

#### 2. Fail Secure
```python
async def check_access(user_id: str, resource_id: str) -> bool:
    try:
        # Check permissions
        permission = await get_permission(user_id, resource_id)
        return permission and permission.can_access
    except Exception as e:
        # Log error
        logger.error(f"Permission check failed: {e}")
        
        # Fail secure - deny access on error
        return False
```

#### 3. Least Privilege
```python
class UserRole(Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"

ROLE_PERMISSIONS = {
    UserRole.VIEWER: ["read"],
    UserRole.EDITOR: ["read", "write", "update"],
    UserRole.ADMIN: ["read", "write", "update", "delete", "admin"]
}

def check_permission(user_role: UserRole, required_permission: str) -> bool:
    return required_permission in ROLE_PERMISSIONS.get(user_role, [])
```

## Quick Reference

### Security Headers
```python
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
}
```

### Input Validation Patterns
```python
# Email validation
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# Username validation
USERNAME_REGEX = r'^[a-zA-Z0-9_-]{3,30}$'

# Password requirements
PASSWORD_REGEX = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

# UUID validation
UUID_REGEX = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
```

### Common Security Functions
```python
# Hash password
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
hashed = pwd_context.hash(password)

# Verify password
is_valid = pwd_context.verify(plain_password, hashed_password)

# Generate secure token
import secrets
token = secrets.token_urlsafe(32)

# Sanitize HTML
from bleach import clean
safe_html = clean(user_input, tags=['p', 'br', 'strong', 'em'], strip=True)

# Validate URL
from urllib.parse import urlparse
def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
```

## Assessment

### Knowledge Check

1. **What is the CIA triad?**
   - Answer: Confidentiality, Integrity, and Availability - the three fundamental principles of information security.

2. **Why should you never use string formatting for SQL queries?**
   - Answer: It can lead to SQL injection vulnerabilities. Always use parameterized queries.

3. **What is the purpose of rate limiting?**
   - Answer: To prevent abuse, protect against DoS attacks, and ensure fair resource usage.

4. **How should passwords be stored?**
   - Answer: As salted hashes using a secure algorithm like Argon2 or bcrypt, never in plain text.

5. **What is CSRF and how do you prevent it?**
   - Answer: Cross-Site Request Forgery - prevented using CSRF tokens and SameSite cookies.

### Practical Assessment

Complete the following security implementation tasks:

1. **Task 1**: Implement a secure file upload endpoint
   - Validate file types
   - Check file size
   - Scan for malicious content
   - Store securely

2. **Task 2**: Create a password reset flow
   - Generate secure reset tokens
   - Implement token expiration
   - Prevent enumeration attacks
   - Log security events

3. **Task 3**: Build an audit logging system
   - Log all security events
   - Include relevant context
   - Ensure log integrity
   - Implement log retention

### Certification

Upon completing this training and passing the assessment, you will receive:
- KnowledgeHub Security Certification
- Access to advanced security resources
- Eligibility for security team rotation

---

## Resources

### Internal Resources
- Security Wiki: `https://wiki.knowledgehub.com/security`
- Security Policies: `https://policies.knowledgehub.com`
- Incident Response: `https://incident.knowledgehub.com`

### External Resources
- OWASP Top 10: `https://owasp.org/Top10/`
- SANS Security Resources: `https://www.sans.org/free/`
- Python Security: `https://python.readthedocs.io/en/latest/library/security_warnings.html`

### Contact
- Security Team: security@knowledgehub.com
- Security Hotline: +1-XXX-XXX-XXXX (24/7)
- Bug Bounty: security-bounty@knowledgehub.com

---

**Training Version**: 1.0.0  
**Last Updated**: July 8, 2025  
**Next Update**: October 8, 2025  

**Remember**: Security is not just a feature, it's a mindset. Stay curious, stay cautious, and always validate!