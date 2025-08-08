"""
Advanced Security & Compliance Service for KnowledgeHub Enterprise
Implements OAuth2/OIDC, RBAC, audit logging, encryption, and GDPR compliance
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from fastapi import Header
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from sqlalchemy import text, Column, String, Boolean, DateTime, JSON, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

from ..database import AsyncSessionLocal, async_engine

logger = logging.getLogger(__name__)

class PermissionScope(str, Enum):
    """Permission scopes for RBAC"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    API_ACCESS = "api_access"
    GPU_ACCESS = "gpu_access"
    TENANT_ADMIN = "tenant_admin"

class ResourceType(str, Enum):
    """Resource types for permission checking"""
    DOCUMENT = "document"
    MEMORY = "memory"
    USER = "user"
    TENANT = "tenant"
    API_KEY = "api_key"
    ANALYTICS = "analytics"
    AI_MODEL = "ai_model"
    SEARCH = "search"

class AuditEventType(str, Enum):
    """Types of audit events"""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    PERMISSION_CHANGED = "permission_changed"
    CONFIG_CHANGED = "config_changed"
    SECURITY_ALERT = "security_alert"

Base = declarative_base()

class Role(Base):
    """Role-based access control roles"""
    __tablename__ = "roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=True)  # NULL for system roles
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Permissions as JSON array
    permissions = Column(JSON, default=list)
    
    # Role hierarchy
    parent_role_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Metadata
    is_system_role = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()"))

class Permission(Base):
    """Fine-grained permissions"""
    __tablename__ = "permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    resource_type = Column(String(50), nullable=False)
    scope = Column(String(50), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class AuditLog(Base):
    """Security audit logging"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True, index=True)
    resource_id = Column(String(255), nullable=True, index=True)
    
    # Context
    ip_address = Column(String(45), nullable=True, index=True)  # IPv6 compatible
    user_agent = Column(Text)
    session_id = Column(String(255), nullable=True)
    
    # Details
    event_data = Column(JSON, default=dict)
    risk_score = Column(Integer, default=0)  # 0-100 risk assessment
    
    # GDPR compliance
    anonymized = Column(Boolean, default=False)
    retention_date = Column(DateTime(timezone=True))
    
    timestamp = Column(DateTime(timezone=True), server_default=text("NOW()"), index=True)

class DataProcessingRecord(Base):
    """GDPR data processing records"""
    __tablename__ = "data_processing_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    
    # GDPR Article 30 requirements
    processing_purpose = Column(Text, nullable=False)
    data_categories = Column(JSON, default=list)  # Types of personal data
    data_subjects = Column(JSON, default=list)  # Categories of data subjects
    recipients = Column(JSON, default=list)  # Who receives the data
    retention_period = Column(String(255))
    security_measures = Column(JSON, default=list)
    
    # Consent tracking
    consent_given = Column(Boolean, default=False)
    consent_date = Column(DateTime(timezone=True))
    consent_withdrawn = Column(Boolean, default=False)
    withdrawal_date = Column(DateTime(timezone=True))
    
    # Legal basis
    legal_basis = Column(String(50))  # consent, contract, legal_obligation, etc.
    
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()"))

@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: uuid.UUID
    tenant_id: Optional[uuid.UUID]
    roles: List[str]
    permissions: Set[str]
    session_id: str
    ip_address: str
    user_agent: str
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if context has specific role"""
        return role in self.roles

class SecurityComplianceService:
    """Advanced security and compliance service"""
    
    def __init__(self, jwt_secret: str = None, encryption_key: str = None):
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        
        # System roles and permissions cache
        self._system_roles: Dict[str, Role] = {}
        self._permissions_cache: Dict[str, Set[str]] = {}
        
    async def initialize_system_roles(self):
        """Initialize default system roles and permissions"""
        
        system_permissions = [
            # Document permissions
            {"name": "document:read", "resource_type": "document", "scope": "read"},
            {"name": "document:write", "resource_type": "document", "scope": "write"},
            {"name": "document:delete", "resource_type": "document", "scope": "delete"},
            
            # Memory permissions
            {"name": "memory:read", "resource_type": "memory", "scope": "read"},
            {"name": "memory:write", "resource_type": "memory", "scope": "write"},
            {"name": "memory:delete", "resource_type": "memory", "scope": "delete"},
            
            # AI model permissions
            {"name": "ai_model:use", "resource_type": "ai_model", "scope": "read"},
            {"name": "ai_model:train", "resource_type": "ai_model", "scope": "write"},
            {"name": "gpu:access", "resource_type": "ai_model", "scope": "gpu_access"},
            
            # Admin permissions
            {"name": "user:admin", "resource_type": "user", "scope": "admin"},
            {"name": "tenant:admin", "resource_type": "tenant", "scope": "tenant_admin"},
            {"name": "api:admin", "resource_type": "api_key", "scope": "admin"},
            
            # Analytics permissions
            {"name": "analytics:read", "resource_type": "analytics", "scope": "read"},
            {"name": "analytics:export", "resource_type": "analytics", "scope": "write"}
        ]
        
        system_roles = [
            {
                "name": "system_admin",
                "description": "Full system administrator",
                "permissions": [p["name"] for p in system_permissions]
            },
            {
                "name": "tenant_admin", 
                "description": "Tenant administrator",
                "permissions": [
                    "document:read", "document:write", "document:delete",
                    "memory:read", "memory:write", "memory:delete",
                    "ai_model:use", "gpu:access",
                    "user:admin", "analytics:read"
                ]
            },
            {
                "name": "power_user",
                "description": "Advanced user with AI access",
                "permissions": [
                    "document:read", "document:write",
                    "memory:read", "memory:write",
                    "ai_model:use", "gpu:access",
                    "analytics:read"
                ]
            },
            {
                "name": "standard_user",
                "description": "Standard user",
                "permissions": [
                    "document:read", "document:write",
                    "memory:read", "memory:write",
                    "ai_model:use"
                ]
            },
            {
                "name": "read_only_user",
                "description": "Read-only access",
                "permissions": [
                    "document:read",
                    "memory:read"
                ]
            }
        ]
        
        async with AsyncSessionLocal() as db:
            # Create permissions
            for perm_data in system_permissions:
                # Check if permission exists
                result = await db.execute(
                    text("SELECT id FROM permissions WHERE name = :name"),
                    {"name": perm_data["name"]}
                )
                if not result.fetchone():
                    permission = Permission(**perm_data, description=f"Permission for {perm_data['name']}")
                    db.add(permission)
            
            # Create roles
            for role_data in system_roles:
                # Check if role exists
                result = await db.execute(
                    text("SELECT id FROM roles WHERE name = :name AND is_system_role = true"),
                    {"name": role_data["name"]}
                )
                if not result.fetchone():
                    role = Role(
                        name=role_data["name"],
                        description=role_data["description"],
                        permissions=role_data["permissions"],
                        is_system_role=True
                    )
                    db.add(role)
            
            await db.commit()
            logger.info("Initialized system roles and permissions")
    
    async def create_jwt_token(
        self,
        user_id: uuid.UUID,
        tenant_id: Optional[uuid.UUID] = None,
        roles: List[str] = None,
        expires_in: int = 3600
    ) -> str:
        """Create JWT token for authentication"""
        
        payload = {
            "user_id": str(user_id),
            "tenant_id": str(tenant_id) if tenant_id else None,
            "roles": roles or [],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(seconds=expires_in)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    async def verify_jwt_token(self, token: str) -> Optional[SecurityContext]:
        """Verify JWT token and return security context"""
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            user_id = uuid.UUID(payload["user_id"])
            tenant_id = uuid.UUID(payload["tenant_id"]) if payload.get("tenant_id") else None
            roles = payload.get("roles", [])
            
            # Get permissions for roles
            permissions = await self._get_permissions_for_roles(roles, tenant_id)
            
            return SecurityContext(
                user_id=user_id,
                tenant_id=tenant_id,
                roles=roles,
                permissions=permissions,
                session_id=secrets.token_urlsafe(16),
                ip_address="",  # Will be set by middleware
                user_agent=""   # Will be set by middleware
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying JWT token: {e}")
            return None
    
    async def _get_permissions_for_roles(
        self,
        roles: List[str],
        tenant_id: Optional[uuid.UUID] = None
    ) -> Set[str]:
        """Get all permissions for given roles"""
        
        cache_key = f"{':'.join(sorted(roles))}:{tenant_id}"
        if cache_key in self._permissions_cache:
            return self._permissions_cache[cache_key]
        
        permissions = set()
        
        async with AsyncSessionLocal() as db:
            for role_name in roles:
                # Get role permissions
                result = await db.execute(
                    text("""
                        SELECT permissions FROM roles 
                        WHERE name = :role_name 
                        AND (tenant_id = :tenant_id OR is_system_role = true)
                        AND is_active = true
                    """),
                    {"role_name": role_name, "tenant_id": tenant_id}
                )
                
                row = result.fetchone()
                if row and row.permissions:
                    permissions.update(row.permissions)
        
        self._permissions_cache[cache_key] = permissions
        return permissions
    
    async def check_permission(
        self,
        context: SecurityContext,
        permission: str,
        resource_id: Optional[str] = None
    ) -> bool:
        """Check if security context has permission for resource"""
        
        # System admin has all permissions
        if "system_admin" in context.roles:
            return True
        
        # Check direct permission
        if context.has_permission(permission):
            return True
        
        # Check resource-specific permissions if resource_id provided
        if resource_id and context.tenant_id:
            # Additional resource-level checks can be implemented here
            pass
        
        return False
    
    async def log_audit_event(
        self,
        context: Optional[SecurityContext],
        event_type: AuditEventType,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
        risk_score: int = 0
    ):
        """Log security audit event"""
        
        try:
            async with AsyncSessionLocal() as db:
                audit_log = AuditLog(
                    tenant_id=context.tenant_id if context else None,
                    user_id=context.user_id if context else None,
                    event_type=event_type.value,
                    resource_type=resource_type.value if resource_type else None,
                    resource_id=resource_id,
                    ip_address=context.ip_address if context else None,
                    user_agent=context.user_agent if context else None,
                    session_id=context.session_id if context else None,
                    event_data=event_data or {},
                    risk_score=risk_score,
                    retention_date=datetime.utcnow() + timedelta(days=2555)  # 7 years GDPR
                )
                
                db.add(audit_log)
                await db.commit()
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    async def create_database_tables(self):
        """Create security and compliance database tables"""
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

# Global security service instance
security_compliance_service = SecurityComplianceService()

# Authentication middleware dependency
async def get_security_context(authorization: Optional[str] = Header(None)) -> Optional[SecurityContext]:
    """Get security context from authorization header"""
    
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization[7:]  # Remove "Bearer " prefix
    return await security_compliance_service.verify_jwt_token(token)

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract security context from kwargs or request
            context = kwargs.get("security_context")
            if not context:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check permission
            if not await security_compliance_service.check_permission(context, permission):
                from fastapi import HTTPException
                await security_compliance_service.log_audit_event(
                    context, 
                    AuditEventType.ACCESS_DENIED,
                    event_data={"permission": permission, "function": func.__name__}
                )
                raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
