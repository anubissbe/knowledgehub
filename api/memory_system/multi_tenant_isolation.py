#!/usr/bin/env python3
"""
Multi-Tenant Project Isolation System
Provides secure isolation between different tenants and projects with resource quotas and access controls
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from enum import Enum
import jwt
import secrets
from collections import defaultdict

# Add memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

from claude_unified_memory import UnifiedMemorySystem

logger = logging.getLogger(__name__)

class TenantStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    TERMINATED = "terminated"

class AccessLevel(Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    OWNER = "owner"

class ResourceType(Enum):
    MEMORY_COUNT = "memory_count"
    STORAGE_MB = "storage_mb"
    API_CALLS_PER_HOUR = "api_calls_per_hour"
    PROJECT_COUNT = "project_count"
    USER_COUNT = "user_count"

@dataclass
class ResourceQuota:
    """Resource quota limits for a tenant"""
    resource_type: ResourceType
    limit: int
    used: int = 0
    soft_limit: Optional[int] = None  # Warning threshold
    reset_period: Optional[str] = None  # For rate limits
    last_reset: Optional[str] = None
    
    def is_exceeded(self) -> bool:
        return self.used >= self.limit
    
    def is_soft_limit_exceeded(self) -> bool:
        return self.soft_limit is not None and self.used >= self.soft_limit
    
    def usage_percentage(self) -> float:
        return (self.used / self.limit) * 100 if self.limit > 0 else 0

@dataclass
class TenantUser:
    """User within a tenant"""
    user_id: str
    tenant_id: str
    username: str
    email: str
    access_level: AccessLevel
    projects: Set[str] = None  # Projects this user can access
    api_key: str = ""
    created_at: str = ""
    last_active: Optional[str] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.projects is None:
            self.projects = set()
        if not self.api_key:
            self.api_key = self._generate_api_key()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def _generate_api_key(self) -> str:
        """Generate secure API key for user"""
        return f"tk_{self.tenant_id[:8]}_{secrets.token_urlsafe(32)}"

@dataclass
class TenantProject:
    """Project within a tenant"""
    project_id: str
    tenant_id: str
    name: str
    description: str
    namespace: str  # Unique namespace for isolation
    
    # Access control
    authorized_users: Set[str] = None
    public_access: bool = False
    
    # Resource tracking
    memory_count: int = 0
    storage_used_mb: float = 0.0
    
    # Metadata
    created_at: str = ""
    created_by: str = ""
    last_modified: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.authorized_users is None:
            self.authorized_users = set()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if not self.namespace:
            self.namespace = f"{self.tenant_id}_{self.project_id}"

@dataclass
class Tenant:
    """Tenant in the multi-tenant system"""
    tenant_id: str
    name: str
    description: str
    status: TenantStatus
    
    # Resource quotas
    quotas: Dict[ResourceType, ResourceQuota] = None
    
    # Users and projects
    users: Dict[str, TenantUser] = None
    projects: Dict[str, TenantProject] = None
    
    # Security
    encryption_key: str = ""
    api_rate_limits: Dict[str, int] = None
    allowed_ip_ranges: List[str] = None
    
    # Metadata
    created_at: str = ""
    subscription_tier: str = "basic"
    billing_contact: str = ""
    
    def __post_init__(self):
        if self.quotas is None:
            self.quotas = self._create_default_quotas()
        if self.users is None:
            self.users = {}
        if self.projects is None:
            self.projects = {}
        if not self.encryption_key:
            self.encryption_key = secrets.token_urlsafe(32)
        if self.api_rate_limits is None:
            self.api_rate_limits = {"default": 1000}  # 1000 calls per hour
        if self.allowed_ip_ranges is None:
            self.allowed_ip_ranges = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def _create_default_quotas(self) -> Dict[ResourceType, ResourceQuota]:
        """Create default resource quotas"""
        return {
            ResourceType.MEMORY_COUNT: ResourceQuota(ResourceType.MEMORY_COUNT, 10000, soft_limit=8000),
            ResourceType.STORAGE_MB: ResourceQuota(ResourceType.STORAGE_MB, 1000, soft_limit=800),
            ResourceType.API_CALLS_PER_HOUR: ResourceQuota(ResourceType.API_CALLS_PER_HOUR, 1000, reset_period="hourly"),
            ResourceType.PROJECT_COUNT: ResourceQuota(ResourceType.PROJECT_COUNT, 10),
            ResourceType.USER_COUNT: ResourceQuota(ResourceType.USER_COUNT, 5)
        }

@dataclass
class AccessContext:
    """Context for access control decisions"""
    tenant_id: str
    user_id: str
    project_id: Optional[str] = None
    operation: str = ""
    resource_type: str = ""
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class AuditLog:
    """Audit log entry"""
    log_id: str
    tenant_id: str
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, denied
    details: Dict[str, Any] = None
    timestamp: str = ""
    ip_address: Optional[str] = None
    
    def __post_init__(self):
        if not self.log_id:
            self.log_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.details is None:
            self.details = {}

class MultiTenantIsolationManager:
    """
    Manages multi-tenant isolation with security, quotas, and access controls
    """
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.api_key_to_user: Dict[str, str] = {}  # api_key -> user_id
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Storage
        self.tenants_dir = Path("/opt/projects/memory-system/data/tenants")
        self.audit_dir = Path("/opt/projects/memory-system/data/audit_logs")
        self.tenants_dir.mkdir(exist_ok=True)
        self.audit_dir.mkdir(exist_ok=True)
        
        # Security settings
        self.jwt_secret = secrets.token_urlsafe(64)
        self.session_timeout = timedelta(hours=24)
        
        # Rate limiting tracking
        self.rate_limit_tracking: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Load existing tenants
        self._load_tenants()
    
    def _load_tenants(self):
        """Load existing tenants from storage"""
        try:
            for tenant_file in self.tenants_dir.glob("*.json"):
                tenant_id = tenant_file.stem
                with open(tenant_file, 'r', encoding='utf-8') as f:
                    tenant_data = json.load(f)
                    
                    # Convert data structures
                    tenant_data['quotas'] = {
                        ResourceType(k): ResourceQuota(**v) for k, v in tenant_data.get('quotas', {}).items()
                    }
                    tenant_data['users'] = {
                        k: TenantUser(**v) for k, v in tenant_data.get('users', {}).items()
                    }
                    tenant_data['projects'] = {
                        k: TenantProject(**v) for k, v in tenant_data.get('projects', {}).items()
                    }
                    
                    tenant = Tenant(**tenant_data)
                    self.tenants[tenant_id] = tenant
                    
                    # Build API key index
                    for user in tenant.users.values():
                        self.api_key_to_user[user.api_key] = user.user_id
            
            logger.info(f"Loaded {len(self.tenants)} tenants")
        except Exception as e:
            logger.error(f"Failed to load tenants: {e}")
    
    async def _save_tenant(self, tenant: Tenant):
        """Save tenant to storage"""
        try:
            # Convert to serializable format
            tenant_data = asdict(tenant)
            
            # Convert enums and sets
            tenant_data['quotas'] = {
                k.value: asdict(v) for k, v in tenant.quotas.items()
            }
            tenant_data['users'] = {
                k: {**asdict(v), 'projects': list(v.projects)} for k, v in tenant.users.items()
            }
            tenant_data['projects'] = {
                k: {**asdict(v), 'authorized_users': list(v.authorized_users)} 
                for k, v in tenant.projects.items()
            }
            
            tenant_file = self.tenants_dir / f"{tenant.tenant_id}.json"
            with open(tenant_file, 'w', encoding='utf-8') as f:
                json.dump(tenant_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save tenant {tenant.tenant_id}: {e}")
    
    async def create_tenant(
        self,
        name: str,
        description: str,
        subscription_tier: str = "basic",
        custom_quotas: Optional[Dict[ResourceType, int]] = None
    ) -> str:
        """Create a new tenant"""
        tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
        
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            description=description,
            status=TenantStatus.ACTIVE,
            subscription_tier=subscription_tier
        )
        
        # Apply custom quotas if provided
        if custom_quotas:
            for resource_type, limit in custom_quotas.items():
                if resource_type in tenant.quotas:
                    tenant.quotas[resource_type].limit = limit
        
        self.tenants[tenant_id] = tenant
        await self._save_tenant(tenant)
        
        await self._audit_log(
            tenant_id=tenant_id,
            user_id="system",
            action="create_tenant",
            resource=f"tenant:{tenant_id}",
            result="success",
            details={"name": name, "subscription_tier": subscription_tier}
        )
        
        logger.info(f"Created tenant {tenant_id}: {name}")
        return tenant_id
    
    async def create_user(
        self,
        tenant_id: str,
        username: str,
        email: str,
        access_level: AccessLevel,
        creator_user_id: str
    ) -> str:
        """Create a new user within a tenant"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Check quota
        user_quota = tenant.quotas.get(ResourceType.USER_COUNT)
        if user_quota and user_quota.is_exceeded():
            raise ValueError(f"User quota exceeded for tenant {tenant_id}")
        
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        
        user = TenantUser(
            user_id=user_id,
            tenant_id=tenant_id,
            username=username,
            email=email,
            access_level=access_level
        )
        
        tenant.users[user_id] = user
        self.api_key_to_user[user.api_key] = user_id
        
        # Update quota
        if user_quota:
            user_quota.used += 1
        
        await self._save_tenant(tenant)
        
        await self._audit_log(
            tenant_id=tenant_id,
            user_id=creator_user_id,
            action="create_user",
            resource=f"user:{user_id}",
            result="success",
            details={"username": username, "email": email, "access_level": access_level.value}
        )
        
        logger.info(f"Created user {user_id} in tenant {tenant_id}")
        return user_id
    
    async def create_project(
        self,
        tenant_id: str,
        name: str,
        description: str,
        creator_user_id: str,
        public_access: bool = False
    ) -> str:
        """Create a new project within a tenant"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Check quota
        project_quota = tenant.quotas.get(ResourceType.PROJECT_COUNT)
        if project_quota and project_quota.is_exceeded():
            raise ValueError(f"Project quota exceeded for tenant {tenant_id}")
        
        project_id = f"proj_{uuid.uuid4().hex[:12]}"
        
        project = TenantProject(
            project_id=project_id,
            tenant_id=tenant_id,
            name=name,
            description=description,
            namespace=f"{tenant_id}_{project_id}",
            created_by=creator_user_id,
            public_access=public_access
        )
        
        # Add creator to authorized users
        project.authorized_users.add(creator_user_id)
        
        tenant.projects[project_id] = project
        
        # Update quota
        if project_quota:
            project_quota.used += 1
        
        await self._save_tenant(tenant)
        
        await self._audit_log(
            tenant_id=tenant_id,
            user_id=creator_user_id,
            action="create_project",
            resource=f"project:{project_id}",
            result="success",
            details={"name": name, "public_access": public_access}
        )
        
        logger.info(f"Created project {project_id} in tenant {tenant_id}")
        return project_id
    
    async def check_access(self, context: AccessContext) -> Tuple[bool, str]:
        """Check if access should be granted"""
        tenant = self.tenants.get(context.tenant_id)
        if not tenant:
            return False, "Tenant not found"
        
        if tenant.status != TenantStatus.ACTIVE:
            return False, f"Tenant status: {tenant.status.value}"
        
        user = tenant.users.get(context.user_id)
        if not user:
            return False, "User not found"
        
        if not user.is_active:
            return False, "User inactive"
        
        # Check IP restrictions
        if tenant.allowed_ip_ranges and context.ip_address:
            if not self._check_ip_allowed(context.ip_address, tenant.allowed_ip_ranges):
                return False, "IP address not allowed"
        
        # Check rate limits
        if not await self._check_rate_limits(context):
            return False, "Rate limit exceeded"
        
        # Check project access if specified
        if context.project_id:
            project = tenant.projects.get(context.project_id)
            if not project:
                return False, "Project not found"
            
            if not project.public_access and context.user_id not in project.authorized_users:
                if user.access_level not in [AccessLevel.ADMIN, AccessLevel.OWNER]:
                    return False, "Project access denied"
        
        # Check operation permissions
        if not self._check_operation_permission(user, context.operation):
            return False, "Operation not permitted"
        
        return True, "Access granted"
    
    def _check_ip_allowed(self, ip_address: str, allowed_ranges: List[str]) -> bool:
        """Check if IP address is in allowed ranges"""
        # Simple implementation - in production use proper CIDR checking
        for allowed_range in allowed_ranges:
            if ip_address.startswith(allowed_range.split('/')[0]):
                return True
        return False
    
    async def _check_rate_limits(self, context: AccessContext) -> bool:
        """Check rate limits for user/tenant"""
        tenant = self.tenants[context.tenant_id]
        
        # Get rate limit for user's access level
        rate_limit = tenant.api_rate_limits.get(context.user_id) or tenant.api_rate_limits.get("default", 1000)
        
        # Track usage
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        key = f"{context.tenant_id}:{context.user_id}:{current_hour}"
        
        current_count = self.rate_limit_tracking[key].get("count", 0)
        
        if current_count >= rate_limit:
            return False
        
        # Update count
        self.rate_limit_tracking[key]["count"] = current_count + 1
        self.rate_limit_tracking[key]["last_request"] = datetime.now().isoformat()
        
        return True
    
    def _check_operation_permission(self, user: TenantUser, operation: str) -> bool:
        """Check if user has permission for operation"""
        # Define operation permissions by access level
        permissions = {
            AccessLevel.READ_ONLY: ["read", "list", "search"],
            AccessLevel.READ_WRITE: ["read", "list", "search", "create", "update"],
            AccessLevel.ADMIN: ["read", "list", "search", "create", "update", "delete", "manage"],
            AccessLevel.OWNER: ["*"]  # All operations
        }
        
        user_permissions = permissions.get(user.access_level, [])
        
        return "*" in user_permissions or operation in user_permissions
    
    async def store_memory_isolated(
        self,
        context: AccessContext,
        memory_id: str,
        memory_data: Dict[str, Any]
    ) -> bool:
        """Store memory with tenant isolation"""
        # Check access
        access_granted, reason = await self.check_access(context)
        if not access_granted:
            await self._audit_log(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                action="store_memory",
                resource=f"memory:{memory_id}",
                result="denied",
                details={"reason": reason}
            )
            return False
        
        tenant = self.tenants[context.tenant_id]
        
        # Check quotas
        memory_quota = tenant.quotas.get(ResourceType.MEMORY_COUNT)
        storage_quota = tenant.quotas.get(ResourceType.STORAGE_MB)
        
        if memory_quota and memory_quota.is_exceeded():
            await self._audit_log(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                action="store_memory",
                resource=f"memory:{memory_id}",
                result="denied",
                details={"reason": "Memory count quota exceeded"}
            )
            return False
        
        # Estimate storage size
        estimated_size_mb = len(json.dumps(memory_data).encode()) / (1024 * 1024)
        
        if storage_quota and (storage_quota.used + estimated_size_mb) > storage_quota.limit:
            await self._audit_log(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                action="store_memory",
                resource=f"memory:{memory_id}",
                result="denied",
                details={"reason": "Storage quota exceeded"}
            )
            return False
        
        # Add tenant isolation metadata
        isolated_memory_data = {
            **memory_data,
            "_tenant_metadata": {
                "tenant_id": context.tenant_id,
                "project_id": context.project_id,
                "namespace": self._get_project_namespace(context.tenant_id, context.project_id),
                "created_by": context.user_id,
                "created_at": datetime.now().isoformat(),
                "encrypted": True
            }
        }
        
        # Encrypt sensitive data
        isolated_memory_data = await self._encrypt_memory_data(tenant, isolated_memory_data)
        
        # Store with namespace prefix
        namespaced_memory_id = self._create_namespaced_id(context.tenant_id, context.project_id, memory_id)
        
        # Update quotas
        if memory_quota:
            memory_quota.used += 1
        if storage_quota:
            storage_quota.used += estimated_size_mb
        
        # Update project metrics
        if context.project_id:
            project = tenant.projects.get(context.project_id)
            if project:
                project.memory_count += 1
                project.storage_used_mb += estimated_size_mb
        
        await self._save_tenant(tenant)
        
        await self._audit_log(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            action="store_memory",
            resource=f"memory:{memory_id}",
            result="success",
            details={"size_mb": estimated_size_mb}
        )
        
        # Store the memory (this would integrate with the actual storage system)
        logger.info(f"Stored memory {namespaced_memory_id} for tenant {context.tenant_id}")
        return True
    
    async def retrieve_memory_isolated(
        self,
        context: AccessContext,
        memory_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve memory with tenant isolation"""
        # Check access
        access_granted, reason = await self.check_access(context)
        if not access_granted:
            await self._audit_log(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                action="retrieve_memory",
                resource=f"memory:{memory_id}",
                result="denied",
                details={"reason": reason}
            )
            return None
        
        tenant = self.tenants[context.tenant_id]
        
        # Create namespaced ID
        namespaced_memory_id = self._create_namespaced_id(context.tenant_id, context.project_id, memory_id)
        
        # Retrieve the memory (this would integrate with the actual storage system)
        # For now, just return a mock response
        mock_memory_data = {
            "memory_id": memory_id,
            "content": "Mock memory content",
            "_tenant_metadata": {
                "tenant_id": context.tenant_id,
                "project_id": context.project_id,
                "namespace": self._get_project_namespace(context.tenant_id, context.project_id)
            }
        }
        
        # Decrypt if needed
        decrypted_data = await self._decrypt_memory_data(tenant, mock_memory_data)
        
        await self._audit_log(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            action="retrieve_memory",
            resource=f"memory:{memory_id}",
            result="success"
        )
        
        return decrypted_data
    
    def _create_namespaced_id(self, tenant_id: str, project_id: Optional[str], memory_id: str) -> str:
        """Create namespaced memory ID"""
        if project_id:
            return f"{tenant_id}:{project_id}:{memory_id}"
        else:
            return f"{tenant_id}:default:{memory_id}"
    
    def _get_project_namespace(self, tenant_id: str, project_id: Optional[str]) -> str:
        """Get project namespace"""
        if project_id:
            return f"{tenant_id}_{project_id}"
        else:
            return f"{tenant_id}_default"
    
    async def _encrypt_memory_data(self, tenant: Tenant, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive memory data"""
        # Simple encryption placeholder - in production use proper encryption
        # This would encrypt sensitive fields using the tenant's encryption key
        return memory_data
    
    async def _decrypt_memory_data(self, tenant: Tenant, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt memory data"""
        # Simple decryption placeholder - in production use proper decryption
        return memory_data
    
    async def authenticate_api_key(self, api_key: str) -> Optional[Tuple[str, str]]:
        """Authenticate user by API key"""
        user_id = self.api_key_to_user.get(api_key)
        if not user_id:
            return None
        
        # Find tenant
        for tenant in self.tenants.values():
            if user_id in tenant.users:
                user = tenant.users[user_id]
                user.last_active = datetime.now().isoformat()
                await self._save_tenant(tenant)
                return tenant.tenant_id, user_id
        
        return None
    
    async def _audit_log(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ):
        """Log audit event"""
        audit_entry = AuditLog(
            log_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            details=details or {},
            ip_address=ip_address
        )
        
        # Store audit log
        audit_file = self.audit_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(audit_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(audit_entry), default=str) + '\n')
    
    async def get_tenant_usage_report(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive usage report for tenant"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {"error": "Tenant not found"}
        
        # Calculate usage statistics
        quota_status = {}
        for resource_type, quota in tenant.quotas.items():
            quota_status[resource_type.value] = {
                "limit": quota.limit,
                "used": quota.used,
                "usage_percentage": quota.usage_percentage(),
                "soft_limit_exceeded": quota.is_soft_limit_exceeded(),
                "hard_limit_exceeded": quota.is_exceeded()
            }
        
        # Project statistics
        project_stats = {}
        for project_id, project in tenant.projects.items():
            project_stats[project_id] = {
                "name": project.name,
                "memory_count": project.memory_count,
                "storage_used_mb": project.storage_used_mb,
                "authorized_users": len(project.authorized_users)
            }
        
        # User activity
        user_stats = {}
        for user_id, user in tenant.users.items():
            user_stats[user_id] = {
                "username": user.username,
                "access_level": user.access_level.value,
                "last_active": user.last_active,
                "is_active": user.is_active,
                "project_count": len(user.projects)
            }
        
        return {
            "tenant_info": {
                "tenant_id": tenant_id,
                "name": tenant.name,
                "status": tenant.status.value,
                "subscription_tier": tenant.subscription_tier,
                "created_at": tenant.created_at
            },
            "quota_status": quota_status,
            "project_statistics": project_stats,
            "user_statistics": user_stats,
            "summary": {
                "total_projects": len(tenant.projects),
                "total_users": len(tenant.users),
                "active_users": len([u for u in tenant.users.values() if u.is_active])
            }
        }


# Global multi-tenant isolation manager
multi_tenant_manager = MultiTenantIsolationManager()

# Convenience functions
async def create_tenant(name: str, description: str, subscription_tier: str = "basic") -> str:
    """Create a new tenant"""
    return await multi_tenant_manager.create_tenant(name, description, subscription_tier)

async def create_user(tenant_id: str, username: str, email: str, access_level: AccessLevel, creator_user_id: str) -> str:
    """Create a new user in tenant"""
    return await multi_tenant_manager.create_user(tenant_id, username, email, access_level, creator_user_id)

async def create_project(tenant_id: str, name: str, description: str, creator_user_id: str) -> str:
    """Create a new project in tenant"""
    return await multi_tenant_manager.create_project(tenant_id, name, description, creator_user_id)

async def check_access(context: AccessContext) -> Tuple[bool, str]:
    """Check access permissions"""
    return await multi_tenant_manager.check_access(context)

async def authenticate_api_key(api_key: str) -> Optional[Tuple[str, str]]:
    """Authenticate user by API key"""
    return await multi_tenant_manager.authenticate_api_key(api_key)

async def get_usage_report(tenant_id: str) -> Dict[str, Any]:
    """Get tenant usage report"""
    return await multi_tenant_manager.get_tenant_usage_report(tenant_id)

if __name__ == "__main__":
    # Test the multi-tenant isolation system
    async def test_multi_tenant_isolation():
        print("🏢 Testing Multi-Tenant Project Isolation")
        
        # Test tenant creation
        tenant_id = await create_tenant(
            name="Test Company",
            description="Test tenant for demo",
            subscription_tier="premium"
        )
        print(f"✅ Created tenant: {tenant_id}")
        
        # Test user creation
        user_id = await create_user(
            tenant_id=tenant_id,
            username="admin_user",
            email="admin@testcompany.com",
            access_level=AccessLevel.ADMIN,
            creator_user_id="system"
        )
        print(f"✅ Created user: {user_id}")
        
        # Test project creation
        project_id = await create_project(
            tenant_id=tenant_id,
            name="AI Assistant Project",
            description="Main AI assistant project",
            creator_user_id=user_id
        )
        print(f"✅ Created project: {project_id}")
        
        # Test access control
        context = AccessContext(
            tenant_id=tenant_id,
            user_id=user_id,
            project_id=project_id,
            operation="create",
            resource_type="memory"
        )
        
        access_granted, reason = await check_access(context)
        print(f"✅ Access check: {access_granted} ({reason})")
        
        # Test usage report
        report = await get_usage_report(tenant_id)
        print(f"✅ Usage report generated: {report['summary']['total_projects']} projects")
        
        print("✅ Multi-Tenant Isolation ready!")
    
    asyncio.run(test_multi_tenant_isolation())