"""
RBAC (Role-Based Access Control) Service
Implements multi-tenant security with fine-grained permissions
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum
import json

from sqlalchemy.orm import Session

from ..models.user import User
from ..models.base import Base
from ..database import get_session
from ..services.cache import CacheService
from ..services.real_ai_intelligence import RealAIIntelligence

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """System permissions"""
    # Document permissions
    DOC_READ = "doc:read"
    DOC_WRITE = "doc:write"
    DOC_DELETE = "doc:delete"
    DOC_ADMIN = "doc:admin"
    
    # Memory permissions
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"
    MEMORY_ADMIN = "memory:admin"
    
    # RAG permissions
    RAG_QUERY = "rag:query"
    RAG_INGEST = "rag:ingest"
    RAG_SCRAPE = "rag:scrape"
    RAG_ADMIN = "rag:admin"
    
    # User permissions
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_ADMIN = "user:admin"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"


class Role(str, Enum):
    """System roles"""
    VIEWER = "viewer"
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


# Role to permissions mapping
ROLE_PERMISSIONS = {
    Role.VIEWER: {
        Permission.DOC_READ,
        Permission.MEMORY_READ,
        Permission.RAG_QUERY,
        Permission.USER_READ,
    },
    Role.USER: {
        Permission.DOC_READ,
        Permission.DOC_WRITE,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.RAG_QUERY,
        Permission.USER_READ,
        Permission.USER_WRITE,
    },
    Role.DEVELOPER: {
        Permission.DOC_READ,
        Permission.DOC_WRITE,
        Permission.DOC_DELETE,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_DELETE,
        Permission.RAG_QUERY,
        Permission.RAG_INGEST,
        Permission.RAG_SCRAPE,
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.SYSTEM_MONITOR,
    },
    Role.ADMIN: {
        Permission.DOC_READ,
        Permission.DOC_WRITE,
        Permission.DOC_DELETE,
        Permission.DOC_ADMIN,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
        Permission.MEMORY_DELETE,
        Permission.MEMORY_ADMIN,
        Permission.RAG_QUERY,
        Permission.RAG_INGEST,
        Permission.RAG_SCRAPE,
        Permission.RAG_ADMIN,
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.USER_ADMIN,
        Permission.SYSTEM_MONITOR,
        Permission.SYSTEM_CONFIG,
    },
    Role.SUPER_ADMIN: {
        # Super admin has all permissions
        *list(Permission),
    }
}


class RBACService:
    """
    Service for managing role-based access control and multi-tenant security
    """
    
    def __init__(self):
        self.logger = logger
        self.cache = CacheService()
        self.ai_intelligence = RealAIIntelligence()
        
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """
        Get all permissions for a user based on their role
        
        Args:
            user: User object
            
        Returns:
            Set of permissions
        """
        # Check cache first
        cache_key = f"rbac:permissions:{user.id}"
        cached_perms = self.cache.get_sync(cache_key)
        if cached_perms:
            return set(Permission(p) for p in cached_perms)
            
        # Get base permissions from role
        role = Role(user.role) if user.role in [r.value for r in Role] else Role.USER
        permissions = ROLE_PERMISSIONS.get(role, set()).copy()
        
        # Add any custom permissions (from database in production)
        # custom_perms = self._get_custom_permissions(user)
        # permissions.update(custom_perms)
        
        # Cache permissions
        self.cache.set_sync(cache_key, list(permissions), ttl=300)  # 5 minutes
        
        return permissions
        
    def has_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if user has a specific permission
        
        Args:
            user: User object
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
        
    def has_any_permission(self, user: User, permissions: List[Permission]) -> bool:
        """
        Check if user has any of the specified permissions
        
        Args:
            user: User object
            permissions: List of permissions to check
            
        Returns:
            True if user has any permission
        """
        user_permissions = self.get_user_permissions(user)
        return any(p in user_permissions for p in permissions)
        
    def has_all_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """
        Check if user has all specified permissions
        
        Args:
            user: User object
            permissions: List of permissions to check
            
        Returns:
            True if user has all permissions
        """
        user_permissions = self.get_user_permissions(user)
        return all(p in user_permissions for p in permissions)
        
    def check_document_access(
        self,
        user: User,
        document_id: str,
        permission: Permission,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Check if user has access to a specific document
        
        Args:
            user: User object
            document_id: Document ID
            permission: Required permission
            tenant_id: Optional tenant filter
            
        Returns:
            True if user has access
        """
        # Check base permission
        if not self.has_permission(user, permission):
            return False
            
        # Check tenant isolation
        if tenant_id and hasattr(user, 'tenant_id'):
            if user.tenant_id != tenant_id:
                # Check if user is admin with cross-tenant access
                if not self.has_permission(user, Permission.SYSTEM_ADMIN):
                    return False
                    
        # Check document-specific permissions (from database in production)
        # doc_perms = self._get_document_permissions(document_id, user.id)
        # return permission in doc_perms
        
        return True
        
    def check_memory_access(
        self,
        user: User,
        session_id: str,
        permission: Permission
    ) -> bool:
        """
        Check if user has access to a memory session
        
        Args:
            user: User object
            session_id: Session ID
            permission: Required permission
            
        Returns:
            True if user has access
        """
        # Check base permission
        if not self.has_permission(user, permission):
            return False
            
        # Check session ownership (from database in production)
        # session_owner = self._get_session_owner(session_id)
        # if session_owner != user.id:
        #     return self.has_permission(user, Permission.MEMORY_ADMIN)
            
        return True
        
    def filter_by_tenant(
        self,
        user: User,
        query: Any,
        tenant_field: str = "tenant_id"
    ) -> Any:
        """
        Filter query results by user's tenant
        
        Args:
            user: User object
            query: SQLAlchemy query
            tenant_field: Field name for tenant ID
            
        Returns:
            Filtered query
        """
        # If user is super admin, no filtering
        if self.has_permission(user, Permission.SYSTEM_ADMIN):
            return query
            
        # Filter by user's tenant
        if hasattr(user, 'tenant_id') and user.tenant_id:
            return query.filter_by(**{tenant_field: user.tenant_id})
            
        return query
        
    def get_accessible_tenants(self, user: User) -> List[str]:
        """
        Get list of tenants accessible by user
        
        Args:
            user: User object
            
        Returns:
            List of tenant IDs
        """
        # Super admin can access all tenants
        if self.has_permission(user, Permission.SYSTEM_ADMIN):
            # Return all tenant IDs (from database in production)
            return ["*"]  # Special value for all tenants
            
        # Regular users can only access their own tenant
        if hasattr(user, 'tenant_id') and user.tenant_id:
            return [user.tenant_id]
            
        return []
        
    async def log_access(
        self,
        user: User,
        resource_type: str,
        resource_id: str,
        action: str,
        granted: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log access attempt for auditing
        
        Args:
            user: User object
            resource_type: Type of resource
            resource_id: Resource ID
            action: Action attempted
            granted: Whether access was granted
            metadata: Additional metadata
        """
        log_entry = {
            "user_id": str(user.id),
            "user_role": user.role,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "granted": granted,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Log to AI intelligence for pattern analysis
        await self.ai_intelligence.track_event(
            event_type="rbac_access",
            user_id=str(user.id),
            event_data=log_entry
        )
        
        # In production, also log to audit database
        logger.info(f"RBAC Access: {json.dumps(log_entry)}")
        
    def create_api_key(
        self,
        user: User,
        name: str,
        permissions: List[Permission],
        expires_at: Optional[datetime] = None
    ) -> str:
        """
        Create an API key with specific permissions
        
        Args:
            user: User creating the key
            name: Key name
            permissions: Permissions for the key
            expires_at: Optional expiration
            
        Returns:
            API key string
        """
        # Verify user can create keys
        if not self.has_permission(user, Permission.USER_ADMIN):
            raise PermissionError("User cannot create API keys")
            
        # Verify user can grant these permissions
        user_perms = self.get_user_permissions(user)
        for perm in permissions:
            if perm not in user_perms:
                raise PermissionError(f"User cannot grant permission: {perm}")
                
        # Generate API key (simplified - use proper crypto in production)
        import secrets
        api_key = f"kh_{secrets.token_urlsafe(32)}"
        
        # Store key metadata (in database in production)
        key_data = {
            "key": api_key,
            "name": name,
            "user_id": str(user.id),
            "permissions": [p.value for p in permissions],
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None
        }
        
        # Cache key data
        self.cache.set_sync(f"api_key:{api_key}", key_data, ttl=86400)  # 24 hours
        
        return api_key
        
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key and return its metadata
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key metadata if valid, None otherwise
        """
        # Check cache
        key_data = self.cache.get_sync(f"api_key:{api_key}")
        if not key_data:
            # Check database in production
            return None
            
        # Check expiration
        if key_data.get("expires_at"):
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if datetime.utcnow() > expires_at:
                return None
                
        return key_data


# Singleton instance
_rbac_service = None


def get_rbac_service() -> RBACService:
    """Get singleton RBAC service instance"""
    global _rbac_service
    if _rbac_service is None:
        _rbac_service = RBACService()
    return _rbac_service