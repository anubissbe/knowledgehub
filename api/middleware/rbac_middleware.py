"""
RBAC Middleware and Decorators
Provides permission checking for API endpoints
"""

from functools import wraps
from typing import List, Callable, Optional
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..services.rbac_service import RBACService, Permission, get_rbac_service
from ..services.auth import get_current_user
from ..models.user import User

# Security scheme
security = HTTPBearer()


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission for an endpoint
    
    Args:
        permission: Required permission
        
    Usage:
        @router.get("/admin")
        @require_permission(Permission.SYSTEM_ADMIN)
        async def admin_endpoint(user: User = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs
            user = kwargs.get('current_user')
            if not user:
                # Try to get from function arguments
                for arg in args:
                    if isinstance(arg, User):
                        user = arg
                        break
                        
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
                
            # Check permission
            rbac = get_rbac_service()
            if not rbac.has_permission(user, permission):
                # Log access denial
                await rbac.log_access(
                    user=user,
                    resource_type="endpoint",
                    resource_id=func.__name__,
                    action=permission.value,
                    granted=False
                )
                
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission.value} required"
                )
                
            # Log successful access
            await rbac.log_access(
                user=user,
                resource_type="endpoint",
                resource_id=func.__name__,
                action=permission.value,
                granted=True
            )
            
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


def require_any_permission(permissions: List[Permission]):
    """
    Decorator to require any of the specified permissions
    
    Args:
        permissions: List of permissions (any one is sufficient)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')
            if not user:
                for arg in args:
                    if isinstance(arg, User):
                        user = arg
                        break
                        
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
                
            rbac = get_rbac_service()
            if not rbac.has_any_permission(user, permissions):
                await rbac.log_access(
                    user=user,
                    resource_type="endpoint",
                    resource_id=func.__name__,
                    action=f"any_of:{[p.value for p in permissions]}",
                    granted=False
                )
                
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: one of {[p.value for p in permissions]} required"
                )
                
            await rbac.log_access(
                user=user,
                resource_type="endpoint",
                resource_id=func.__name__,
                action=f"any_of:{[p.value for p in permissions]}",
                granted=True
            )
            
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


def require_all_permissions(permissions: List[Permission]):
    """
    Decorator to require all specified permissions
    
    Args:
        permissions: List of permissions (all are required)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')
            if not user:
                for arg in args:
                    if isinstance(arg, User):
                        user = arg
                        break
                        
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
                
            rbac = get_rbac_service()
            if not rbac.has_all_permissions(user, permissions):
                await rbac.log_access(
                    user=user,
                    resource_type="endpoint",
                    resource_id=func.__name__,
                    action=f"all_of:{[p.value for p in permissions]}",
                    granted=False
                )
                
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: all of {[p.value for p in permissions]} required"
                )
                
            await rbac.log_access(
                user=user,
                resource_type="endpoint",
                resource_id=func.__name__,
                action=f"all_of:{[p.value for p in permissions]}",
                granted=True
            )
            
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


class TenantFilterMiddleware:
    """
    Middleware to automatically filter queries by tenant
    """
    
    def __init__(self, rbac_service: RBACService):
        self.rbac = rbac_service
        
    async def __call__(self, request: Request, call_next):
        # Get user from request if authenticated
        user = getattr(request.state, "user", None)
        
        if user:
            # Add tenant filter to request state
            request.state.tenant_filter = lambda query: self.rbac.filter_by_tenant(
                user, query
            )
        else:
            # No filtering for unauthenticated requests
            request.state.tenant_filter = lambda query: query
            
        response = await call_next(request)
        return response


def check_document_permission(
    permission: Permission,
    document_id_param: str = "document_id"
):
    """
    Dependency to check document-specific permissions
    
    Args:
        permission: Required permission
        document_id_param: Parameter name for document ID
    """
    async def dependency(
        request: Request,
        current_user: User = Depends(get_current_user)
    ):
        # Get document ID from path parameters
        document_id = request.path_params.get(document_id_param)
        if not document_id:
            raise HTTPException(
                status_code=400,
                detail=f"Missing {document_id_param}"
            )
            
        rbac = get_rbac_service()
        
        # Check access
        if not rbac.check_document_access(
            user=current_user,
            document_id=document_id,
            permission=permission
        ):
            await rbac.log_access(
                user=current_user,
                resource_type="document",
                resource_id=document_id,
                action=permission.value,
                granted=False
            )
            
            raise HTTPException(
                status_code=403,
                detail=f"No access to document: {permission.value} required"
            )
            
        await rbac.log_access(
            user=current_user,
            resource_type="document",
            resource_id=document_id,
            action=permission.value,
            granted=True
        )
        
        return current_user
        
    return dependency


def check_memory_permission(
    permission: Permission,
    session_id_param: str = "session_id"
):
    """
    Dependency to check memory session permissions
    
    Args:
        permission: Required permission
        session_id_param: Parameter name for session ID
    """
    async def dependency(
        request: Request,
        current_user: User = Depends(get_current_user)
    ):
        # Get session ID from path parameters
        session_id = request.path_params.get(session_id_param)
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail=f"Missing {session_id_param}"
            )
            
        rbac = get_rbac_service()
        
        # Check access
        if not rbac.check_memory_access(
            user=current_user,
            session_id=session_id,
            permission=permission
        ):
            await rbac.log_access(
                user=current_user,
                resource_type="memory_session",
                resource_id=session_id,
                action=permission.value,
                granted=False
            )
            
            raise HTTPException(
                status_code=403,
                detail=f"No access to session: {permission.value} required"
            )
            
        await rbac.log_access(
            user=current_user,
            resource_type="memory_session",
            resource_id=session_id,
            action=permission.value,
            granted=True
        )
        
        return current_user
        
    return dependency


# API Key authentication
async def get_api_key_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """
    Get user from API key
    
    Args:
        credentials: Bearer token credentials
        
    Returns:
        User if valid API key, None otherwise
    """
    token = credentials.credentials
    
    # Check if it's an API key (starts with kh_)
    if not token.startswith("kh_"):
        return None
        
    rbac = get_rbac_service()
    key_data = rbac.validate_api_key(token)
    
    if not key_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
        
    # Create a pseudo-user with API key permissions
    # In production, fetch actual user from database
    api_user = User(
        id=key_data["user_id"],
        email=f"api_key_{key_data['name']}",
        role="api_key",
        # Store permissions in user object for RBAC checks
        api_key_permissions=key_data["permissions"]
    )
    
    return api_user