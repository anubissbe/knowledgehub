"""Authentication service for admin operations"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, Request
from sqlalchemy.orm import Session
from ..models.auth import APIKey


def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from request"""
    if hasattr(request.state, 'api_key') and request.state.authenticated:
        return {
            "id": request.state.api_key["id"],
            "name": request.state.api_key["name"],
            "permissions": request.state.api_key["permissions"],
            "type": "api_key"
        }
    return None


def require_admin(request: Request) -> Dict[str, Any]:
    """Require admin permissions for endpoint access"""
    # In development mode, allow access without authentication
    from ..config import settings
    if settings.APP_ENV == "development":
        return {
            "id": "dev-admin",
            "name": "Development Admin",
            "permissions": ["admin", "read", "write"],
            "type": "development"
        }
    
    current_user = get_current_user(request)
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for admin access"
        )
    
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail="Admin permissions required"
        )
    
    return current_user


def check_permissions(request: Request, required_permissions: list) -> Dict[str, Any]:
    """Check if user has required permissions"""
    current_user = get_current_user(request)
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    user_permissions = current_user.get("permissions", [])
    
    if not any(perm in user_permissions for perm in required_permissions):
        raise HTTPException(
            status_code=403,
            detail=f"Required permissions: {required_permissions}"
        )
    
    return current_user


def is_authenticated(request: Request) -> bool:
    """Check if request is authenticated"""
    return hasattr(request.state, 'api_key') and request.state.authenticated


def has_permission(request: Request, permission: str) -> bool:
    """Check if user has specific permission"""
    current_user = get_current_user(request)
    if not current_user:
        return False
    
    return permission in current_user.get("permissions", [])


def is_admin(request: Request) -> bool:
    """Check if user has admin permissions"""
    return has_permission(request, "admin")