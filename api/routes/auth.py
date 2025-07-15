"""Authentication and API key management endpoints"""

import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..models import get_db
from ..models.auth import APIKey
from ..config import settings
from ..security.sanitization import InputSanitizer

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating API keys"""
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable name for the API key")
    permissions: List[str] = Field(default=["read"], description="List of permissions for the API key")
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365, description="Expiration in days (optional)")


class APIKeyResponse(BaseModel):
    """Response model for API key operations"""
    id: str
    name: str
    permissions: List[str]
    created_at: str
    expires_at: Optional[str]
    last_used_at: Optional[str]
    is_active: bool


class CreateAPIKeyResponse(BaseModel):
    """Response model for API key creation"""
    api_key: str = Field(..., description="The actual API key - store this securely!")
    key_info: APIKeyResponse


def _hash_api_key(api_key: str) -> str:
    """Securely hash API key using HMAC-SHA256"""
    return hmac.new(
        settings.SECRET_KEY.encode(),
        api_key.encode(),
        hashlib.sha256
    ).hexdigest()


def _generate_api_key() -> str:
    """Generate a cryptographically secure API key"""
    # Generate 32 bytes of random data and encode as hex (64 characters)
    return f"kh_{secrets.token_hex(32)}"


def _check_admin_permissions(request: Request):
    """Check if current user has admin permissions"""
    if not (hasattr(request.state, 'api_key') and request.state.authenticated):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    current_user = request.state.api_key
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail="Admin permissions required"
        )
    return current_user


@router.post("/setup", response_model=CreateAPIKeyResponse)
async def setup_initial_api_key(
    req: CreateAPIKeyRequest,
    db: Session = Depends(get_db)
):
    """
    Create the first API key for system setup.
    Only works if no API keys exist in the database.
    """
    # Check if any API keys already exist
    existing_keys = db.query(APIKey).count()
    if existing_keys > 0:
        raise HTTPException(
            status_code=409,
            detail="API keys already exist. Use the authenticated endpoint to create additional keys."
        )
    
    # Sanitize input
    name = InputSanitizer.sanitize_text(req.name, max_length=255, allow_html=False)
    
    # Generate API key
    api_key = _generate_api_key()
    key_hash = _hash_api_key(api_key)
    
    # Calculate expiration
    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=req.expires_in_days)
    
    # Create API key record
    api_key_obj = APIKey(
        name=name,
        key_hash=key_hash,
        permissions=req.permissions,
        expires_at=expires_at
    )
    
    db.add(api_key_obj)
    db.commit()
    db.refresh(api_key_obj)
    
    return CreateAPIKeyResponse(
        api_key=api_key,
        key_info=APIKeyResponse(**api_key_obj.to_dict())
    )


@router.post("/keys", response_model=CreateAPIKeyResponse)
async def create_api_key(
    req: CreateAPIKeyRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Create a new API key (requires admin permissions)"""
    _check_admin_permissions(request)
    
    # Sanitize input
    name = InputSanitizer.sanitize_text(req.name, max_length=255, allow_html=False)
    
    # Generate API key
    api_key = _generate_api_key()
    key_hash = _hash_api_key(api_key)
    
    # Calculate expiration
    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=req.expires_in_days)
    
    # Create API key record
    api_key_obj = APIKey(
        name=name,
        key_hash=key_hash,
        permissions=req.permissions,
        expires_at=expires_at
    )
    
    db.add(api_key_obj)
    db.commit()
    db.refresh(api_key_obj)
    
    return CreateAPIKeyResponse(
        api_key=api_key,
        key_info=APIKeyResponse(**api_key_obj.to_dict())
    )


@router.get("/keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    request: Request,
    db: Session = Depends(get_db)
):
    """List all API keys (requires admin permissions)"""
    _check_admin_permissions(request)
    
    api_keys = db.query(APIKey).all()
    return [APIKeyResponse(**key.to_dict()) for key in api_keys]


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Revoke an API key (requires admin permissions)"""
    _check_admin_permissions(request)
    
    # Find and deactivate the API key
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    api_key.is_active = False
    db.commit()
    
    return {"message": f"API key '{api_key.name}' has been revoked"}


@router.get("/me", response_model=dict)
async def get_current_auth_info(request: Request):
    """Get current authentication information"""
    if hasattr(request.state, 'api_key') and request.state.authenticated:
        current_user = {
            "id": request.state.api_key["id"],
            "name": request.state.api_key["name"], 
            "permissions": request.state.api_key["permissions"],
            "type": "api_key"
        }
        auth_method = "api_key"
        authenticated = True
    else:
        current_user = {"id": "system", "name": "system", "permissions": ["read"], "type": "system"}
        auth_method = "none"
        authenticated = False
    
    return {
        "authenticated": authenticated,
        "user": current_user,
        "auth_method": auth_method
    }


@router.get("/status")
async def auth_status(db: Session = Depends(get_db)):
    """Get authentication system status"""
    total_keys = db.query(APIKey).count()
    active_keys = db.query(APIKey).filter(APIKey.is_active == True).count()
    
    return {
        "auth_enabled": True,
        "total_api_keys": total_keys,
        "active_api_keys": active_keys,
        "setup_required": total_keys == 0
    }