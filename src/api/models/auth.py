"""Authentication models"""

from sqlalchemy import Column, String, DateTime, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from .base import Base


class APIKey(Base):
    """Model for API key authentication"""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    permissions = Column(JSON, default=["read"])
    last_used_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', is_active={self.is_active})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "name": self.name,
            "permissions": self.permissions,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
        }
    
    def has_permission(self, permission: str) -> bool:
        """Check if API key has specific permission"""
        return permission in self.permissions or "admin" in self.permissions
    
    def is_valid(self) -> bool:
        """Check if API key is valid (active and not expired)"""
        if not self.is_active:
            return False
        if self.expires_at:
            # Handle timezone-aware comparison
            from datetime import timezone
            now = datetime.now(timezone.utc)
            expires_at = self.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if now > expires_at:
                return False
        return True
    
    def update_last_used(self):
        """Update last used timestamp"""
        from datetime import timezone
        self.last_used_at = datetime.now(timezone.utc)