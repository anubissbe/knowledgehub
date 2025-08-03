"""User model for authentication and user management"""

from sqlalchemy import Column, String, Boolean, DateTime, JSON, Enum, Integer
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime, timezone
import enum

from .base import Base


class UserRole(str, enum.Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    API_KEY = "api_key"
    SERVICE = "service"


class UserStatus(str, enum.Enum):
    """User status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(Base):
    """User model for authentication and access control"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    status = Column(Enum(UserStatus), default=UserStatus.PENDING, nullable=False)
    
    # Authentication
    hashed_password = Column(String(255))  # For regular users
    api_key_hash = Column(String(255))  # For API key authentication
    
    # Metadata
    permissions = Column(JSON, default=list)  # List of specific permissions
    user_metadata = Column("metadata", JSON, default=dict)  # Additional user metadata
    
    # Activity tracking
    last_login = Column(DateTime(timezone=True))
    last_activity = Column(DateTime(timezone=True))
    login_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        """Convert user to dictionary"""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value if self.role else None,
            "status": self.status.value if self.status else None,
            "permissions": self.permissions or [],
            "metadata": self.user_metadata or {},
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "login_count": self.login_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)
    
    def update_login(self):
        """Update login information"""
        self.last_login = datetime.now(timezone.utc)
        self.last_activity = self.last_login
        self.login_count += 1
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        if self.role == UserRole.ADMIN:
            return True  # Admins have all permissions
        return permission in (self.permissions or [])
    
    def is_active(self) -> bool:
        """Check if user is active"""
        return self.status == UserStatus.ACTIVE