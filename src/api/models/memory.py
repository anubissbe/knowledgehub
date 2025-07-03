"""Memory item model for MCP memory storage"""

from sqlalchemy import Column, String, DateTime, JSON, Text, Integer, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from typing import Optional, Dict, Any, List

from .base import Base


class MemoryItem(Base):
    """Model for storing memory items from MCP clients"""
    
    __tablename__ = "memory_items"
    
    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content: str = Column(Text, nullable=False)
    content_hash: Optional[str] = Column(String(64), unique=True)
    tags: List[str] = Column(ARRAY(Text), default=[])
    meta_data: Dict[str, Any] = Column('metadata', JSON, default={})
    embedding_id: Optional[str] = Column(String(255))
    access_count: int = Column(Integer, default=0)
    created_at: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: datetime = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    accessed_at: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MemoryItem(id={self.id}, tags={self.tags}, access_count={self.access_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "content": self.content,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "metadata": self.meta_data,
            "embedding_id": self.embedding_id,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
        }
    
    def increment_access(self) -> None:
        """Increment access count and update accessed timestamp"""
        self.access_count += 1
        self.accessed_at = datetime.utcnow()