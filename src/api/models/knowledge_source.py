"""Knowledge source model"""

from sqlalchemy import Column, String, DateTime, JSON, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum
from typing import Optional, Dict, Any, List

from .base import Base


class SourceStatus(str, enum.Enum):
    """Status of a knowledge source"""
    PENDING = "pending"
    CRAWLING = "crawling"
    INDEXING = "indexing"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class KnowledgeSource(Base):
    """Model for knowledge sources (documentation sites)"""
    
    __tablename__ = "knowledge_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    url = Column(Text, nullable=False, unique=True)
    status: SourceStatus = Column(
        SQLEnum(SourceStatus, name="source_status", values_callable=lambda obj: [e.value for e in obj]),
        default=SourceStatus.PENDING,
        nullable=False
    )
    config = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scraped_at = Column(DateTime(timezone=True), nullable=True)
    stats = Column(JSON, default={"documents": 0, "chunks": 0, "errors": 0})
    
    # Relationships
    documents = relationship("Document", back_populates="source", cascade="all, delete-orphan")
    jobs = relationship("ScrapingJob", back_populates="source", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<KnowledgeSource(id={self.id}, name='{self.name}', url='{self.url}', status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "name": self.name,
            "url": self.url,
            "status": self.status.value if isinstance(self.status, SourceStatus) else self.status,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_scraped_at": self.last_scraped_at.isoformat() if self.last_scraped_at else None,
            "stats": self.stats,
        }