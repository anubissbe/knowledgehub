"""Job model for tracking background tasks"""

from sqlalchemy import Column, String, DateTime, JSON, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum
from typing import Optional, Dict, Any

from .base import Base


class JobStatus(str, enum.Enum):
    """Status of a job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    """Type of job"""
    SCRAPING = "scraping"
    REINDEXING = "reindexing"
    DELETION = "deletion"


class ScrapingJob(Base):
    """Model for tracking scraping and other background jobs"""
    
    __tablename__ = "scraping_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False)
    job_type = Column(
        SQLEnum(JobType, name="job_type", values_callable=lambda x: [e.value for e in x]),
        default=JobType.SCRAPING,
        nullable=False
    )
    status = Column(
        SQLEnum(JobStatus, name="job_status", values_callable=lambda x: [e.value for e in x]),
        default=JobStatus.PENDING,
        nullable=False
    )
    config = Column(JSON, default={})
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error = Column(Text)
    stats = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    source = relationship("KnowledgeSource", back_populates="jobs")
    
    def __repr__(self):
        return f"<ScrapingJob(id={self.id}, type={self.job_type}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "source_id": str(self.source_id),
            "job_type": self.job_type.value if isinstance(self.job_type, JobType) else self.job_type,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "config": self.config,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "stats": self.stats,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @property
    def duration_seconds(self):
        """Calculate job duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# Add Job alias
Job = ScrapingJob  # Alias for compatibility
