"""Task model for task tracking system"""

from sqlalchemy import Column, String, DateTime, JSON, Text, Boolean, Integer, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum
from typing import Optional, Dict, Any

from .base import Base


class TaskStatus(str, enum.Enum):
    """Status of a task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, enum.Enum):
    """Priority of a task"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Task(Base):
    """Model for tracking tasks"""
    
    __tablename__ = "tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    status: TaskStatus = Column(
        SQLEnum(TaskStatus, name="task_status", values_callable=lambda x: [e.value for e in x]),
        default=TaskStatus.PENDING,
        nullable=False
    )
    priority: TaskPriority = Column(
        SQLEnum(TaskPriority, name="task_priority", values_callable=lambda x: [e.value for e in x]),
        default=TaskPriority.MEDIUM,
        nullable=False
    )
    assignee = Column(String(255))
    tags = Column(JSON, default=[])
    task_metadata = Column(JSON, default={})
    progress = Column(Integer, default=0)  # Progress percentage 0-100
    estimated_hours = Column(Integer)
    actual_hours = Column(Integer)
    due_date = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255))
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Task(id={self.id}, title={self.title}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "priority": self.priority.value if isinstance(self.priority, TaskPriority) else self.priority,
            "assignee": self.assignee,
            "tags": self.tags or [],
            "metadata": self.task_metadata or {},
            "progress": self.progress,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "is_active": self.is_active,
        }
    
    @property
    def duration_hours(self):
        """Calculate task duration in hours"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() / 3600
        return None
    
    @property
    def is_overdue(self):
        """Check if task is overdue"""
        if not self.due_date:
            return False
        return self.due_date < datetime.utcnow() and self.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
    
    @property
    def is_in_progress(self):
        """Check if task is in progress"""
        return self.status == TaskStatus.IN_PROGRESS
    
    @property
    def is_completed(self):
        """Check if task is completed"""
        return self.status == TaskStatus.COMPLETED