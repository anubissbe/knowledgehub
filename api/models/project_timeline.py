"""Project Timeline Database Models

Models for tracking project timelines, milestones, and progress across different projects.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import uuid4
from enum import Enum

from sqlalchemy import Column, String, DateTime, Float, Integer, Text, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Base


class ProjectStatus(str, Enum):
    """Project status enumeration"""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MilestoneType(str, Enum):
    """Milestone type enumeration"""
    PLANNING = "planning"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    RELEASE = "release"
    REVIEW = "review"
    CUSTOM = "custom"


class MilestoneStatus(str, Enum):
    """Milestone status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ProjectTimeline(Base):
    """Main project timeline model"""
    __tablename__ = 'project_timelines'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(String(255), nullable=False, unique=True, index=True)
    project_name = Column(String(255), nullable=False)
    project_description = Column(Text)
    
    # Project metadata
    status = Column(String(50), nullable=False, default=ProjectStatus.PLANNING.value, index=True)
    priority = Column(String(50), nullable=False, default="medium")
    category = Column(String(100), index=True)
    tags = Column(JSON, default=list)
    
    # Timeline dates
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), index=True)
    planned_end_date = Column(DateTime(timezone=True), index=True)
    actual_end_date = Column(DateTime(timezone=True), index=True)
    
    # Progress metrics
    progress_percentage = Column(Float, default=0.0)
    total_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)
    
    # Metadata
    owner_id = Column(String(255), nullable=False, index=True)
    team_members = Column(JSON, default=list)
    external_references = Column(JSON, default=dict)  # Links to external systems like ProjectHub
    custom_fields = Column(JSON, default=dict)
    
    # Relationships
    milestones = relationship("ProjectMilestone", back_populates="timeline", cascade="all, delete-orphan")
    progress_snapshots = relationship("ProgressSnapshot", back_populates="timeline", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ProjectTimeline(id={self.id}, project_name='{self.project_name}', status='{self.status}')>"
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if project is currently active"""
        return self.status == ProjectStatus.ACTIVE.value
    
    @hybrid_property
    def is_completed(self) -> bool:
        """Check if project is completed"""
        return self.status == ProjectStatus.COMPLETED.value
    
    @hybrid_property
    def is_overdue(self) -> bool:
        """Check if project is overdue"""
        if not self.planned_end_date:
            return False
        return datetime.now(timezone.utc) > self.planned_end_date and not self.is_completed
    
    @hybrid_property
    def duration_days(self) -> Optional[int]:
        """Calculate project duration in days"""
        if not self.started_at:
            return None
        
        end_date = self.actual_end_date or datetime.now(timezone.utc)
        return (end_date - self.started_at).days
    
    def update_progress(self):
        """Update progress percentage based on completed tasks"""
        if self.total_tasks > 0:
            self.progress_percentage = (self.completed_tasks / self.total_tasks) * 100
        else:
            self.progress_percentage = 0.0
        
        self.updated_at = datetime.now(timezone.utc)
    
    def add_team_member(self, member_id: str, role: str = "member"):
        """Add team member to project"""
        if self.team_members is None:
            self.team_members = []
        
        # Check if member already exists
        for member in self.team_members:
            if member.get('id') == member_id:
                member['role'] = role
                return
        
        self.team_members.append({
            'id': member_id,
            'role': role,
            'added_at': datetime.now(timezone.utc).isoformat()
        })
    
    def remove_team_member(self, member_id: str):
        """Remove team member from project"""
        if self.team_members is None:
            return
        
        self.team_members = [m for m in self.team_members if m.get('id') != member_id]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get project summary"""
        return {
            'id': str(self.id),
            'project_id': self.project_id,
            'project_name': self.project_name,
            'status': self.status,
            'priority': self.priority,
            'progress_percentage': self.progress_percentage,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'planned_end_date': self.planned_end_date.isoformat() if self.planned_end_date else None,
            'actual_end_date': self.actual_end_date.isoformat() if self.actual_end_date else None,
            'is_overdue': self.is_overdue,
            'duration_days': self.duration_days,
            'team_members_count': len(self.team_members) if self.team_members else 0,
            'milestones_count': len(self.milestones) if self.milestones else 0
        }


class ProjectMilestone(Base):
    """Project milestone model"""
    __tablename__ = 'project_milestones'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    timeline_id = Column(PGUUID(as_uuid=True), ForeignKey('project_timelines.id'), nullable=False)
    
    # Milestone details
    name = Column(String(255), nullable=False)
    description = Column(Text)
    milestone_type = Column(String(50), nullable=False, default=MilestoneType.CUSTOM.value)
    status = Column(String(50), nullable=False, default=MilestoneStatus.PENDING.value, index=True)
    
    # Dates
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    planned_date = Column(DateTime(timezone=True), index=True)
    actual_date = Column(DateTime(timezone=True), index=True)
    
    # Progress and metrics
    progress_percentage = Column(Float, default=0.0)
    weight = Column(Float, default=1.0)  # Milestone importance weight
    
    # Dependencies
    dependencies = Column(JSON, default=list)  # List of milestone IDs this depends on
    dependents = Column(JSON, default=list)    # List of milestone IDs that depend on this
    
    # Metadata
    assignee_id = Column(String(255), index=True)
    tags = Column(JSON, default=list)
    custom_fields = Column(JSON, default=dict)
    
    # Auto-detection fields
    auto_detected = Column(Boolean, default=False)
    detection_confidence = Column(Float, default=0.0)
    detection_metadata = Column(JSON, default=dict)
    
    # Relationships
    timeline = relationship("ProjectTimeline", back_populates="milestones")
    
    def __repr__(self):
        return f"<ProjectMilestone(id={self.id}, name='{self.name}', status='{self.status}')>"
    
    @hybrid_property
    def is_completed(self) -> bool:
        """Check if milestone is completed"""
        return self.status == MilestoneStatus.COMPLETED.value
    
    @hybrid_property
    def is_overdue(self) -> bool:
        """Check if milestone is overdue"""
        if not self.planned_date or self.is_completed:
            return False
        return datetime.now(timezone.utc) > self.planned_date
    
    @hybrid_property
    def days_until_due(self) -> Optional[int]:
        """Calculate days until milestone is due"""
        if not self.planned_date:
            return None
        
        delta = self.planned_date - datetime.now(timezone.utc)
        return delta.days
    
    def complete_milestone(self, completion_date: Optional[datetime] = None):
        """Mark milestone as completed"""
        self.status = MilestoneStatus.COMPLETED.value
        self.actual_date = completion_date or datetime.now(timezone.utc)
        self.progress_percentage = 100.0
        self.updated_at = datetime.now(timezone.utc)
    
    def add_dependency(self, milestone_id: str):
        """Add milestone dependency"""
        if self.dependencies is None:
            self.dependencies = []
        
        if milestone_id not in self.dependencies:
            self.dependencies.append(milestone_id)
    
    def remove_dependency(self, milestone_id: str):
        """Remove milestone dependency"""
        if self.dependencies is None:
            return
        
        self.dependencies = [dep for dep in self.dependencies if dep != milestone_id]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get milestone summary"""
        return {
            'id': str(self.id),
            'timeline_id': str(self.timeline_id),
            'name': self.name,
            'description': self.description,
            'milestone_type': self.milestone_type,
            'status': self.status,
            'progress_percentage': self.progress_percentage,
            'weight': self.weight,
            'created_at': self.created_at.isoformat(),
            'planned_date': self.planned_date.isoformat() if self.planned_date else None,
            'actual_date': self.actual_date.isoformat() if self.actual_date else None,
            'is_overdue': self.is_overdue,
            'days_until_due': self.days_until_due,
            'assignee_id': self.assignee_id,
            'auto_detected': self.auto_detected,
            'detection_confidence': self.detection_confidence,
            'dependencies_count': len(self.dependencies) if self.dependencies else 0,
            'dependents_count': len(self.dependents) if self.dependents else 0
        }


class ProgressSnapshot(Base):
    """Progress snapshot for tracking progress over time"""
    __tablename__ = 'progress_snapshots'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    timeline_id = Column(PGUUID(as_uuid=True), ForeignKey('project_timelines.id'), nullable=False)
    
    # Snapshot data
    snapshot_date = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    progress_percentage = Column(Float, nullable=False)
    total_tasks = Column(Integer, nullable=False)
    completed_tasks = Column(Integer, nullable=False)
    
    # Milestone progress
    milestones_total = Column(Integer, default=0)
    milestones_completed = Column(Integer, default=0)
    
    # Velocity metrics
    tasks_completed_since_last = Column(Integer, default=0)
    days_since_last_snapshot = Column(Integer, default=0)
    velocity = Column(Float, default=0.0)  # tasks per day
    
    # Metadata
    snapshot_metadata = Column(JSON, default=dict)
    created_by = Column(String(255))
    auto_generated = Column(Boolean, default=False)
    
    # Relationships
    timeline = relationship("ProjectTimeline", back_populates="progress_snapshots")
    
    def __repr__(self):
        return f"<ProgressSnapshot(id={self.id}, timeline_id={self.timeline_id}, progress={self.progress_percentage}%)>"
    
    @hybrid_property
    def completion_rate(self) -> float:
        """Calculate completion rate (completed/total tasks)"""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks
    
    def calculate_velocity(self, previous_snapshot: Optional['ProgressSnapshot'] = None):
        """Calculate velocity based on previous snapshot"""
        if not previous_snapshot:
            self.velocity = 0.0
            return
        
        # Calculate days between snapshots
        days_diff = (self.snapshot_date - previous_snapshot.snapshot_date).days
        if days_diff == 0:
            days_diff = 1  # Avoid division by zero
        
        # Calculate tasks completed since last snapshot
        tasks_diff = self.completed_tasks - previous_snapshot.completed_tasks
        
        # Calculate velocity (tasks per day)
        self.velocity = tasks_diff / days_diff
        self.tasks_completed_since_last = tasks_diff
        self.days_since_last_snapshot = days_diff
    
    def get_summary(self) -> Dict[str, Any]:
        """Get snapshot summary"""
        return {
            'id': str(self.id),
            'timeline_id': str(self.timeline_id),
            'snapshot_date': self.snapshot_date.isoformat(),
            'progress_percentage': self.progress_percentage,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'milestones_total': self.milestones_total,
            'milestones_completed': self.milestones_completed,
            'velocity': self.velocity,
            'completion_rate': self.completion_rate,
            'tasks_completed_since_last': self.tasks_completed_since_last,
            'days_since_last_snapshot': self.days_since_last_snapshot,
            'auto_generated': self.auto_generated
        }