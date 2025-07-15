"""Project Timeline Service

Core service for managing project timelines, milestones, and progress tracking.
Provides comprehensive timeline management with automatic milestone detection.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from ..models.project_timeline import (
    ProjectTimeline, ProjectMilestone, ProgressSnapshot,
    ProjectStatus, MilestoneType, MilestoneStatus
)

logger = logging.getLogger(__name__)


class ProjectTimelineService:
    """Service for managing project timelines and milestones"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # Project Timeline Methods
    
    def create_project_timeline(
        self,
        project_id: str,
        project_name: str,
        owner_id: str,
        project_description: Optional[str] = None,
        priority: str = "medium",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        planned_end_date: Optional[datetime] = None,
        external_references: Optional[Dict[str, Any]] = None,
        custom_fields: Optional[Dict[str, Any]] = None
    ) -> ProjectTimeline:
        """Create a new project timeline"""
        
        try:
            timeline = ProjectTimeline(
                project_id=project_id,
                project_name=project_name,
                project_description=project_description,
                priority=priority,
                category=category,
                tags=tags or [],
                planned_end_date=planned_end_date,
                owner_id=owner_id,
                external_references=external_references or {},
                custom_fields=custom_fields or {}
            )
            
            self.db.add(timeline)
            self.db.commit()
            self.db.refresh(timeline)
            
            # Create initial progress snapshot
            self._create_progress_snapshot(timeline.id, auto_generated=True)
            
            logger.info(f"Created project timeline {timeline.id} for project {project_id}")
            return timeline
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Failed to create project timeline: {e}")
            raise ValueError(f"Project ID '{project_id}' already exists")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating project timeline: {e}")
            raise
    
    def get_project_timeline(self, project_id: str) -> Optional[ProjectTimeline]:
        """Get project timeline by project ID"""
        
        return self.db.query(ProjectTimeline).filter(
            ProjectTimeline.project_id == project_id
        ).first()
    
    def get_project_timeline_by_uuid(self, timeline_id: UUID) -> Optional[ProjectTimeline]:
        """Get project timeline by UUID"""
        
        return self.db.query(ProjectTimeline).filter(
            ProjectTimeline.id == timeline_id
        ).first()
    
    def update_project_timeline(
        self,
        project_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ProjectTimeline]:
        """Update project timeline"""
        
        timeline = self.get_project_timeline(project_id)
        if not timeline:
            return None
        
        try:
            # Update allowed fields
            allowed_fields = [
                'project_name', 'project_description', 'status', 'priority',
                'category', 'tags', 'planned_end_date', 'actual_end_date',
                'total_tasks', 'completed_tasks', 'team_members',
                'external_references', 'custom_fields'
            ]
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(timeline, field):
                    setattr(timeline, field, value)
            
            # Update progress if task counts changed
            if 'total_tasks' in updates or 'completed_tasks' in updates:
                timeline.update_progress()
            
            # Set started_at if moving to active status
            if updates.get('status') == ProjectStatus.ACTIVE.value and not timeline.started_at:
                timeline.started_at = datetime.now(timezone.utc)
            
            # Set actual_end_date if completing
            if updates.get('status') == ProjectStatus.COMPLETED.value and not timeline.actual_end_date:
                timeline.actual_end_date = datetime.now(timezone.utc)
            
            self.db.commit()
            
            # Create progress snapshot if significant change
            if any(field in updates for field in ['total_tasks', 'completed_tasks', 'status']):
                self._create_progress_snapshot(timeline.id, auto_generated=True)
            
            logger.info(f"Updated project timeline {timeline.id}")
            return timeline
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating project timeline: {e}")
            raise
    
    def delete_project_timeline(self, project_id: str) -> bool:
        """Delete project timeline"""
        
        timeline = self.get_project_timeline(project_id)
        if not timeline:
            return False
        
        try:
            self.db.delete(timeline)
            self.db.commit()
            
            logger.info(f"Deleted project timeline {timeline.id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting project timeline: {e}")
            raise
    
    def list_project_timelines(
        self,
        owner_id: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ProjectTimeline]:
        """List project timelines with filters"""
        
        query = self.db.query(ProjectTimeline)
        
        if owner_id:
            query = query.filter(ProjectTimeline.owner_id == owner_id)
        
        if status:
            query = query.filter(ProjectTimeline.status == status)
        
        if category:
            query = query.filter(ProjectTimeline.category == category)
        
        if priority:
            query = query.filter(ProjectTimeline.priority == priority)
        
        return query.order_by(desc(ProjectTimeline.updated_at)).offset(offset).limit(limit).all()
    
    def get_overdue_projects(self, owner_id: Optional[str] = None) -> List[ProjectTimeline]:
        """Get projects that are overdue"""
        
        query = self.db.query(ProjectTimeline).filter(
            and_(
                ProjectTimeline.planned_end_date < datetime.now(timezone.utc),
                ProjectTimeline.status != ProjectStatus.COMPLETED.value,
                ProjectTimeline.status != ProjectStatus.CANCELLED.value
            )
        )
        
        if owner_id:
            query = query.filter(ProjectTimeline.owner_id == owner_id)
        
        return query.order_by(ProjectTimeline.planned_end_date).all()
    
    # Milestone Methods
    
    def create_milestone(
        self,
        timeline_id: UUID,
        name: str,
        milestone_type: str = MilestoneType.CUSTOM.value,
        description: Optional[str] = None,
        planned_date: Optional[datetime] = None,
        assignee_id: Optional[str] = None,
        weight: float = 1.0,
        tags: Optional[List[str]] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
        auto_detected: bool = False,
        detection_confidence: float = 0.0,
        detection_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProjectMilestone]:
        """Create a new milestone"""
        
        # Verify timeline exists
        timeline = self.get_project_timeline_by_uuid(timeline_id)
        if not timeline:
            raise ValueError(f"Timeline {timeline_id} not found")
        
        try:
            milestone = ProjectMilestone(
                timeline_id=timeline_id,
                name=name,
                description=description,
                milestone_type=milestone_type,
                planned_date=planned_date,
                assignee_id=assignee_id,
                weight=weight,
                tags=tags or [],
                custom_fields=custom_fields or {},
                auto_detected=auto_detected,
                detection_confidence=detection_confidence,
                detection_metadata=detection_metadata or {}
            )
            
            self.db.add(milestone)
            self.db.commit()
            self.db.refresh(milestone)
            
            logger.info(f"Created milestone {milestone.id} for timeline {timeline_id}")
            return milestone
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating milestone: {e}")
            raise
    
    def get_milestone(self, milestone_id: UUID) -> Optional[ProjectMilestone]:
        """Get milestone by ID"""
        
        return self.db.query(ProjectMilestone).filter(
            ProjectMilestone.id == milestone_id
        ).first()
    
    def update_milestone(
        self,
        milestone_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[ProjectMilestone]:
        """Update milestone"""
        
        milestone = self.get_milestone(milestone_id)
        if not milestone:
            return None
        
        try:
            # Update allowed fields
            allowed_fields = [
                'name', 'description', 'milestone_type', 'status',
                'planned_date', 'actual_date', 'progress_percentage',
                'weight', 'assignee_id', 'tags', 'custom_fields',
                'dependencies', 'dependents'
            ]
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(milestone, field):
                    setattr(milestone, field, value)
            
            # Auto-complete if progress is 100%
            if updates.get('progress_percentage') == 100.0:
                milestone.status = MilestoneStatus.COMPLETED.value
                if not milestone.actual_date:
                    milestone.actual_date = datetime.now(timezone.utc)
            
            self.db.commit()
            
            logger.info(f"Updated milestone {milestone_id}")
            return milestone
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating milestone: {e}")
            raise
    
    def complete_milestone(
        self,
        milestone_id: UUID,
        completion_date: Optional[datetime] = None
    ) -> Optional[ProjectMilestone]:
        """Complete a milestone"""
        
        milestone = self.get_milestone(milestone_id)
        if not milestone:
            return None
        
        try:
            milestone.complete_milestone(completion_date)
            self.db.commit()
            
            # Update parent timeline progress
            self._update_timeline_progress_from_milestones(milestone.timeline_id)
            
            logger.info(f"Completed milestone {milestone_id}")
            return milestone
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error completing milestone: {e}")
            raise
    
    def delete_milestone(self, milestone_id: UUID) -> bool:
        """Delete milestone"""
        
        milestone = self.get_milestone(milestone_id)
        if not milestone:
            return False
        
        try:
            timeline_id = milestone.timeline_id
            self.db.delete(milestone)
            self.db.commit()
            
            # Update parent timeline progress
            self._update_timeline_progress_from_milestones(timeline_id)
            
            logger.info(f"Deleted milestone {milestone_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting milestone: {e}")
            raise
    
    def list_milestones(
        self,
        timeline_id: UUID,
        status: Optional[str] = None,
        milestone_type: Optional[str] = None,
        assignee_id: Optional[str] = None
    ) -> List[ProjectMilestone]:
        """List milestones for a timeline"""
        
        query = self.db.query(ProjectMilestone).filter(
            ProjectMilestone.timeline_id == timeline_id
        )
        
        if status:
            query = query.filter(ProjectMilestone.status == status)
        
        if milestone_type:
            query = query.filter(ProjectMilestone.milestone_type == milestone_type)
        
        if assignee_id:
            query = query.filter(ProjectMilestone.assignee_id == assignee_id)
        
        return query.order_by(ProjectMilestone.planned_date).all()
    
    def get_overdue_milestones(
        self,
        timeline_id: Optional[UUID] = None,
        assignee_id: Optional[str] = None
    ) -> List[ProjectMilestone]:
        """Get overdue milestones"""
        
        query = self.db.query(ProjectMilestone).filter(
            and_(
                ProjectMilestone.planned_date < datetime.now(timezone.utc),
                ProjectMilestone.status != MilestoneStatus.COMPLETED.value,
                ProjectMilestone.status != MilestoneStatus.CANCELLED.value
            )
        )
        
        if timeline_id:
            query = query.filter(ProjectMilestone.timeline_id == timeline_id)
        
        if assignee_id:
            query = query.filter(ProjectMilestone.assignee_id == assignee_id)
        
        return query.order_by(ProjectMilestone.planned_date).all()
    
    # Progress Tracking Methods
    
    def create_progress_snapshot(
        self,
        timeline_id: UUID,
        created_by: Optional[str] = None,
        snapshot_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProgressSnapshot]:
        """Create manual progress snapshot"""
        
        return self._create_progress_snapshot(
            timeline_id=timeline_id,
            created_by=created_by,
            snapshot_metadata=snapshot_metadata,
            auto_generated=False
        )
    
    def get_progress_history(
        self,
        timeline_id: UUID,
        limit: int = 30
    ) -> List[ProgressSnapshot]:
        """Get progress history for a timeline"""
        
        return self.db.query(ProgressSnapshot).filter(
            ProgressSnapshot.timeline_id == timeline_id
        ).order_by(desc(ProgressSnapshot.snapshot_date)).limit(limit).all()
    
    def calculate_project_velocity(
        self,
        timeline_id: UUID,
        days_back: int = 30
    ) -> Dict[str, float]:
        """Calculate project velocity metrics"""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        snapshots = self.db.query(ProgressSnapshot).filter(
            and_(
                ProgressSnapshot.timeline_id == timeline_id,
                ProgressSnapshot.snapshot_date >= cutoff_date
            )
        ).order_by(ProgressSnapshot.snapshot_date).all()
        
        if len(snapshots) < 2:
            return {'velocity': 0.0, 'trend': 'insufficient_data'}
        
        # Calculate average velocity
        velocities = [s.velocity for s in snapshots if s.velocity > 0]
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        
        # Calculate trend
        recent_velocity = sum(s.velocity for s in snapshots[-7:]) / min(7, len(snapshots))
        older_velocity = sum(s.velocity for s in snapshots[:-7]) / max(1, len(snapshots) - 7)
        
        if recent_velocity > older_velocity * 1.1:
            trend = 'improving'
        elif recent_velocity < older_velocity * 0.9:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'velocity': avg_velocity,
            'trend': trend,
            'recent_velocity': recent_velocity,
            'snapshots_analyzed': len(snapshots)
        }
    
    def get_project_insights(self, timeline_id: UUID) -> Dict[str, Any]:
        """Get comprehensive project insights"""
        
        timeline = self.get_project_timeline_by_uuid(timeline_id)
        if not timeline:
            return {}
        
        milestones = self.list_milestones(timeline_id)
        velocity = self.calculate_project_velocity(timeline_id)
        
        # Calculate milestone completion rate
        completed_milestones = [m for m in milestones if m.is_completed]
        milestone_completion_rate = len(completed_milestones) / len(milestones) if milestones else 0
        
        # Calculate estimated completion date
        if velocity['velocity'] > 0 and timeline.total_tasks > timeline.completed_tasks:
            remaining_tasks = timeline.total_tasks - timeline.completed_tasks
            days_to_completion = remaining_tasks / velocity['velocity']
            estimated_completion = datetime.now(timezone.utc) + timedelta(days=days_to_completion)
        else:
            estimated_completion = None
        
        return {
            'timeline_summary': timeline.get_summary(),
            'milestone_stats': {
                'total': len(milestones),
                'completed': len(completed_milestones),
                'overdue': len([m for m in milestones if m.is_overdue]),
                'completion_rate': milestone_completion_rate
            },
            'velocity_metrics': velocity,
            'estimated_completion': estimated_completion.isoformat() if estimated_completion else None,
            'health_score': self._calculate_project_health_score(timeline, milestones, velocity)
        }
    
    # Private Helper Methods
    
    def _create_progress_snapshot(
        self,
        timeline_id: UUID,
        created_by: Optional[str] = None,
        snapshot_metadata: Optional[Dict[str, Any]] = None,
        auto_generated: bool = True
    ) -> Optional[ProgressSnapshot]:
        """Create progress snapshot (internal method)"""
        
        timeline = self.get_project_timeline_by_uuid(timeline_id)
        if not timeline:
            return None
        
        try:
            # Get milestone counts
            milestones = self.list_milestones(timeline_id)
            completed_milestones = [m for m in milestones if m.is_completed]
            
            # Get previous snapshot for velocity calculation
            previous_snapshot = self.db.query(ProgressSnapshot).filter(
                ProgressSnapshot.timeline_id == timeline_id
            ).order_by(desc(ProgressSnapshot.snapshot_date)).first()
            
            snapshot = ProgressSnapshot(
                timeline_id=timeline_id,
                progress_percentage=timeline.progress_percentage,
                total_tasks=timeline.total_tasks,
                completed_tasks=timeline.completed_tasks,
                milestones_total=len(milestones),
                milestones_completed=len(completed_milestones),
                snapshot_metadata=snapshot_metadata or {},
                created_by=created_by,
                auto_generated=auto_generated
            )
            
            # Calculate velocity
            snapshot.calculate_velocity(previous_snapshot)
            
            self.db.add(snapshot)
            self.db.commit()
            self.db.refresh(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating progress snapshot: {e}")
            raise
    
    def _update_timeline_progress_from_milestones(self, timeline_id: UUID):
        """Update timeline progress based on milestone completion"""
        
        timeline = self.get_project_timeline_by_uuid(timeline_id)
        if not timeline:
            return
        
        milestones = self.list_milestones(timeline_id)
        if not milestones:
            return
        
        # Calculate weighted progress
        total_weight = sum(m.weight for m in milestones)
        completed_weight = sum(m.weight for m in milestones if m.is_completed)
        
        if total_weight > 0:
            milestone_progress = (completed_weight / total_weight) * 100
            
            # Update timeline progress (blend with task progress if available)
            if timeline.total_tasks > 0:
                task_progress = timeline.progress_percentage
                # 60% from tasks, 40% from milestones
                blended_progress = (task_progress * 0.6) + (milestone_progress * 0.4)
                timeline.progress_percentage = blended_progress
            else:
                timeline.progress_percentage = milestone_progress
        
        self.db.commit()
        
        # Create progress snapshot
        self._create_progress_snapshot(timeline_id, auto_generated=True)
    
    def _calculate_project_health_score(
        self,
        timeline: ProjectTimeline,
        milestones: List[ProjectMilestone],
        velocity: Dict[str, Any]
    ) -> float:
        """Calculate project health score (0-100)"""
        
        score = 100.0
        
        # Progress factor (0-30 points)
        progress_factor = timeline.progress_percentage * 0.3
        
        # Schedule factor (0-25 points)
        schedule_factor = 25.0
        if timeline.is_overdue:
            schedule_factor = 0.0
        elif timeline.planned_end_date:
            days_remaining = (timeline.planned_end_date - datetime.now(timezone.utc)).days
            if days_remaining < 0:
                schedule_factor = 0.0
            elif days_remaining < 7:
                schedule_factor = 10.0
            elif days_remaining < 30:
                schedule_factor = 20.0
        
        # Milestone factor (0-25 points)
        milestone_factor = 25.0
        if milestones:
            overdue_milestones = [m for m in milestones if m.is_overdue]
            if overdue_milestones:
                milestone_factor = max(0, 25.0 - (len(overdue_milestones) * 5))
        
        # Velocity factor (0-20 points)
        velocity_factor = 20.0
        if velocity['trend'] == 'declining':
            velocity_factor = 10.0
        elif velocity['trend'] == 'insufficient_data':
            velocity_factor = 15.0
        
        return max(0, min(100, progress_factor + schedule_factor + milestone_factor + velocity_factor))