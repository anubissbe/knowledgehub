"""Project Timeline API Router

API endpoints for project timeline management, milestone tracking, and progress analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..models import get_db
from ..services.project_timeline_service import ProjectTimelineService
from ..services.milestone_detector import MilestoneDetector
from ..services.progress_analyzer import ProgressAnalyzer
from ..models.project_timeline import ProjectStatus, MilestoneType, MilestoneStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/project-timeline", tags=["project-timeline"])


# Pydantic models for API

class CreateTimelineRequest(BaseModel):
    project_id: str = Field(..., description="Unique project identifier")
    project_name: str = Field(..., description="Project name")
    owner_id: str = Field(..., description="Project owner identifier")
    project_description: Optional[str] = Field(None, description="Project description")
    priority: str = Field("medium", description="Project priority")
    category: Optional[str] = Field(None, description="Project category")
    tags: Optional[List[str]] = Field(None, description="Project tags")
    planned_end_date: Optional[datetime] = Field(None, description="Planned end date")
    external_references: Optional[Dict[str, Any]] = Field(None, description="External system references")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom fields")


class UpdateTimelineRequest(BaseModel):
    project_name: Optional[str] = Field(None, description="Project name")
    project_description: Optional[str] = Field(None, description="Project description")
    status: Optional[str] = Field(None, description="Project status")
    priority: Optional[str] = Field(None, description="Project priority")
    category: Optional[str] = Field(None, description="Project category")
    tags: Optional[List[str]] = Field(None, description="Project tags")
    planned_end_date: Optional[datetime] = Field(None, description="Planned end date")
    actual_end_date: Optional[datetime] = Field(None, description="Actual end date")
    total_tasks: Optional[int] = Field(None, description="Total number of tasks")
    completed_tasks: Optional[int] = Field(None, description="Number of completed tasks")
    team_members: Optional[List[Dict[str, Any]]] = Field(None, description="Team members")
    external_references: Optional[Dict[str, Any]] = Field(None, description="External system references")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom fields")


class CreateMilestoneRequest(BaseModel):
    name: str = Field(..., description="Milestone name")
    milestone_type: str = Field(MilestoneType.CUSTOM.value, description="Milestone type")
    description: Optional[str] = Field(None, description="Milestone description")
    planned_date: Optional[datetime] = Field(None, description="Planned completion date")
    assignee_id: Optional[str] = Field(None, description="Assignee identifier")
    weight: float = Field(1.0, description="Milestone weight/importance")
    tags: Optional[List[str]] = Field(None, description="Milestone tags")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom fields")


class UpdateMilestoneRequest(BaseModel):
    name: Optional[str] = Field(None, description="Milestone name")
    description: Optional[str] = Field(None, description="Milestone description")
    milestone_type: Optional[str] = Field(None, description="Milestone type")
    status: Optional[str] = Field(None, description="Milestone status")
    planned_date: Optional[datetime] = Field(None, description="Planned completion date")
    actual_date: Optional[datetime] = Field(None, description="Actual completion date")
    progress_percentage: Optional[float] = Field(None, description="Progress percentage")
    weight: Optional[float] = Field(None, description="Milestone weight/importance")
    assignee_id: Optional[str] = Field(None, description="Assignee identifier")
    tags: Optional[List[str]] = Field(None, description="Milestone tags")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom fields")


class TimelineResponse(BaseModel):
    id: str
    project_id: str
    project_name: str
    project_description: Optional[str]
    status: str
    priority: str
    category: Optional[str]
    tags: List[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    planned_end_date: Optional[str]
    actual_end_date: Optional[str]
    progress_percentage: float
    total_tasks: int
    completed_tasks: int
    owner_id: str
    team_members: List[Dict[str, Any]]
    external_references: Dict[str, Any]
    custom_fields: Dict[str, Any]
    is_active: bool
    is_completed: bool
    is_overdue: bool
    duration_days: Optional[int]


class MilestoneResponse(BaseModel):
    id: str
    timeline_id: str
    name: str
    description: Optional[str]
    milestone_type: str
    status: str
    created_at: str
    updated_at: str
    planned_date: Optional[str]
    actual_date: Optional[str]
    progress_percentage: float
    weight: float
    assignee_id: Optional[str]
    tags: List[str]
    custom_fields: Dict[str, Any]
    auto_detected: bool
    detection_confidence: float
    is_completed: bool
    is_overdue: bool
    days_until_due: Optional[int]


# Timeline Management Endpoints

@router.post("/timelines", response_model=TimelineResponse)
def create_timeline(
    request: CreateTimelineRequest,
    db: Session = Depends(get_db)
):
    """Create a new project timeline"""
    
    try:
        service = ProjectTimelineService(db)
        
        timeline = service.create_project_timeline(
            project_id=request.project_id,
            project_name=request.project_name,
            owner_id=request.owner_id,
            project_description=request.project_description,
            priority=request.priority,
            category=request.category,
            tags=request.tags,
            planned_end_date=request.planned_end_date,
            external_references=request.external_references,
            custom_fields=request.custom_fields
        )
        
        return TimelineResponse(
            id=str(timeline.id),
            project_id=timeline.project_id,
            project_name=timeline.project_name,
            project_description=timeline.project_description,
            status=timeline.status,
            priority=timeline.priority,
            category=timeline.category,
            tags=timeline.tags or [],
            created_at=timeline.created_at.isoformat(),
            updated_at=timeline.updated_at.isoformat(),
            started_at=timeline.started_at.isoformat() if timeline.started_at else None,
            planned_end_date=timeline.planned_end_date.isoformat() if timeline.planned_end_date else None,
            actual_end_date=timeline.actual_end_date.isoformat() if timeline.actual_end_date else None,
            progress_percentage=timeline.progress_percentage,
            total_tasks=timeline.total_tasks,
            completed_tasks=timeline.completed_tasks,
            owner_id=timeline.owner_id,
            team_members=timeline.team_members or [],
            external_references=timeline.external_references or {},
            custom_fields=timeline.custom_fields or {},
            is_active=timeline.is_active,
            is_completed=timeline.is_completed,
            is_overdue=timeline.is_overdue,
            duration_days=timeline.duration_days
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating timeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/timelines/{project_id}", response_model=TimelineResponse)
def get_timeline(
    project_id: str,
    db: Session = Depends(get_db)
):
    """Get project timeline by project ID"""
    
    try:
        service = ProjectTimelineService(db)
        timeline = service.get_project_timeline(project_id)
        
        if not timeline:
            raise HTTPException(status_code=404, detail="Timeline not found")
        
        return TimelineResponse(
            id=str(timeline.id),
            project_id=timeline.project_id,
            project_name=timeline.project_name,
            project_description=timeline.project_description,
            status=timeline.status,
            priority=timeline.priority,
            category=timeline.category,
            tags=timeline.tags or [],
            created_at=timeline.created_at.isoformat(),
            updated_at=timeline.updated_at.isoformat(),
            started_at=timeline.started_at.isoformat() if timeline.started_at else None,
            planned_end_date=timeline.planned_end_date.isoformat() if timeline.planned_end_date else None,
            actual_end_date=timeline.actual_end_date.isoformat() if timeline.actual_end_date else None,
            progress_percentage=timeline.progress_percentage,
            total_tasks=timeline.total_tasks,
            completed_tasks=timeline.completed_tasks,
            owner_id=timeline.owner_id,
            team_members=timeline.team_members or [],
            external_references=timeline.external_references or {},
            custom_fields=timeline.custom_fields or {},
            is_active=timeline.is_active,
            is_completed=timeline.is_completed,
            is_overdue=timeline.is_overdue,
            duration_days=timeline.duration_days
        )
        
    except Exception as e:
        logger.error(f"Error getting timeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/timelines/{project_id}", response_model=TimelineResponse)
def update_timeline(
    project_id: str,
    request: UpdateTimelineRequest,
    db: Session = Depends(get_db)
):
    """Update project timeline"""
    
    try:
        service = ProjectTimelineService(db)
        
        # Convert request to dict, excluding None values
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        timeline = service.update_project_timeline(project_id, updates)
        
        if not timeline:
            raise HTTPException(status_code=404, detail="Timeline not found")
        
        return TimelineResponse(
            id=str(timeline.id),
            project_id=timeline.project_id,
            project_name=timeline.project_name,
            project_description=timeline.project_description,
            status=timeline.status,
            priority=timeline.priority,
            category=timeline.category,
            tags=timeline.tags or [],
            created_at=timeline.created_at.isoformat(),
            updated_at=timeline.updated_at.isoformat(),
            started_at=timeline.started_at.isoformat() if timeline.started_at else None,
            planned_end_date=timeline.planned_end_date.isoformat() if timeline.planned_end_date else None,
            actual_end_date=timeline.actual_end_date.isoformat() if timeline.actual_end_date else None,
            progress_percentage=timeline.progress_percentage,
            total_tasks=timeline.total_tasks,
            completed_tasks=timeline.completed_tasks,
            owner_id=timeline.owner_id,
            team_members=timeline.team_members or [],
            external_references=timeline.external_references or {},
            custom_fields=timeline.custom_fields or {},
            is_active=timeline.is_active,
            is_completed=timeline.is_completed,
            is_overdue=timeline.is_overdue,
            duration_days=timeline.duration_days
        )
        
    except Exception as e:
        logger.error(f"Error updating timeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/timelines/{project_id}")
def delete_timeline(
    project_id: str,
    db: Session = Depends(get_db)
):
    """Delete project timeline"""
    
    try:
        service = ProjectTimelineService(db)
        success = service.delete_project_timeline(project_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Timeline not found")
        
        return {"message": "Timeline deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting timeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/timelines", response_model=List[TimelineResponse])
def list_timelines(
    owner_id: Optional[str] = Query(None, description="Filter by owner ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(50, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db)
):
    """List project timelines with filters"""
    
    try:
        service = ProjectTimelineService(db)
        timelines = service.list_project_timelines(
            owner_id=owner_id,
            status=status,
            category=category,
            priority=priority,
            limit=limit,
            offset=offset
        )
        
        return [
            TimelineResponse(
                id=str(timeline.id),
                project_id=timeline.project_id,
                project_name=timeline.project_name,
                project_description=timeline.project_description,
                status=timeline.status,
                priority=timeline.priority,
                category=timeline.category,
                tags=timeline.tags or [],
                created_at=timeline.created_at.isoformat(),
                updated_at=timeline.updated_at.isoformat(),
                started_at=timeline.started_at.isoformat() if timeline.started_at else None,
                planned_end_date=timeline.planned_end_date.isoformat() if timeline.planned_end_date else None,
                actual_end_date=timeline.actual_end_date.isoformat() if timeline.actual_end_date else None,
                progress_percentage=timeline.progress_percentage,
                total_tasks=timeline.total_tasks,
                completed_tasks=timeline.completed_tasks,
                owner_id=timeline.owner_id,
                team_members=timeline.team_members or [],
                external_references=timeline.external_references or {},
                custom_fields=timeline.custom_fields or {},
                is_active=timeline.is_active,
                is_completed=timeline.is_completed,
                is_overdue=timeline.is_overdue,
                duration_days=timeline.duration_days
            )
            for timeline in timelines
        ]
        
    except Exception as e:
        logger.error(f"Error listing timelines: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/timelines/overdue", response_model=List[TimelineResponse])
def get_overdue_timelines(
    owner_id: Optional[str] = Query(None, description="Filter by owner ID"),
    db: Session = Depends(get_db)
):
    """Get overdue project timelines"""
    
    try:
        service = ProjectTimelineService(db)
        timelines = service.get_overdue_projects(owner_id)
        
        return [
            TimelineResponse(
                id=str(timeline.id),
                project_id=timeline.project_id,
                project_name=timeline.project_name,
                project_description=timeline.project_description,
                status=timeline.status,
                priority=timeline.priority,
                category=timeline.category,
                tags=timeline.tags or [],
                created_at=timeline.created_at.isoformat(),
                updated_at=timeline.updated_at.isoformat(),
                started_at=timeline.started_at.isoformat() if timeline.started_at else None,
                planned_end_date=timeline.planned_end_date.isoformat() if timeline.planned_end_date else None,
                actual_end_date=timeline.actual_end_date.isoformat() if timeline.actual_end_date else None,
                progress_percentage=timeline.progress_percentage,
                total_tasks=timeline.total_tasks,
                completed_tasks=timeline.completed_tasks,
                owner_id=timeline.owner_id,
                team_members=timeline.team_members or [],
                external_references=timeline.external_references or {},
                custom_fields=timeline.custom_fields or {},
                is_active=timeline.is_active,
                is_completed=timeline.is_completed,
                is_overdue=timeline.is_overdue,
                duration_days=timeline.duration_days
            )
            for timeline in timelines
        ]
        
    except Exception as e:
        logger.error(f"Error getting overdue timelines: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Milestone Management Endpoints

@router.post("/timelines/{timeline_id}/milestones", response_model=MilestoneResponse)
def create_milestone(
    timeline_id: UUID,
    request: CreateMilestoneRequest,
    db: Session = Depends(get_db)
):
    """Create a new milestone"""
    
    try:
        service = ProjectTimelineService(db)
        
        milestone = service.create_milestone(
            timeline_id=timeline_id,
            name=request.name,
            milestone_type=request.milestone_type,
            description=request.description,
            planned_date=request.planned_date,
            assignee_id=request.assignee_id,
            weight=request.weight,
            tags=request.tags,
            custom_fields=request.custom_fields
        )
        
        if not milestone:
            raise HTTPException(status_code=400, detail="Failed to create milestone")
        
        return MilestoneResponse(
            id=str(milestone.id),
            timeline_id=str(milestone.timeline_id),
            name=milestone.name,
            description=milestone.description,
            milestone_type=milestone.milestone_type,
            status=milestone.status,
            created_at=milestone.created_at.isoformat(),
            updated_at=milestone.updated_at.isoformat(),
            planned_date=milestone.planned_date.isoformat() if milestone.planned_date else None,
            actual_date=milestone.actual_date.isoformat() if milestone.actual_date else None,
            progress_percentage=milestone.progress_percentage,
            weight=milestone.weight,
            assignee_id=milestone.assignee_id,
            tags=milestone.tags or [],
            custom_fields=milestone.custom_fields or {},
            auto_detected=milestone.auto_detected,
            detection_confidence=milestone.detection_confidence,
            is_completed=milestone.is_completed,
            is_overdue=milestone.is_overdue,
            days_until_due=milestone.days_until_due
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating milestone: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/timelines/{timeline_id}/milestones", response_model=List[MilestoneResponse])
def list_milestones(
    timeline_id: UUID,
    status: Optional[str] = Query(None, description="Filter by status"),
    milestone_type: Optional[str] = Query(None, description="Filter by milestone type"),
    assignee_id: Optional[str] = Query(None, description="Filter by assignee"),
    db: Session = Depends(get_db)
):
    """List milestones for a timeline"""
    
    try:
        service = ProjectTimelineService(db)
        milestones = service.list_milestones(
            timeline_id=timeline_id,
            status=status,
            milestone_type=milestone_type,
            assignee_id=assignee_id
        )
        
        return [
            MilestoneResponse(
                id=str(milestone.id),
                timeline_id=str(milestone.timeline_id),
                name=milestone.name,
                description=milestone.description,
                milestone_type=milestone.milestone_type,
                status=milestone.status,
                created_at=milestone.created_at.isoformat(),
                updated_at=milestone.updated_at.isoformat(),
                planned_date=milestone.planned_date.isoformat() if milestone.planned_date else None,
                actual_date=milestone.actual_date.isoformat() if milestone.actual_date else None,
                progress_percentage=milestone.progress_percentage,
                weight=milestone.weight,
                assignee_id=milestone.assignee_id,
                tags=milestone.tags or [],
                custom_fields=milestone.custom_fields or {},
                auto_detected=milestone.auto_detected,
                detection_confidence=milestone.detection_confidence,
                is_completed=milestone.is_completed,
                is_overdue=milestone.is_overdue,
                days_until_due=milestone.days_until_due
            )
            for milestone in milestones
        ]
        
    except Exception as e:
        logger.error(f"Error listing milestones: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/milestones/{milestone_id}", response_model=MilestoneResponse)
def update_milestone(
    milestone_id: UUID,
    request: UpdateMilestoneRequest,
    db: Session = Depends(get_db)
):
    """Update milestone"""
    
    try:
        service = ProjectTimelineService(db)
        
        # Convert request to dict, excluding None values
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        milestone = service.update_milestone(milestone_id, updates)
        
        if not milestone:
            raise HTTPException(status_code=404, detail="Milestone not found")
        
        return MilestoneResponse(
            id=str(milestone.id),
            timeline_id=str(milestone.timeline_id),
            name=milestone.name,
            description=milestone.description,
            milestone_type=milestone.milestone_type,
            status=milestone.status,
            created_at=milestone.created_at.isoformat(),
            updated_at=milestone.updated_at.isoformat(),
            planned_date=milestone.planned_date.isoformat() if milestone.planned_date else None,
            actual_date=milestone.actual_date.isoformat() if milestone.actual_date else None,
            progress_percentage=milestone.progress_percentage,
            weight=milestone.weight,
            assignee_id=milestone.assignee_id,
            tags=milestone.tags or [],
            custom_fields=milestone.custom_fields or {},
            auto_detected=milestone.auto_detected,
            detection_confidence=milestone.detection_confidence,
            is_completed=milestone.is_completed,
            is_overdue=milestone.is_overdue,
            days_until_due=milestone.days_until_due
        )
        
    except Exception as e:
        logger.error(f"Error updating milestone: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/milestones/{milestone_id}/complete")
def complete_milestone(
    milestone_id: UUID,
    completion_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Complete a milestone"""
    
    try:
        service = ProjectTimelineService(db)
        milestone = service.complete_milestone(milestone_id, completion_date)
        
        if not milestone:
            raise HTTPException(status_code=404, detail="Milestone not found")
        
        return {"message": "Milestone completed successfully", "milestone_id": str(milestone.id)}
        
    except Exception as e:
        logger.error(f"Error completing milestone: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/milestones/{milestone_id}")
def delete_milestone(
    milestone_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete milestone"""
    
    try:
        service = ProjectTimelineService(db)
        success = service.delete_milestone(milestone_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Milestone not found")
        
        return {"message": "Milestone deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting milestone: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Milestone Detection Endpoints

@router.post("/timelines/{timeline_id}/milestones/detect")
def detect_milestones(
    timeline_id: UUID,
    min_confidence: float = Query(0.6, ge=0.0, le=1.0, description="Minimum confidence for auto-creation"),
    max_milestones: int = Query(20, ge=1, le=50, description="Maximum milestones to create"),
    db: Session = Depends(get_db)
):
    """Automatically detect and create milestones"""
    
    try:
        detector = MilestoneDetector(db)
        created_milestones = detector.auto_create_milestones(
            timeline_id=timeline_id,
            min_confidence=min_confidence,
            max_milestones=max_milestones
        )
        
        return {
            "message": f"Created {len(created_milestones)} milestones",
            "created_milestones": created_milestones
        }
        
    except Exception as e:
        logger.error(f"Error detecting milestones: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Progress Analysis Endpoints

@router.get("/timelines/{timeline_id}/progress/analysis")
def analyze_progress(
    timeline_id: UUID,
    db: Session = Depends(get_db)
):
    """Get comprehensive progress analysis"""
    
    try:
        analyzer = ProgressAnalyzer(db)
        analysis = analyzer.analyze_project_progress(timeline_id)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing progress: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/timelines/{timeline_id}/insights")
def get_project_insights(
    timeline_id: UUID,
    db: Session = Depends(get_db)
):
    """Get project insights and recommendations"""
    
    try:
        service = ProjectTimelineService(db)
        insights = service.get_project_insights(timeline_id)
        
        return insights
        
    except Exception as e:
        logger.error(f"Error getting project insights: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/timelines/{timeline_id}/velocity")
def get_project_velocity(
    timeline_id: UUID,
    days_back: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """Get project velocity metrics"""
    
    try:
        service = ProjectTimelineService(db)
        velocity = service.calculate_project_velocity(timeline_id, days_back)
        
        return velocity
        
    except Exception as e:
        logger.error(f"Error calculating velocity: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/timelines/{timeline_id}/snapshots")
def create_progress_snapshot(
    timeline_id: UUID,
    created_by: Optional[str] = None,
    snapshot_metadata: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Create manual progress snapshot"""
    
    try:
        service = ProjectTimelineService(db)
        snapshot = service.create_progress_snapshot(
            timeline_id=timeline_id,
            created_by=created_by,
            snapshot_metadata=snapshot_metadata
        )
        
        if not snapshot:
            raise HTTPException(status_code=400, detail="Failed to create snapshot")
        
        return {
            "message": "Progress snapshot created successfully",
            "snapshot_id": str(snapshot.id),
            "progress_percentage": snapshot.progress_percentage,
            "velocity": snapshot.velocity
        }
        
    except Exception as e:
        logger.error(f"Error creating progress snapshot: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/timelines/{timeline_id}/snapshots")
def get_progress_history(
    timeline_id: UUID,
    limit: int = Query(30, ge=1, le=100, description="Number of snapshots to return"),
    db: Session = Depends(get_db)
):
    """Get progress history for a timeline"""
    
    try:
        service = ProjectTimelineService(db)
        snapshots = service.get_progress_history(timeline_id, limit)
        
        return [
            {
                "id": str(snapshot.id),
                "timeline_id": str(snapshot.timeline_id),
                "snapshot_date": snapshot.snapshot_date.isoformat(),
                "progress_percentage": snapshot.progress_percentage,
                "total_tasks": snapshot.total_tasks,
                "completed_tasks": snapshot.completed_tasks,
                "velocity": snapshot.velocity,
                "auto_generated": snapshot.auto_generated
            }
            for snapshot in snapshots
        ]
        
    except Exception as e:
        logger.error(f"Error getting progress history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")