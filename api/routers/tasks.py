"""Task router for task tracking system"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, desc, func
from uuid import UUID
from typing import Optional, List, Dict, Any
import logging
from pydantic import BaseModel, Field
from datetime import datetime

from ..database import get_db_session
from ..models.task import Task, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for API requests
class TaskCreate(BaseModel):
    """Model for creating a task"""
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    assignee: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    estimated_hours: Optional[int] = None
    due_date: Optional[datetime] = None
    created_by: Optional[str] = None


class TaskUpdate(BaseModel):
    """Model for updating a task"""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assignee: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    progress: Optional[int] = Field(None, ge=0, le=100)
    estimated_hours: Optional[int] = None
    actual_hours: Optional[int] = None
    due_date: Optional[datetime] = None


class TaskStatusUpdate(BaseModel):
    """Model for task status updates"""
    status: TaskStatus
    progress: Optional[int] = Field(None, ge=0, le=100)
    actual_hours: Optional[int] = None


@router.get("/tasks")
@router.get("/api/tasks")
async def list_tasks(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    priority: Optional[TaskPriority] = Query(None, description="Filter by task priority"),
    assignee: Optional[str] = Query(None, description="Filter by assignee"),
    is_active: Optional[bool] = Query(True, description="Filter active/inactive tasks"),
    search: Optional[str] = Query(None, description="Search in title and description"),
    overdue_only: bool = Query(False, description="Show only overdue tasks"),
    db: AsyncSession = Depends(get_db_session)
):
    """List all tasks with optional filtering and pagination"""
    try:
        # Build query
        query = select(Task)
        
        # Apply filters
        conditions = []
        
        if status:
            conditions.append(Task.status == status)
        
        if priority:
            conditions.append(Task.priority == priority)
            
        if assignee:
            conditions.append(Task.assignee == assignee)
            
        if is_active is not None:
            conditions.append(Task.is_active == is_active)
            
        if search:
            conditions.append(
                or_(
                    Task.title.ilike(f"%{search}%"),
                    Task.description.ilike(f"%{search}%")
                )
            )
            
        if overdue_only:
            conditions.append(
                and_(
                    Task.due_date < datetime.utcnow(),
                    Task.status.notin_([TaskStatus.COMPLETED, TaskStatus.CANCELLED])
                )
            )
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Order by created date (descending - newest first)
        query = query.order_by(desc(Task.created_at))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        tasks = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(func.count(Task.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        
        count_result = await db.execute(count_query)
        total = count_result.scalar()
        
        return {
            "tasks": [task.to_dict() for task in tasks],
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tasks: {str(e)}")


@router.post("/tasks")
async def create_task(
    task_data: TaskCreate,
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new task"""
    try:
        # Create task
        task = Task(
            title=task_data.title,
            description=task_data.description,
            priority=task_data.priority,
            assignee=task_data.assignee,
            tags=task_data.tags,
            task_metadata=task_data.metadata,
            estimated_hours=task_data.estimated_hours,
            due_date=task_data.due_date,
            created_by=task_data.created_by
        )
        
        db.add(task)
        await db.commit()
        await db.refresh(task)
        
        logger.info(f"Created task: {task.id}")
        return task.to_dict()
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/task/{task_id}")
async def get_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db_session)
):
    """Get a specific task by ID"""
    try:
        query = select(Task).where(Task.id == task_id)
        result = await db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return task.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting task: {str(e)}")


@router.put("/task/{task_id}")
async def update_task(
    task_id: UUID,
    task_data: TaskUpdate,
    db: AsyncSession = Depends(get_db_session)
):
    """Update a task"""
    try:
        # Get existing task
        query = select(Task).where(Task.id == task_id)
        result = await db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Update fields
        update_data = {}
        for field, value in task_data.dict(exclude_unset=True).items():
            update_data[field] = value
        
        # Handle status changes
        if 'status' in update_data:
            if update_data['status'] == TaskStatus.IN_PROGRESS and not task.started_at:
                update_data['started_at'] = datetime.utcnow()
            elif update_data['status'] == TaskStatus.COMPLETED and not task.completed_at:
                update_data['completed_at'] = datetime.utcnow()
                update_data['progress'] = 100
        
        update_data['updated_at'] = datetime.utcnow()
        
        # Execute update
        stmt = update(Task).where(Task.id == task_id).values(**update_data)
        await db.execute(stmt)
        await db.commit()
        
        # Get updated task
        result = await db.execute(query)
        updated_task = result.scalar_one()
        
        logger.info(f"Updated task: {task_id}")
        return updated_task.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating task: {str(e)}")


@router.patch("/task/{task_id}/status")
async def update_task_status(
    task_id: UUID,
    status_data: TaskStatusUpdate,
    db: AsyncSession = Depends(get_db_session)
):
    """Update task status and progress"""
    try:
        # Get existing task
        query = select(Task).where(Task.id == task_id)
        result = await db.execute(query)
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Prepare update data
        update_data = {
            'status': status_data.status,
            'updated_at': datetime.utcnow()
        }
        
        if status_data.progress is not None:
            update_data['progress'] = status_data.progress
            
        if status_data.actual_hours is not None:
            update_data['actual_hours'] = status_data.actual_hours
        
        # Handle status-specific updates
        if status_data.status == TaskStatus.IN_PROGRESS and not task.started_at:
            update_data['started_at'] = datetime.utcnow()
        elif status_data.status == TaskStatus.COMPLETED and not task.completed_at:
            update_data['completed_at'] = datetime.utcnow()
            update_data['progress'] = 100
        
        # Execute update
        stmt = update(Task).where(Task.id == task_id).values(**update_data)
        await db.execute(stmt)
        await db.commit()
        
        # Get updated task
        result = await db.execute(query)
        updated_task = result.scalar_one()
        
        logger.info(f"Updated task status: {task_id} -> {status_data.status}")
        return updated_task.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating task status {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating task status: {str(e)}")


@router.delete("/task/{task_id}")
async def delete_task(
    task_id: UUID,
    hard_delete: bool = Query(False, description="Permanently delete (default: soft delete)"),
    db: AsyncSession = Depends(get_db_session)
):
    """Delete a task (soft delete by default)"""
    try:
        if hard_delete:
            # Hard delete
            stmt = delete(Task).where(Task.id == task_id)
            result = await db.execute(stmt)
            
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Task not found")
            
            await db.commit()
            logger.info(f"Hard deleted task: {task_id}")
            return {"message": "Task permanently deleted"}
        else:
            # Soft delete
            stmt = update(Task).where(Task.id == task_id).values(is_active=False, updated_at=datetime.utcnow())
            result = await db.execute(stmt)
            
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Task not found")
            
            await db.commit()
            logger.info(f"Soft deleted task: {task_id}")
            return {"message": "Task deactivated"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting task: {str(e)}")


@router.get("/tasks/stats")
async def get_task_stats(
    assignee: Optional[str] = Query(None, description="Filter stats by assignee"),
    db: AsyncSession = Depends(get_db_session)
):
    """Get task statistics summary"""
    try:
        # Base query
        query = select(Task).where(Task.is_active == True)
        
        if assignee:
            query = query.where(Task.assignee == assignee)
        
        result = await db.execute(query)
        tasks = result.scalars().all()
        
        # Calculate stats
        stats = {
            "total": len(tasks),
            "by_status": {},
            "by_priority": {},
            "overdue": 0,
            "completed_this_month": 0,
            "avg_completion_time_hours": None
        }
        
        completion_times = []
        now = datetime.utcnow()
        
        for task in tasks:
            # Status stats
            status_key = task.status.value if isinstance(task.status, TaskStatus) else task.status
            stats["by_status"][status_key] = stats["by_status"].get(status_key, 0) + 1
            
            # Priority stats
            priority_key = task.priority.value if isinstance(task.priority, TaskPriority) else task.priority
            stats["by_priority"][priority_key] = stats["by_priority"].get(priority_key, 0) + 1
            
            # Overdue tasks
            if task.is_overdue:
                stats["overdue"] += 1
            
            # Completed this month
            if task.completed_at and task.completed_at.month == now.month and task.completed_at.year == now.year:
                stats["completed_this_month"] += 1
            
            # Completion times
            if task.duration_hours:
                completion_times.append(task.duration_hours)
        
        # Average completion time
        if completion_times:
            stats["avg_completion_time_hours"] = sum(completion_times) / len(completion_times)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting task stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting task stats: {str(e)}")