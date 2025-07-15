"""Job schemas"""

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class JobStatus(str, Enum):
    """Job status enum"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreate(BaseModel):
    """Schema for creating a job"""
    source_id: str
    job_type: str
    config: Optional[dict] = {}


class JobResponse(BaseModel):
    """Schema for job response"""
    job_id: str
    message: str
    status: str