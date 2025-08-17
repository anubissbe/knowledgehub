"""
Agent Workflow Data Models for Multi-Agent System.

This module defines the data models for the LangGraph-based multi-agent 
workflow system that orchestrates AI agents for complex tasks.
"""

from sqlalchemy import Column, String, Text, JSON, DateTime, Float, Integer, Boolean, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import uuid

from .base import Base


class AgentRole(str, Enum):
    """Available agent roles in the system."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"
    PLANNER = "planner"
    EXECUTOR = "executor"


class WorkflowType(str, Enum):
    """Types of workflows available."""
    SIMPLE_QA = "simple_qa"
    MULTI_STEP_RESEARCH = "multi_step_research"
    COMPLEX_ANALYSIS = "complex_analysis"
    PLANNING_WORKFLOW = "planning_workflow"
    VALIDATION_WORKFLOW = "validation_workflow"
    CUSTOM = "custom"


class ExecutionStatus(str, Enum):
    """Execution status for workflows and tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskType(str, Enum):
    """Types of tasks that agents can perform."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"


class AgentDefinition(Base):
    """Agent definitions and configurations."""
    __tablename__ = "agent_definitions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    role = Column(String(50), nullable=False)
    description = Column(Text)
    
    # Capabilities and Configuration
    capabilities = Column(JSON, default=lambda: [])
    tools_available = Column(JSON, default=lambda: [])
    model_config = Column(JSON, default=lambda: {})
    system_prompt = Column(Text)
    
    # Performance and Limits
    max_concurrent_tasks = Column(Integer, default=1)
    timeout_seconds = Column(Integer, default=300)
    rate_limit_per_minute = Column(Integer, default=60)
    
    # State
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    tasks = relationship("AgentTask", back_populates="agent")
    
    __table_args__ = (
        Index('idx_agent_definitions_role', 'role'),
        Index('idx_agent_definitions_active', 'is_active'),
        Index('idx_agent_definitions_name', 'name'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools_available": self.tools_available,
            "model_config": self.model_config,
            "system_prompt": self.system_prompt,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "timeout_seconds": self.timeout_seconds,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class WorkflowDefinition(Base):
    """Workflow definitions with graph structure."""
    __tablename__ = "workflow_definitions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    workflow_type = Column(String(50), nullable=False)
    
    # Graph Definition (LangGraph structure)
    graph_definition = Column(JSON, nullable=False)
    entry_point = Column(String(100), nullable=False)
    exit_points = Column(JSON, default=lambda: [])
    
    # Configuration
    config = Column(JSON, default=lambda: {})
    agents_required = Column(JSON, default=lambda: [])
    tools_required = Column(JSON, default=lambda: [])
    
    # Metadata
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_by = Column(UUID(as_uuid=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    executions = relationship("WorkflowExecution", back_populates="workflow")
    
    __table_args__ = (
        Index('idx_workflow_definitions_type', 'workflow_type'),
        Index('idx_workflow_definitions_active', 'is_active'),
        Index('idx_workflow_definitions_name', 'name'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "workflow_type": self.workflow_type,
            "graph_definition": self.graph_definition,
            "entry_point": self.entry_point,
            "exit_points": self.exit_points,
            "config": self.config,
            "agents_required": self.agents_required,
            "tools_required": self.tools_required,
            "version": self.version,
            "is_active": self.is_active,
            "created_by": str(self.created_by) if self.created_by else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class WorkflowExecution(Base):
    """Workflow execution instances with state management."""
    __tablename__ = "workflow_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflow_definitions.id'), nullable=False)
    session_id = Column(String(255))
    user_id = Column(String(255), nullable=False)
    
    # Execution Details
    execution_type = Column(String(50), default="synchronous")
    status = Column(String(50), default=ExecutionStatus.PENDING.value)
    
    # Input/Output
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON)
    error_details = Column(JSON)
    
    # Execution Metadata
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    execution_time_ms = Column(Integer)
    
    # State Management (for LangGraph checkpoints)
    checkpoint_data = Column(JSON)
    current_state = Column(JSON, default=lambda: {})
    
    # Performance Tracking
    steps_completed = Column(Integer, default=0)
    agents_involved = Column(JSON, default=lambda: [])
    tools_used = Column(JSON, default=lambda: [])
    
    # Relationships
    workflow = relationship("WorkflowDefinition", back_populates="executions")
    tasks = relationship("AgentTask", back_populates="workflow_execution")
    
    __table_args__ = (
        Index('idx_workflow_executions_workflow_id', 'workflow_id'),
        Index('idx_workflow_executions_user_id', 'user_id'),
        Index('idx_workflow_executions_status', 'status'),
        Index('idx_workflow_executions_started_at', 'started_at'),
        Index('idx_workflow_executions_session_id', 'session_id'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "workflow_id": str(self.workflow_id),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "execution_type": self.execution_type,
            "status": self.status,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_details": self.error_details,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "checkpoint_data": self.checkpoint_data,
            "current_state": self.current_state,
            "steps_completed": self.steps_completed,
            "agents_involved": self.agents_involved,
            "tools_used": self.tools_used,
        }
    
    def calculate_execution_time(self):
        """Calculate execution time if completed."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.execution_time_ms = int(delta.total_seconds() * 1000)


class AgentTask(Base):
    """Individual tasks within workflows executed by agents."""
    __tablename__ = "agent_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_execution_id = Column(UUID(as_uuid=True), ForeignKey('workflow_executions.id'), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_definitions.id'), nullable=False)
    
    # Task Details
    task_type = Column(String(50), nullable=False)
    step_name = Column(String(100))
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON)
    
    # Execution
    status = Column(String(50), default=ExecutionStatus.PENDING.value)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    execution_time_ms = Column(Integer)
    
    # Error Handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Dependencies
    depends_on_tasks = Column(JSON, default=lambda: [])
    blocked_by_tasks = Column(JSON, default=lambda: [])
    
    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="tasks")
    agent = relationship("AgentDefinition", back_populates="tasks")
    
    __table_args__ = (
        Index('idx_agent_tasks_workflow_execution', 'workflow_execution_id'),
        Index('idx_agent_tasks_agent_id', 'agent_id'),
        Index('idx_agent_tasks_status', 'status'),
        Index('idx_agent_tasks_task_type', 'task_type'),
        Index('idx_agent_tasks_started_at', 'started_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "workflow_execution_id": str(self.workflow_execution_id),
            "agent_id": str(self.agent_id),
            "task_type": self.task_type,
            "step_name": self.step_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "depends_on_tasks": self.depends_on_tasks,
            "blocked_by_tasks": self.blocked_by_tasks,
        }
    
    def can_execute(self) -> bool:
        """Check if task can be executed (no blocking dependencies)."""
        return len(self.blocked_by_tasks) == 0 and self.status == ExecutionStatus.PENDING.value
    
    def calculate_execution_time(self):
        """Calculate execution time if completed."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.execution_time_ms = int(delta.total_seconds() * 1000)


# Pydantic Models for API

class AgentDefinitionCreate(BaseModel):
    """Schema for creating agent definitions."""
    name: str = Field(..., min_length=1, max_length=100)
    role: AgentRole
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    tools_available: List[str] = Field(default_factory=list)
    model_config: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: Optional[str] = None
    max_concurrent_tasks: int = Field(default=1, ge=1, le=10)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip().lower()


class AgentDefinitionUpdate(BaseModel):
    """Schema for updating agent definitions."""
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    tools_available: Optional[List[str]] = None
    model_config: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    max_concurrent_tasks: Optional[int] = Field(None, ge=1, le=10)
    timeout_seconds: Optional[int] = Field(None, ge=30, le=3600)
    rate_limit_per_minute: Optional[int] = Field(None, ge=1, le=1000)
    is_active: Optional[bool] = None


class WorkflowDefinitionCreate(BaseModel):
    """Schema for creating workflow definitions."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    workflow_type: WorkflowType
    graph_definition: Dict[str, Any] = Field(..., description="LangGraph workflow structure")
    entry_point: str = Field(..., min_length=1)
    exit_points: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    agents_required: List[str] = Field(default_factory=list)
    tools_required: List[str] = Field(default_factory=list)
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip().lower()
    
    @field_validator('graph_definition')
    def validate_graph_definition(cls, v):
        required_keys = ['nodes', 'edges']
        for key in required_keys:
            if key not in v:
                raise ValueError(f'Graph definition must contain {key}')
        return v


class WorkflowExecutionCreate(BaseModel):
    """Schema for creating workflow executions."""
    workflow_id: str = Field(..., description="UUID of workflow definition")
    session_id: Optional[str] = None
    execution_type: str = Field(default="synchronous")
    input_data: Dict[str, Any] = Field(..., description="Input data for workflow")
    
    @field_validator('workflow_id')
    def validate_workflow_id(cls, v):
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('Invalid workflow ID format')


class AgentTaskCreate(BaseModel):
    """Schema for creating agent tasks."""
    workflow_execution_id: str = Field(..., description="UUID of workflow execution")
    agent_id: str = Field(..., description="UUID of agent definition")
    task_type: TaskType
    step_name: Optional[str] = None
    input_data: Dict[str, Any] = Field(..., description="Input data for task")
    depends_on_tasks: List[str] = Field(default_factory=list, description="Task IDs this depends on")
    max_retries: int = Field(default=3, ge=0, le=10)


class WorkflowExecutionResponse(BaseModel):
    """Schema for workflow execution responses."""
    id: str
    workflow_id: str
    session_id: Optional[str]
    user_id: str
    execution_type: str
    status: ExecutionStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error_details: Optional[Dict[str, Any]]
    started_at: datetime
    completed_at: Optional[datetime]
    execution_time_ms: Optional[int]
    steps_completed: int
    agents_involved: List[str]
    tools_used: List[str]
    
    model_config = ConfigDict(from_attributes=True)


class AgentTaskResponse(BaseModel):
    """Schema for agent task responses."""
    id: str
    workflow_execution_id: str
    agent_id: str
    task_type: TaskType
    step_name: Optional[str]
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime]
    execution_time_ms: Optional[int]
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    
    model_config = ConfigDict(from_attributes=True)


class WorkflowPerformanceMetrics(BaseModel):
    """Schema for workflow performance metrics."""
    workflow_name: str
    workflow_type: WorkflowType
    total_executions: int
    avg_execution_time: float
    successful_executions: int
    failed_executions: int
    success_rate: float
    last_execution: Optional[datetime]


class AgentPerformanceMetrics(BaseModel):
    """Schema for agent performance metrics."""
    agent_name: str
    agent_role: AgentRole
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_execution_time: float
    success_rate: float
    tools_used_count: Dict[str, int]


class WorkflowAnalytics(BaseModel):
    """Schema for workflow analytics data."""
    total_workflows: int
    active_workflows: int
    total_executions: int
    executions_by_status: Dict[str, int]
    executions_by_type: Dict[str, int]
    avg_execution_time: float
    success_rate: float
    most_used_workflows: List[Dict[str, Any]]
    performance_trends: List[Dict[str, Any]]