"""
Workflow Automation Models.

This module defines models for workflow pattern detection,
automation rules, task templates, and execution monitoring.
"""

from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime, JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from .base import Base, TimeStampedModel


class WorkflowTriggerType(str, Enum):
    """Types of workflow triggers."""
    TIME_BASED = "time_based"          # Scheduled execution
    EVENT_BASED = "event_based"        # Triggered by events
    CONDITION_BASED = "condition_based"  # Triggered by conditions
    MANUAL = "manual"                   # Manual execution


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PatternType(str, Enum):
    """Types of workflow patterns."""
    SEQUENTIAL = "sequential"          # Tasks in sequence
    PARALLEL = "parallel"              # Tasks in parallel
    CONDITIONAL = "conditional"        # Conditional branching
    LOOP = "loop"                      # Repeating patterns
    ERROR_HANDLING = "error_handling"  # Error recovery patterns


# SQLAlchemy Models

class WorkflowPattern(TimeStampedModel):
    """
    Detected workflow patterns from user behavior.
    
    Automatically learns common task sequences and suggests automation.
    """
    __tablename__ = "workflow_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_name = Column(String(200), nullable=False)
    pattern_type = Column(String(50), nullable=False)  # PatternType enum
    description = Column(Text)
    
    # Pattern definition
    tasks_sequence = Column(JSONB, nullable=False)  # List of task definitions
    conditions = Column(JSONB, default=dict)        # Conditions for pattern activation
    trigger_events = Column(JSONB, default=list)    # Events that trigger pattern
    
    # Statistics
    occurrence_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    avg_execution_time = Column(Float)  # Hours
    avg_effort_saved = Column(Float)    # Hours
    
    # Pattern quality metrics
    confidence_score = Column(Float, default=0.0)    # How confident we are this is a pattern
    automation_potential = Column(Float, default=0.0)  # How automatable this pattern is
    user_adoption_rate = Column(Float, default=0.0)   # How often users accept automation
    
    # Context
    user_id = Column(String(100))
    project_id = Column(String(100))
    team_id = Column(String(100))
    context_tags = Column(JSONB, default=list)
    
    # Learning metadata
    first_detected = Column(DateTime, default=func.now())
    last_detected = Column(DateTime)
    detection_algorithm = Column(String(100))
    pattern_version = Column(Integer, default=1)
    
    # Relationships
    workflows = relationship("WorkflowTemplate", back_populates="pattern")
    executions = relationship("WorkflowExecution", back_populates="pattern")
    
    __table_args__ = (
        Index('idx_workflow_patterns_user_project', 'user_id', 'project_id'),
        Index('idx_workflow_patterns_type_confidence', 'pattern_type', 'confidence_score'),
        Index('idx_workflow_patterns_success_rate', 'success_rate'),
    )
    
    def calculate_efficiency_gain(self) -> float:
        """Calculate efficiency gain from automation."""
        if self.avg_execution_time and self.avg_effort_saved:
            return min(self.avg_effort_saved / self.avg_execution_time, 1.0)
        return 0.0
    
    def get_automation_recommendation(self) -> Dict[str, Any]:
        """Get automation recommendation based on pattern analysis."""
        return {
            "should_automate": self.automation_potential > 0.7 and self.success_rate > 0.8,
            "confidence": self.confidence_score,
            "potential_savings_hours": self.avg_effort_saved * self.occurrence_count,
            "risk_level": "low" if self.success_rate > 0.9 else "medium" if self.success_rate > 0.7 else "high",
            "recommendation": self._generate_recommendation()
        }
    
    def _generate_recommendation(self) -> str:
        """Generate automation recommendation text."""
        if self.automation_potential > 0.8:
            return f"Highly recommended for automation. Detected {self.occurrence_count} times with {self.success_rate:.1%} success rate."
        elif self.automation_potential > 0.6:
            return f"Good candidate for automation. Could save {self.avg_effort_saved:.1f} hours per execution."
        else:
            return f"Pattern detected but automation not recommended. Success rate: {self.success_rate:.1%}"


class WorkflowTemplate(TimeStampedModel):
    """
    Reusable workflow templates for automation.
    
    Defines how to execute a workflow pattern automatically.
    """
    __tablename__ = "workflow_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Template definition
    template_definition = Column(JSONB, nullable=False)  # Workflow steps and logic
    input_schema = Column(JSONB, default=dict)           # Expected inputs
    output_schema = Column(JSONB, default=dict)          # Expected outputs
    
    # Execution configuration
    trigger_type = Column(String(50), nullable=False)    # WorkflowTriggerType enum
    trigger_config = Column(JSONB, default=dict)         # Trigger-specific configuration
    timeout_minutes = Column(Integer, default=60)
    retry_count = Column(Integer, default=3)
    
    # Automation settings
    auto_approval_required = Column(Boolean, default=True)
    risk_level = Column(String(20), default="medium")    # low, medium, high
    estimated_duration = Column(Float)  # Hours
    
    # Relationships
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('workflow_patterns.id'))
    pattern = relationship("WorkflowPattern", back_populates="workflows")
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    
    # Metadata
    created_by = Column(String(100))
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    tags = Column(JSONB, default=list)
    
    __table_args__ = (
        Index('idx_workflow_templates_trigger_type', 'trigger_type'),
        Index('idx_workflow_templates_active', 'is_active'),
        Index('idx_workflow_templates_created_by', 'created_by'),
    )
    
    def calculate_success_rate(self) -> float:
        """Calculate template success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self.usage_count,
            "successful_executions": self.success_count,
            "success_rate": self.calculate_success_rate(),
            "estimated_time_saved": self.success_count * (self.estimated_duration or 0),
            "last_used": self.updated_at.isoformat() if self.updated_at else None
        }


class WorkflowExecution(TimeStampedModel):
    """
    Individual workflow execution instances.
    
    Tracks the execution of workflow templates with detailed monitoring.
    """
    __tablename__ = "workflow_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_name = Column(String(200))
    
    # Execution context
    template_id = Column(UUID(as_uuid=True), ForeignKey('workflow_templates.id'))
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('workflow_patterns.id'))
    
    # Execution details
    status = Column(String(50), nullable=False, default=WorkflowStatus.DRAFT.value)
    trigger_type = Column(String(50), nullable=False)
    trigger_data = Column(JSONB, default=dict)          # Data that triggered execution
    
    # Input/Output
    input_data = Column(JSONB, default=dict)
    output_data = Column(JSONB, default=dict)
    execution_context = Column(JSONB, default=dict)     # Runtime context
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Progress tracking
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    current_step = Column(String(200))
    progress_percentage = Column(Float, default=0.0)
    
    # Results
    success = Column(Boolean)
    error_message = Column(Text)
    error_details = Column(JSONB)
    
    # User context
    executed_by = Column(String(100))                   # User who started execution
    approved_by = Column(String(100))                   # User who approved (if required)
    project_id = Column(String(100))
    session_id = Column(String(100))
    
    # Relationships
    pattern = relationship("WorkflowPattern", back_populates="executions")
    tasks = relationship("TaskExecution", back_populates="workflow")
    
    __table_args__ = (
        Index('idx_workflow_executions_status_started', 'status', 'started_at'),
        Index('idx_workflow_executions_user_project', 'executed_by', 'project_id'),
        Index('idx_workflow_executions_template', 'template_id'),
        CheckConstraint('completed_steps <= total_steps', name='check_execution_progress'),
    )
    
    def calculate_duration(self) -> Optional[float]:
        """Calculate execution duration in hours."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() / 3600
        return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        duration = self.calculate_duration()
        
        return {
            "execution_id": str(self.id),
            "status": self.status,
            "progress": f"{self.completed_steps}/{self.total_steps} steps",
            "progress_percentage": self.progress_percentage,
            "duration_hours": duration,
            "success": self.success,
            "error": self.error_message if not self.success else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class TaskExecution(TimeStampedModel):
    """
    Individual task execution within a workflow.
    
    Provides detailed tracking of each step in a workflow execution.
    """
    __tablename__ = "task_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_name = Column(String(200), nullable=False)
    task_type = Column(String(100), nullable=False)
    
    # Execution context
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflow_executions.id'))
    step_number = Column(Integer, nullable=False)
    
    # Task definition
    task_config = Column(JSONB, nullable=False)         # Task configuration
    input_data = Column(JSONB, default=dict)
    output_data = Column(JSONB, default=dict)
    
    # Execution details
    status = Column(String(50), nullable=False, default=ExecutionStatus.PENDING.value)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Results
    success = Column(Boolean)
    error_message = Column(Text)
    error_details = Column(JSONB)
    retry_count = Column(Integer, default=0)
    
    # Performance metrics
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    api_calls_made = Column(Integer, default=0)
    
    # Relationships
    workflow = relationship("WorkflowExecution", back_populates="tasks")
    
    __table_args__ = (
        Index('idx_task_executions_workflow_step', 'workflow_id', 'step_number'),
        Index('idx_task_executions_status', 'status'),
        UniqueConstraint('workflow_id', 'step_number', name='uq_workflow_step'),
    )
    
    def calculate_duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get task performance metrics."""
        return {
            "duration_seconds": self.calculate_duration(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "api_calls": self.api_calls_made,
            "retry_count": self.retry_count,
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score based on performance metrics."""
        base_score = 1.0
        
        # Penalize high resource usage
        if self.cpu_usage and self.cpu_usage > 0.8:
            base_score -= 0.2
        if self.memory_usage and self.memory_usage > 0.8:
            base_score -= 0.2
        
        # Penalize retries
        if self.retry_count > 0:
            base_score -= min(0.3, self.retry_count * 0.1)
        
        return max(0.0, base_score)


class AutomationRule(TimeStampedModel):
    """
    Rules for automatic workflow triggering.
    
    Defines when and how workflows should be automatically executed.
    """
    __tablename__ = "automation_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Rule definition
    trigger_conditions = Column(JSONB, nullable=False)   # Conditions that trigger rule
    action_template_id = Column(UUID(as_uuid=True), ForeignKey('workflow_templates.id'))
    
    # Rule configuration
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=100)              # Lower = higher priority
    cooldown_minutes = Column(Integer, default=60)       # Min time between executions
    
    # Approval settings
    requires_approval = Column(Boolean, default=True)
    auto_approve_conditions = Column(JSONB, default=dict)
    approval_timeout_minutes = Column(Integer, default=1440)  # 24 hours
    
    # Constraints
    max_executions_per_day = Column(Integer, default=10)
    max_concurrent_executions = Column(Integer, default=1)
    
    # Context filters
    user_filters = Column(JSONB, default=list)           # User ID patterns
    project_filters = Column(JSONB, default=list)        # Project ID patterns
    time_constraints = Column(JSONB, default=dict)       # Time-based constraints
    
    # Usage tracking
    execution_count = Column(Integer, default=0)
    last_execution = Column(DateTime)
    success_count = Column(Integer, default=0)
    
    # Metadata
    created_by = Column(String(100))
    
    __table_args__ = (
        Index('idx_automation_rules_active_priority', 'is_active', 'priority'),
        Index('idx_automation_rules_template', 'action_template_id'),
        Index('idx_automation_rules_created_by', 'created_by'),
    )
    
    def can_execute(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check if rule can execute in given context."""
        # Check if rule is active
        if not self.is_active:
            return False, "Rule is not active"
        
        # Check cooldown
        if self.last_execution:
            cooldown_delta = timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() - self.last_execution < cooldown_delta:
                return False, "Rule is in cooldown period"
        
        # Check daily execution limit
        today = datetime.utcnow().date()
        # In production, would query actual executions for today
        # For now, simplified check
        
        # Check filters
        user_id = context.get("user_id")
        if self.user_filters and user_id not in self.user_filters:
            return False, "User not in allowed filters"
        
        project_id = context.get("project_id")
        if self.project_filters and project_id not in self.project_filters:
            return False, "Project not in allowed filters"
        
        return True, "Rule can execute"
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """Get rule execution statistics."""
        success_rate = 0.0
        if self.execution_count > 0:
            success_rate = self.success_count / self.execution_count
        
        return {
            "total_executions": self.execution_count,
            "successful_executions": self.success_count,
            "success_rate": success_rate,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "is_active": self.is_active,
            "priority": self.priority
        }


# Pydantic Models for API

class WorkflowPatternCreate(BaseModel):
    """Create workflow pattern request."""
    pattern_name: str = Field(..., min_length=1, max_length=200)
    pattern_type: PatternType
    description: Optional[str] = None
    tasks_sequence: List[Dict[str, Any]]
    conditions: Dict[str, Any] = Field(default_factory=dict)
    trigger_events: List[str] = Field(default_factory=list)
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    context_tags: List[str] = Field(default_factory=list)


class WorkflowTemplateCreate(BaseModel):
    """Create workflow template request."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    template_definition: Dict[str, Any]
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    trigger_type: WorkflowTriggerType
    trigger_config: Dict[str, Any] = Field(default_factory=dict)
    timeout_minutes: int = Field(60, ge=1, le=1440)
    retry_count: int = Field(3, ge=0, le=10)
    auto_approval_required: bool = True
    risk_level: str = Field("medium", pattern="^(low|medium|high)$")
    estimated_duration: Optional[float] = Field(None, ge=0)
    pattern_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class WorkflowExecutionCreate(BaseModel):
    """Create workflow execution request."""
    execution_name: Optional[str] = None
    template_id: str
    trigger_type: WorkflowTriggerType
    trigger_data: Dict[str, Any] = Field(default_factory=dict)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    project_id: Optional[str] = None
    session_id: Optional[str] = None


class AutomationRuleCreate(BaseModel):
    """Create automation rule request."""
    rule_name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    trigger_conditions: Dict[str, Any]
    action_template_id: str
    is_active: bool = True
    priority: int = Field(100, ge=1, le=1000)
    cooldown_minutes: int = Field(60, ge=0, le=10080)  # Max 1 week
    requires_approval: bool = True
    auto_approve_conditions: Dict[str, Any] = Field(default_factory=dict)
    approval_timeout_minutes: int = Field(1440, ge=60, le=10080)
    max_executions_per_day: int = Field(10, ge=1, le=100)
    max_concurrent_executions: int = Field(1, ge=1, le=10)
    user_filters: List[str] = Field(default_factory=list)
    project_filters: List[str] = Field(default_factory=list)
    time_constraints: Dict[str, Any] = Field(default_factory=dict)


class WorkflowPatternResponse(BaseModel):
    """Workflow pattern response."""
    id: str
    pattern_name: str
    pattern_type: str
    description: Optional[str]
    occurrence_count: int
    success_rate: float
    automation_potential: float
    confidence_score: float
    avg_execution_time: Optional[float]
    avg_effort_saved: Optional[float]
    first_detected: datetime
    last_detected: Optional[datetime]
    
    model_config = ConfigDict(from_attributes = True)


class WorkflowExecutionResponse(BaseModel):
    """Workflow execution response."""
    id: str
    execution_name: Optional[str]
    status: str
    progress_percentage: float
    total_steps: int
    completed_steps: int
    current_step: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    success: Optional[bool]
    error_message: Optional[str]
    executed_by: Optional[str]
    
    model_config = ConfigDict(from_attributes = True)


class WorkflowAnalytics(BaseModel):
    """Workflow analytics response."""
    total_patterns: int
    active_templates: int
    total_executions: int
    success_rate: float
    avg_execution_time: float
    total_time_saved: float
    automation_adoption_rate: float
    top_patterns: List[Dict[str, Any]]
    execution_trends: List[Dict[str, Any]]
    efficiency_metrics: Dict[str, Any]