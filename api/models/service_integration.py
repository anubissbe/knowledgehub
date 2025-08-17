"""
Service Integration Data Models.

This module defines data models for integrating external services like Zep memory,
Firecrawl, and other third-party services into the KnowledgeHub ecosystem.
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


class ServiceType(str, Enum):
    """Types of integrated services."""
    ZEP_MEMORY = "zep_memory"
    FIRECRAWL = "firecrawl"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    WEAVIATE = "weaviate"
    NEO4J = "neo4j"
    TIMESCALE = "timescale"
    REDIS = "redis"
    MINIO = "minio"


class SyncStatus(str, Enum):
    """Synchronization status between services."""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SYNCING = "syncing"
    DISCONNECTED = "disconnected"


class HealthStatus(str, Enum):
    """Health status for services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    ERROR = "error"
    UNKNOWN = "unknown"


class ZepSessionMapping(Base):
    """Mapping between KnowledgeHub and Zep sessions."""
    __tablename__ = "zep_session_mapping"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    knowledgehub_session_id = Column(String(255), nullable=False, unique=True)
    zep_session_id = Column(String(255), nullable=False, unique=True)
    user_id = Column(String(255), nullable=False)
    
    # Sync State
    last_sync_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    sync_status = Column(String(50), default=SyncStatus.ACTIVE.value)
    
    # Configuration
    zep_config = Column(JSON, default=lambda: {})
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_zep_session_mapping_kh_session', 'knowledgehub_session_id'),
        Index('idx_zep_session_mapping_zep_session', 'zep_session_id'),
        Index('idx_zep_session_mapping_user_id', 'user_id'),
        Index('idx_zep_session_mapping_status', 'sync_status'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "knowledgehub_session_id": self.knowledgehub_session_id,
            "zep_session_id": self.zep_session_id,
            "user_id": self.user_id,
            "last_sync_timestamp": self.last_sync_timestamp.isoformat() if self.last_sync_timestamp else None,
            "sync_status": self.sync_status,
            "zep_config": self.zep_config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ServiceHealthLog(Base):
    """Service health monitoring logs."""
    __tablename__ = "service_health_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False)
    component = Column(String(100))
    
    # Health Status
    status = Column(String(50), nullable=False)
    health_score = Column(Float)  # 0.0 to 1.0
    
    # Metrics
    response_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    cpu_usage_percent = Column(Float)
    active_connections = Column(Integer)
    
    # Details
    details = Column(JSON, default=lambda: {})
    error_message = Column(Text)
    
    checked_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_service_health_logs_service', 'service_name'),
        Index('idx_service_health_logs_status', 'status'),
        Index('idx_service_health_logs_checked_at', 'checked_at'),
        Index('idx_service_health_logs_component', 'component'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "service_name": self.service_name,
            "component": self.component,
            "status": self.status,
            "health_score": self.health_score,
            "response_time_ms": self.response_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "active_connections": self.active_connections,
            "details": self.details,
            "error_message": self.error_message,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
        }


class PerformanceMonitoring(Base):
    """Performance monitoring across services."""
    __tablename__ = "performance_monitoring"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service = Column(String(100), nullable=False)
    operation = Column(String(100), nullable=False)
    
    # Performance Metrics
    execution_time_ms = Column(Integer, nullable=False)
    memory_used_mb = Column(Float)
    cpu_usage_percent = Column(Float)
    
    # Context
    user_id = Column(String(255))
    session_id = Column(String(255))
    request_size_bytes = Column(Integer)
    response_size_bytes = Column(Integer)
    
    # Metadata
    metadata = Column(JSON, default=lambda: {})
    
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_performance_monitoring_service_operation', 'service', 'operation'),
        Index('idx_performance_monitoring_recorded_at', 'recorded_at'),
        Index('idx_performance_monitoring_execution_time', 'execution_time_ms'),
        Index('idx_performance_monitoring_user', 'user_id'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "service": self.service,
            "operation": self.operation,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_size_bytes": self.request_size_bytes,
            "response_size_bytes": self.response_size_bytes,
            "metadata": self.metadata,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
        }


class ServiceConfiguration(Base):
    """Configuration for integrated services."""
    __tablename__ = "service_configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_type = Column(String(50), nullable=False)
    service_name = Column(String(100), nullable=False)
    
    # Configuration
    config = Column(JSON, nullable=False)
    credentials = Column(JSON)  # Encrypted
    
    # Status
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    
    # Metadata
    description = Column(Text)
    version = Column(String(50))
    environment = Column(String(50))  # development, staging, production
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(String(255))
    
    __table_args__ = (
        Index('idx_service_configurations_type', 'service_type'),
        Index('idx_service_configurations_name', 'service_name'),
        Index('idx_service_configurations_active', 'is_active'),
        Index('idx_service_configurations_default', 'is_default'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "service_type": self.service_type,
            "service_name": self.service_name,
            "config": self.config,
            # Don't include credentials in API responses
            "is_active": self.is_active,
            "is_default": self.is_default,
            "description": self.description,
            "version": self.version,
            "environment": self.environment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
        }


class ServiceIntegrationLog(Base):
    """Logs for service integration events."""
    __tablename__ = "service_integration_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False)
    operation_type = Column(String(100), nullable=False)
    
    # Event Details
    event_data = Column(JSON, nullable=False)
    request_data = Column(JSON)
    response_data = Column(JSON)
    
    # Status
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    
    # Performance
    execution_time_ms = Column(Integer)
    
    # Context
    user_id = Column(String(255))
    session_id = Column(String(255))
    correlation_id = Column(String(100))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_service_integration_logs_service', 'service_name'),
        Index('idx_service_integration_logs_operation', 'operation_type'),
        Index('idx_service_integration_logs_success', 'success'),
        Index('idx_service_integration_logs_created_at', 'created_at'),
        Index('idx_service_integration_logs_correlation', 'correlation_id'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "service_name": self.service_name,
            "operation_type": self.operation_type,
            "event_data": self.event_data,
            "request_data": self.request_data,
            "response_data": self.response_data,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ServiceDependency(Base):
    """Dependencies between services."""
    __tablename__ = "service_dependencies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_service = Column(String(100), nullable=False)
    target_service = Column(String(100), nullable=False)
    dependency_type = Column(String(50), nullable=False)  # 'required', 'optional', 'fallback'
    
    # Configuration
    config = Column(JSON, default=lambda: {})
    
    # Status
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_service_dependencies_source', 'source_service'),
        Index('idx_service_dependencies_target', 'target_service'),
        Index('idx_service_dependencies_type', 'dependency_type'),
        Index('idx_service_dependencies_active', 'is_active'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "source_service": self.source_service,
            "target_service": self.target_service,
            "dependency_type": self.dependency_type,
            "config": self.config,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Pydantic Models for API

class ZepSessionRequest(BaseModel):
    """Schema for Zep session creation requests."""
    user_id: str = Field(..., min_length=1)
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_type: str = Field(default="perpetual")
    
    @field_validator('user_id')
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        return v.strip()


class ServiceHealthCheckRequest(BaseModel):
    """Schema for service health check requests."""
    service_name: str = Field(..., min_length=1)
    component: Optional[str] = None
    include_metrics: bool = Field(default=True)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class ServiceHealthResponse(BaseModel):
    """Schema for service health responses."""
    service_name: str
    component: Optional[str]
    status: HealthStatus
    health_score: Optional[float]
    response_time_ms: Optional[float]
    details: Dict[str, Any]
    checked_at: datetime


class PerformanceMetricsRequest(BaseModel):
    """Schema for performance metrics requests."""
    service: str = Field(..., min_length=1)
    operation: str = Field(..., min_length=1)
    start_time: datetime
    end_time: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class PerformanceMetricsResponse(BaseModel):
    """Schema for performance metrics responses."""
    service: str
    operation: str
    total_operations: int
    avg_execution_time_ms: float
    min_execution_time_ms: int
    max_execution_time_ms: int
    p95_execution_time_ms: float
    p99_execution_time_ms: float
    success_rate: float
    total_data_processed_mb: Optional[float]


class ServiceConfigurationCreate(BaseModel):
    """Schema for creating service configurations."""
    service_type: ServiceType
    service_name: str = Field(..., min_length=1, max_length=100)
    config: Dict[str, Any] = Field(..., description="Service configuration")
    credentials: Optional[Dict[str, Any]] = Field(None, description="Service credentials (will be encrypted)")
    description: Optional[str] = None
    version: Optional[str] = None
    environment: str = Field(default="development")
    is_default: bool = Field(default=False)
    
    @field_validator('service_name')
    def validate_service_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Service name cannot be empty')
        return v.strip()


class ServiceConfigurationUpdate(BaseModel):
    """Schema for updating service configurations."""
    config: Optional[Dict[str, Any]] = None
    credentials: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    version: Optional[str] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None


class ServiceIntegrationEvent(BaseModel):
    """Schema for service integration events."""
    service_name: str = Field(..., min_length=1)
    operation_type: str = Field(..., min_length=1)
    event_data: Dict[str, Any] = Field(..., description="Event-specific data")
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    success: bool = Field(..., description="Whether operation was successful")
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None


class ServiceDependencyCreate(BaseModel):
    """Schema for creating service dependencies."""
    source_service: str = Field(..., min_length=1)
    target_service: str = Field(..., min_length=1)
    dependency_type: str = Field(..., description="Type of dependency: required, optional, fallback")
    config: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('dependency_type')
    def validate_dependency_type(cls, v):
        valid_types = ['required', 'optional', 'fallback']
        if v not in valid_types:
            raise ValueError(f'Dependency type must be one of: {valid_types}')
        return v


class ServiceStatus(BaseModel):
    """Schema for overall service status."""
    service_name: str
    overall_status: HealthStatus
    components: List[ServiceHealthResponse]
    dependencies: List[Dict[str, Any]]
    performance_summary: Dict[str, Any]
    last_check: datetime


class ServiceAnalytics(BaseModel):
    """Schema for service analytics data."""
    total_services: int
    healthy_services: int
    degraded_services: int
    unhealthy_services: int
    total_operations: int
    avg_response_time: float
    success_rate: float
    top_performing_services: List[Dict[str, str]]
    bottleneck_services: List[Dict[str, str]]
    integration_health: Dict[str, Any]