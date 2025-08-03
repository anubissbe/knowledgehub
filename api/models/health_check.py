"""
Health Check Models

SQLAlchemy models for storing health monitoring data, service status,
and performance metrics in the database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from datetime import datetime, timezone
import uuid
import enum

from .base import Base

class ServiceStatus(str, enum.Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"
    UNKNOWN = "unknown"

class ServiceType(str, enum.Enum):
    """Service type enumeration"""
    DATABASE = "database"
    REDIS = "redis"
    WEAVIATE = "weaviate"
    NEO4J = "neo4j"
    TIMESCALE = "timescale"
    MINIO = "minio"
    API = "api"
    WEBSOCKET = "websocket"
    AI_SERVICE = "ai_service"
    MCP_SERVER = "mcp_server"
    EXTERNAL_API = "external_api"

class AlertSeverity(str, enum.Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class HealthCheck(Base):
    """Health check records for services"""
    __tablename__ = "health_checks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False, index=True)
    service_type = Column(Enum(ServiceType), nullable=False)
    status = Column(Enum(ServiceStatus), nullable=False, index=True)
    response_time_ms = Column(Float, nullable=False)
    error_message = Column(Text)
    metrics = Column(JSON)  # Store health metrics as JSON
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    uptime_percentage = Column(Float, default=100.0)
    consecutive_failures = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<HealthCheck(service={self.service_name}, status={self.status}, time={self.timestamp})>"

class ServiceMetric(Base):
    """Individual service metrics"""
    __tablename__ = "service_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    threshold_warning = Column(Float)
    threshold_critical = Column(Float)
    description = Column(Text)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    
    def __repr__(self):
        return f"<ServiceMetric(service={self.service_name}, metric={self.metric_name}, value={self.metric_value})>"

class SystemAlert(Base):
    """System alerts and notifications"""
    __tablename__ = "system_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(String(200), nullable=False, unique=True, index=True)
    service_name = Column(String(100), index=True)
    alert_type = Column(String(100), nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False, index=True)
    message = Column(Text, nullable=False)
    details = Column(JSON)  # Additional alert details
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime(timezone=True))
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
    count = Column(Integer, default=1)  # How many times this alert fired
    first_occurrence = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_occurrence = Column(DateTime(timezone=True), nullable=False, default=func.now())
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<SystemAlert(id={self.alert_id}, severity={self.severity}, resolved={self.resolved})>"

class PerformanceMetric(Base):
    """Performance metrics for system monitoring"""
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_category = Column(String(100), nullable=False, index=True)  # cpu, memory, disk, network, etc.
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    host_name = Column(String(100), index=True)
    service_name = Column(String(100), index=True)
    threshold_warning = Column(Float)
    threshold_critical = Column(Float)
    tags = Column(JSON)  # Additional metadata tags
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    
    def __repr__(self):
        return f"<PerformanceMetric(category={self.metric_category}, name={self.metric_name}, value={self.metric_value})>"

class UptimeRecord(Base):
    """Uptime tracking for services"""
    __tablename__ = "uptime_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    total_checks = Column(Integer, nullable=False, default=0)
    successful_checks = Column(Integer, nullable=False, default=0)
    failed_checks = Column(Integer, nullable=False, default=0)
    uptime_percentage = Column(Float, nullable=False, default=100.0)
    average_response_time = Column(Float, default=0.0)
    max_response_time = Column(Float, default=0.0)
    min_response_time = Column(Float, default=0.0)
    downtime_duration = Column(Integer, default=0)  # Total downtime in seconds
    failure_count = Column(Integer, default=0)  # Number of failure incidents
    mttr_seconds = Column(Float, default=0.0)  # Mean Time To Recovery
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    def __repr__(self):
        return f"<UptimeRecord(service={self.service_name}, date={self.date}, uptime={self.uptime_percentage}%)>"

class HealthDashboard(Base):
    """Health dashboard configuration and views"""
    __tablename__ = "health_dashboards"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dashboard_name = Column(String(200), nullable=False)
    user_id = Column(String(100), index=True)
    is_public = Column(Boolean, default=False)
    configuration = Column(JSON, nullable=False)  # Dashboard layout and widgets
    services_monitored = Column(JSON)  # List of services to monitor
    refresh_interval = Column(Integer, default=30)  # Refresh interval in seconds
    alert_thresholds = Column(JSON)  # Custom alert thresholds
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    last_accessed = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<HealthDashboard(name={self.dashboard_name}, user={self.user_id})>"

class ServiceDependency(Base):
    """Service dependency mapping for health monitoring"""
    __tablename__ = "service_dependencies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False, index=True)
    depends_on_service = Column(String(100), nullable=False, index=True)
    dependency_type = Column(String(50), nullable=False)  # critical, optional, performance
    weight = Column(Float, default=1.0)  # Impact weight on parent service
    timeout_seconds = Column(Integer, default=30)
    retry_attempts = Column(Integer, default=3)
    circuit_breaker_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    def __repr__(self):
        return f"<ServiceDependency(service={self.service_name}, depends_on={self.depends_on_service})>"

class HealthReport(Base):
    """Generated health reports"""
    __tablename__ = "health_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_type = Column(String(100), nullable=False, index=True)  # daily, weekly, monthly, incident
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False, index=True)
    overall_uptime = Column(Float, nullable=False)
    services_analyzed = Column(Integer, nullable=False)
    total_alerts = Column(Integer, default=0)
    critical_alerts = Column(Integer, default=0)
    incidents_count = Column(Integer, default=0)
    report_data = Column(JSON, nullable=False)  # Detailed report data
    summary = Column(Text)
    recommendations = Column(JSON)  # Improvement recommendations
    generated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    generated_by = Column(String(100))  # system or user_id
    
    def __repr__(self):
        return f"<HealthReport(type={self.report_type}, period={self.period_start} to {self.period_end})>"