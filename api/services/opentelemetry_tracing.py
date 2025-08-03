"""
OpenTelemetry Distributed Tracing Service

Real production-grade distributed tracing implementation using OpenTelemetry
for comprehensive request tracking, performance analysis, and debugging.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Union
from contextlib import contextmanager
import uuid

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.propagators.jaeger import JaegerPropagator
    from opentelemetry.propagators.composite import CompositePropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)

class TraceContext:
    """Context holder for current trace information"""
    
    def __init__(self, trace_id: str, span_id: str, parent_span_id: Optional[str] = None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_time = time.time()
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

class OpenTelemetryTracing:
    """
    Real OpenTelemetry distributed tracing service providing:
    - Automatic instrumentation for FastAPI, SQLAlchemy, Redis, HTTP clients
    - Custom span creation and management
    - Performance analysis and bottleneck identification  
    - Cross-service trace correlation
    - Multiple exporter support (Jaeger, OTLP, Console)
    - Custom attribute and event tracking
    """
    
    def __init__(self, config=None):
        self.config = config or settings
        self.enabled = OPENTELEMETRY_AVAILABLE
        
        if not self.enabled:
            logger.warning("OpenTelemetry not available. Tracing disabled.")
            return
            
        # Service information
        self.service_name = "knowledgehub-api"
        self.service_version = "1.0.0"
        self.deployment_environment = getattr(self.config, 'APP_ENV', 'production')
        
        # Tracing configuration
        self.jaeger_endpoint = getattr(self.config, 'JAEGER_ENDPOINT', 'http://localhost:14268/api/traces')
        self.otlp_endpoint = getattr(self.config, 'OTLP_ENDPOINT', 'http://localhost:4317')
        self.sampling_ratio = getattr(self.config, 'TRACE_SAMPLING_RATIO', 1.0)
        
        # Initialize tracing
        self.tracer_provider = None
        self.tracer = None
        self.span_processor = None
        
        # Performance tracking
        self.slow_query_threshold = 0.1  # 100ms
        self.slow_request_threshold = 1.0  # 1 second
        self.critical_path_threshold = 0.05  # 50ms for memory operations
        
        # Custom spans tracking
        self.active_spans: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        
        self._initialize_tracing()
        logger.info("OpenTelemetry tracing service initialized")

    def _initialize_tracing(self) -> None:
        """Initialize OpenTelemetry tracing with multiple exporters"""
        if not self.enabled:
            return
            
        try:
            # Create resource with service information
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: self.service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.deployment_environment,
                ResourceAttributes.HOST_NAME: "knowledgehub-host",
                "service.instance.id": str(uuid.uuid4())
            })
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self.tracer_provider)
            
            # Add exporters
            self._setup_exporters()
            
            # Set up propagators for cross-service tracing
            set_global_textmap(
                CompositePropagator([
                    JaegerPropagator(),
                    B3MultiFormat()
                ])
            )
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Instrument libraries
            self._setup_auto_instrumentation()
            
            logger.info("OpenTelemetry tracing initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.enabled = False

    def _setup_exporters(self) -> None:
        """Setup multiple trace exporters"""
        try:
            # Jaeger exporter for UI visualization
            jaeger_exporter = JaegerExporter(
                endpoint=self.jaeger_endpoint,
                collector_endpoint="http://localhost:14268/api/traces"
            )
            
            jaeger_processor = BatchSpanProcessor(
                jaeger_exporter,
                max_queue_size=512,
                max_export_batch_size=32,
                export_timeout_millis=30000
            )
            self.tracer_provider.add_span_processor(jaeger_processor)
            
            # OTLP exporter for standardized telemetry
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint,
                    insecure=True
                )
                
                otlp_processor = BatchSpanProcessor(
                    otlp_exporter,
                    max_queue_size=512,
                    max_export_batch_size=32
                )
                self.tracer_provider.add_span_processor(otlp_processor)
                
            except Exception as e:
                logger.warning(f"OTLP exporter not available: {e}")
                
        except Exception as e:
            logger.error(f"Failed to setup exporters: {e}")

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries"""
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor.instrument()
            
            # Database instrumentation
            SQLAlchemyInstrumentor().instrument()
            AsyncPGInstrumentor().instrument()
            
            # Redis instrumentation
            RedisInstrumentor().instrument()
            
            # HTTP client instrumentation
            HTTPXClientInstrumentor().instrument()
            
            logger.info("Auto-instrumentation setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup auto-instrumentation: {e}")

    @contextmanager
    def start_span(
        self, 
        name: str, 
        kind: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create and manage a custom span"""
        if not self.enabled:
            yield None
            return
            
        span = self.tracer.start_span(name)
        
        try:
            # Set span kind
            if kind:
                span.set_attribute("span.kind", kind)
            
            # Add custom attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            # Track span
            span_id = str(span.get_span_context().span_id)
            self.active_spans[span_id] = {
                "span": span,
                "start_time": time.time(),
                "name": name
            }
            
            yield span
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
            
        finally:
            # Calculate duration and track performance
            if span_id in self.active_spans:
                duration = time.time() - self.active_spans[span_id]["start_time"]
                self._track_performance(name, duration)
                del self.active_spans[span_id]
            
            span.end()

    def trace_memory_operation(
        self, 
        operation: str, 
        user_id: str, 
        memory_type: str = "semantic"
    ):
        """Decorator for tracing memory operations with performance analysis"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                with self.start_span(
                    f"memory.{operation}",
                    kind="internal",
                    attributes={
                        "memory.operation": operation,
                        "memory.type": memory_type,
                        "user.id": user_id,
                        "memory.target_latency_ms": 50
                    }
                ) as span:
                    start_time = time.time()
                    
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Performance analysis
                        duration = time.time() - start_time
                        duration_ms = duration * 1000
                        
                        span.set_attribute("memory.duration_ms", duration_ms)
                        span.set_attribute("memory.performance_status", 
                                         "fast" if duration_ms < 50 else "slow")
                        
                        # Add performance event
                        if duration_ms > self.critical_path_threshold * 1000:
                            span.add_event("slow_memory_operation", {
                                "duration_ms": duration_ms,
                                "threshold_ms": 50,
                                "operation": operation
                            })
                        
                        # Track result metadata
                        if isinstance(result, dict):
                            if "memories" in result:
                                span.set_attribute("memory.results_count", len(result["memories"]))
                            if "score" in result:
                                span.set_attribute("memory.relevance_score", result["score"])
                        
                        return result
                        
                    except Exception as e:
                        span.add_event("memory_operation_error", {
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "operation": operation
                        })
                        raise
                        
            return wrapper
        return decorator

    def trace_ai_operation(
        self, 
        model_name: str, 
        operation_type: str,
        input_tokens: Optional[int] = None
    ):
        """Decorator for tracing AI operations with detailed performance metrics"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                with self.start_span(
                    f"ai.{operation_type}",
                    kind="internal",
                    attributes={
                        "ai.model": model_name,
                        "ai.operation": operation_type,
                        "ai.input_tokens": input_tokens or 0
                    }
                ) as span:
                    start_time = time.time()
                    
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Performance tracking
                        duration = time.time() - start_time
                        span.set_attribute("ai.duration_seconds", duration)
                        
                        # Track output metrics
                        if isinstance(result, dict):
                            if "embeddings" in result:
                                span.set_attribute("ai.output_dimensions", 
                                                 len(result["embeddings"]) if result["embeddings"] else 0)
                            if "tokens" in result:
                                span.set_attribute("ai.output_tokens", result["tokens"])
                            if "confidence" in result:
                                span.set_attribute("ai.confidence_score", result["confidence"])
                        
                        # Performance classification
                        performance_class = "fast" if duration < 1.0 else "normal" if duration < 5.0 else "slow"
                        span.set_attribute("ai.performance_class", performance_class)
                        
                        return result
                        
                    except Exception as e:
                        span.add_event("ai_operation_error", {
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "model": model_name,
                            "operation": operation_type
                        })
                        raise
                        
            return wrapper
        return decorator

    def trace_database_operation(
        self, 
        operation: str, 
        table: Optional[str] = None,
        query_type: str = "unknown"
    ):
        """Decorator for tracing database operations with query analysis"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                with self.start_span(
                    f"db.{operation}",
                    kind="database",
                    attributes={
                        "db.operation": operation,
                        "db.table": table or "unknown",
                        "db.type": "postgresql",
                        "db.query_type": query_type
                    }
                ) as span:
                    start_time = time.time()
                    
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Performance analysis
                        duration = time.time() - start_time
                        duration_ms = duration * 1000
                        
                        span.set_attribute("db.duration_ms", duration_ms)
                        
                        # Slow query detection
                        if duration > self.slow_query_threshold:
                            span.add_event("slow_query_detected", {
                                "duration_ms": duration_ms,
                                "threshold_ms": self.slow_query_threshold * 1000,
                                "table": table,
                                "operation": operation
                            })
                        
                        # Track result metadata
                        if hasattr(result, 'rowcount'):
                            span.set_attribute("db.rows_affected", result.rowcount)
                        
                        return result
                        
                    except Exception as e:
                        span.add_event("database_error", {
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "operation": operation,
                            "table": table
                        })
                        raise
                        
            return wrapper
        return decorator

    def trace_external_call(
        self, 
        service_name: str, 
        endpoint: str,
        method: str = "GET"
    ):
        """Decorator for tracing external service calls"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                with self.start_span(
                    f"external.{service_name}",
                    kind="client",
                    attributes={
                        "http.method": method,
                        "http.url": endpoint,
                        "external.service": service_name,
                        "component": "http"
                    }
                ) as span:
                    start_time = time.time()
                    
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Track response details
                        duration = time.time() - start_time
                        span.set_attribute("http.duration_ms", duration * 1000)
                        
                        if hasattr(result, 'status_code'):
                            span.set_attribute("http.status_code", result.status_code)
                            
                            # Mark error status
                            if result.status_code >= 400:
                                span.set_status(Status(StatusCode.ERROR, 
                                               f"HTTP {result.status_code}"))
                        
                        return result
                        
                    except Exception as e:
                        span.add_event("external_call_error", {
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "service": service_name,
                            "endpoint": endpoint
                        })
                        raise
                        
            return wrapper
        return decorator

    def add_span_event(self, event_name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the current active span"""
        if not self.enabled:
            return
            
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(event_name, attributes or {})

    def set_span_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current active span"""
        if not self.enabled:
            return
            
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute(key, value)

    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the current span"""
        if not self.enabled:
            return
            
        current_span = trace.get_current_span()
        if current_span:
            current_span.record_exception(exception)
            current_span.set_status(Status(StatusCode.ERROR, str(exception)))

    def get_trace_id(self) -> Optional[str]:
        """Get the current trace ID"""
        if not self.enabled:
            return None
            
        current_span = trace.get_current_span()
        if current_span:
            trace_id = current_span.get_span_context().trace_id
            return f"{trace_id:032x}"
        return None

    def get_span_id(self) -> Optional[str]:
        """Get the current span ID"""
        if not self.enabled:
            return None
            
        current_span = trace.get_current_span()
        if current_span:
            span_id = current_span.get_span_context().span_id
            return f"{span_id:016x}"
        return None

    def _track_performance(self, operation: str, duration: float) -> None:
        """Track performance metrics for operations"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        self.performance_metrics[operation].append(duration)
        
        # Keep only recent measurements (last 100)
        if len(self.performance_metrics[operation]) > 100:
            self.performance_metrics[operation] = self.performance_metrics[operation][-100:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all traced operations"""
        summary = {}
        
        for operation, durations in self.performance_metrics.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)
                
                # Calculate percentiles
                sorted_durations = sorted(durations)
                p50 = sorted_durations[len(sorted_durations) // 2]
                p95 = sorted_durations[int(len(sorted_durations) * 0.95)]
                
                summary[operation] = {
                    "count": len(durations),
                    "avg_duration_ms": avg_duration * 1000,
                    "min_duration_ms": min_duration * 1000,
                    "max_duration_ms": max_duration * 1000,
                    "p50_duration_ms": p50 * 1000,
                    "p95_duration_ms": p95 * 1000,
                    "slow_operations": len([d for d in durations if d > self.slow_request_threshold])
                }
        
        return summary

    def shutdown(self) -> None:
        """Shutdown tracing and flush remaining spans"""
        if not self.enabled or not self.tracer_provider:
            return
            
        try:
            # Force flush any pending spans
            if hasattr(self.tracer_provider, 'force_flush'):
                self.tracer_provider.force_flush(timeout_millis=30000)
            
            logger.info("OpenTelemetry tracing shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during tracing shutdown: {e}")

# Global tracing instance
otel_tracing = OpenTelemetryTracing()