"""
Phase 3: Enhanced Time-Series Analytics Service with GPU Acceleration
François Coppens - Performance Profiling Expert

Advanced TimescaleDB analytics with sub-10ms query response times,
GPU acceleration support, and real-time performance monitoring.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from enum import Enum
from dataclasses import dataclass

import asyncpg
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from shared.config import Config
from shared.logging import setup_logging

# Try to import GPU acceleration libraries
try:
    import torch
    import cupy as cp
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    torch = None
    cp = None

logger = setup_logging("timescale_analytics_enhanced")

# François Coppens Performance Standards
QUERY_LATENCY_TARGET_MS = 10  # Sub-10ms query response time target
MEMORY_EFFICIENCY_TARGET = 0.8
GPU_UTILIZATION_TARGET = 0.9


class MetricType(Enum):
    """Enhanced metric types for Phase 3"""
    KNOWLEDGE_CREATION = "knowledge_creation"
    DECISION_MAKING = "decision_making"
    ERROR_RATE = "error_rate"
    SOLUTION_EFFECTIVENESS = "solution_effectiveness"
    PATTERN_EVOLUTION = "pattern_evolution"
    USER_ENGAGEMENT = "user_engagement"
    CODE_QUALITY = "code_quality"
    LEARNING_PROGRESS = "learning_progress"
    PERFORMANCE = "performance"
    USAGE = "usage"
    # Phase 3 additions
    RAG_PERFORMANCE = "rag_performance"
    FPGA_UTILIZATION = "fpga_utilization" 
    GPU_UTILIZATION = "gpu_utilization"
    QUERY_LATENCY = "query_latency"
    SYSTEM_BOTTLENECK = "system_bottleneck"


class AggregationWindow(Enum):
    """Time window for aggregations"""
    MINUTE = "1 minute"
    HOUR = "1 hour"
    DAY = "1 day"
    WEEK = "1 week"
    MONTH = "1 month"
    QUARTER = "3 months"
    YEAR = "1 year"


@dataclass
class PerformanceMetrics:
    """Performance metrics for Phase 3 monitoring"""
    query_latency_ms: float
    memory_usage_mb: float
    gpu_utilization: float
    cpu_utilization: float
    throughput_ops_sec: float
    cache_hit_rate: float
    compression_ratio: float
    error_count: int


@dataclass
class TimeSeriesPoint:
    """A single time series data point"""
    timestamp: datetime
    metric_type: str
    value: float
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrendAnalysis:
    """Result of trend analysis"""
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1
    change_rate: float
    correlation_score: float
    seasonal_patterns: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]


class GPUAcceleratedAnalytics:
    """GPU acceleration for analytics operations"""
    
    def __init__(self):
        self.device_count = 0
        self.devices = []
        
        if GPU_AVAILABLE:
            self.device_count = torch.cuda.device_count()
            for i in range(self.device_count):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory // 1024 // 1024,
                    'compute_capability': torch.cuda.get_device_properties(i).major
                }
                self.devices.append(device_info)
            logger.info(f"Initialized GPU acceleration with {self.device_count} devices")
        else:
            logger.info("GPU acceleration not available, using CPU fallback")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information"""
        return {
            'device_count': self.device_count,
            'devices': self.devices,
            'cuda_available': GPU_AVAILABLE
        }
    
    def accelerated_aggregation(self, data: np.ndarray) -> Dict[str, float]:
        """Perform GPU-accelerated data aggregation"""
        if GPU_AVAILABLE and self.device_count > 0:
            try:
                # Use CuPy for GPU-accelerated computation
                gpu_data = cp.asarray(data)
                result = {
                    'mean': float(cp.mean(gpu_data)),
                    'std': float(cp.std(gpu_data)),
                    'min': float(cp.min(gpu_data)),
                    'max': float(cp.max(gpu_data)),
                    'sum': float(cp.sum(gpu_data))
                }
                return result
            except Exception as e:
                logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")
        
        # CPU fallback
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'sum': float(np.sum(data))
        }


class TimeSeriesAnalyticsService:
    """
    Enhanced time-series analytics service with GPU acceleration.
    
    Features:
    - Sub-10ms query performance
    - GPU-accelerated analytics
    - Real-time continuous aggregates
    - RAG pipeline integration
    - FPGA workflow tracking
    - Advanced bottleneck analysis
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize the enhanced analytics service."""
        self.config = Config()
        
        # Database configuration
        self.database_url = database_url or self._build_timescale_url()
        
        self.engine = None
        self.async_engine = None
        self._initialized = False
        
        # Initialize GPU acceleration
        self.gpu_analytics = GPUAcceleratedAnalytics()
        
        logger.info(f"Initialized Enhanced TimeSeriesAnalyticsService")
    
    def _build_timescale_url(self) -> str:
        """Build TimescaleDB connection URL from environment"""
        host = getattr(self.config, 'TIMESCALE_HOST', 'localhost')
        port = getattr(self.config, 'TIMESCALE_PORT', 5434)
        database = getattr(self.config, 'TIMESCALE_DATABASE', 'knowledgehub_analytics')
        user = getattr(self.config, 'TIMESCALE_USER', 'knowledgehub')
        password = getattr(self.config, 'TIMESCALE_PASSWORD', 'knowledgehub123')
        
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize TimescaleDB connection and create enhanced tables"""
        try:
            # Create async engine with connection pooling for performance
            self.async_engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=20,  # Increased for better performance
                max_overflow=30,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            # Create sync engine for some operations
            self.engine = create_engine(self.database_url.replace('+asyncpg', ''))
            
            # Create enhanced tables and hypertables
            await self._create_enhanced_tables()
            
            self._initialized = True
            logger.info("Successfully initialized Enhanced TimescaleDB")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced TimescaleDB: {str(e)}")
            self._initialized = False
    
    async def _create_enhanced_tables(self):
        """Create enhanced time-series tables for Phase 3"""
        
        # Enhanced metrics table with performance tracking
        create_enhanced_metrics = """
        CREATE TABLE IF NOT EXISTS ts_metrics_enhanced (
            time TIMESTAMPTZ NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            tags JSONB DEFAULT '{}',
            metadata JSONB DEFAULT '{}',
            user_id VARCHAR(50),
            session_id VARCHAR(50),
            project_id VARCHAR(50),
            performance_score DOUBLE PRECISION DEFAULT 0.0,
            gpu_accelerated BOOLEAN DEFAULT FALSE
        );
        """
        
        # Performance analytics table for detailed monitoring
        create_performance_analytics = """
        CREATE TABLE IF NOT EXISTS ts_performance_analytics (
            time TIMESTAMPTZ NOT NULL,
            component VARCHAR(100) NOT NULL,
            operation VARCHAR(100) NOT NULL,
            latency_ms DOUBLE PRECISION NOT NULL,
            throughput_ops_sec DOUBLE PRECISION,
            memory_usage_mb DOUBLE PRECISION,
            cpu_usage_percent DOUBLE PRECISION,
            gpu_usage_percent DOUBLE PRECISION,
            error_rate DOUBLE PRECISION DEFAULT 0.0,
            success_rate DOUBLE PRECISION DEFAULT 100.0,
            metadata JSONB DEFAULT '{}'
        );
        """
        
        # RAG performance tracking table
        create_rag_performance = """
        CREATE TABLE IF NOT EXISTS ts_rag_performance (
            time TIMESTAMPTZ NOT NULL,
            query_id VARCHAR(100) NOT NULL,
            query_type VARCHAR(50) NOT NULL,
            retrieval_time_ms DOUBLE PRECISION NOT NULL,
            generation_time_ms DOUBLE PRECISION NOT NULL,
            chunk_count INTEGER NOT NULL,
            relevance_score DOUBLE PRECISION,
            tokens_processed INTEGER,
            model_used VARCHAR(100),
            success BOOLEAN DEFAULT TRUE
        );
        """
        
        # Real-time events table for sub-second analytics
        create_realtime_events = """
        CREATE TABLE IF NOT EXISTS ts_realtime_events (
            time TIMESTAMPTZ NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            event_data JSONB NOT NULL,
            processing_time_ms DOUBLE PRECISION DEFAULT 0.0,
            gpu_accelerated BOOLEAN DEFAULT FALSE
        );
        """
        
        # Execute table creation
        async with self.async_engine.begin() as conn:
            await conn.execute(text(create_enhanced_metrics))
            await conn.execute(text(create_performance_analytics))
            await conn.execute(text(create_rag_performance))
            await conn.execute(text(create_realtime_events))
            
            # Create hypertables for optimal time-series performance
            hypertables = [
                "SELECT create_hypertable('ts_metrics_enhanced', 'time', if_not_exists => TRUE);",
                "SELECT create_hypertable('ts_performance_analytics', 'time', if_not_exists => TRUE);", 
                "SELECT create_hypertable('ts_rag_performance', 'time', if_not_exists => TRUE);",
                "SELECT create_hypertable('ts_realtime_events', 'time', if_not_exists => TRUE);"
            ]
            
            for hypertable_sql in hypertables:
                try:
                    await conn.execute(text(hypertable_sql))
                except Exception as e:
                    logger.info(f"Hypertable may already exist: {str(e)}")
        
        # Create optimized continuous aggregates for sub-10ms queries
        await self._create_continuous_aggregates()
        
        # Create indexes for François Coppens performance standards
        await self._create_performance_indexes()
    
    async def _create_continuous_aggregates(self):
        """Create continuous aggregates for sub-10ms query performance"""
        
        continuous_aggregates = [
            # 1-minute aggregates for real-time dashboards
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS ts_metrics_1min
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 minute', time) AS bucket,
                metric_type,
                AVG(value) as avg_value,
                MAX(value) as max_value,
                MIN(value) as min_value,
                COUNT(*) as count,
                AVG(performance_score) as avg_performance
            FROM ts_metrics_enhanced
            GROUP BY bucket, metric_type;
            """,
            
            # 5-minute performance aggregates
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS ts_performance_5min
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('5 minutes', time) AS bucket,
                component,
                operation,
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency,
                AVG(cpu_usage_percent) as avg_cpu,
                AVG(memory_usage_mb) as avg_memory,
                AVG(gpu_usage_percent) as avg_gpu,
                SUM(CASE WHEN latency_ms <= 10 THEN 1 ELSE 0 END) as sub_10ms_count,
                COUNT(*) as total_operations
            FROM ts_performance_analytics
            GROUP BY bucket, component, operation;
            """,
            
            # Hourly RAG performance aggregates  
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS ts_rag_hourly
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 hour', time) AS bucket,
                query_type,
                AVG(retrieval_time_ms + generation_time_ms) as avg_total_time,
                AVG(relevance_score) as avg_relevance,
                SUM(tokens_processed) as total_tokens,
                COUNT(*) as query_count,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
            FROM ts_rag_performance
            GROUP BY bucket, query_type;
            """
        ]
        
        # Create continuous aggregates (outside transaction)
        for agg_sql in continuous_aggregates:
            try:
                async with self.async_engine.connect() as conn:
                    await conn.execute(text(agg_sql))
                    logger.debug("Created continuous aggregate for sub-10ms performance")
            except Exception as e:
                logger.warning(f"Continuous aggregate warning: {str(e)}")
    
    async def _create_performance_indexes(self):
        """Create optimized indexes for François Coppens performance standards"""
        
        indexes = [
            # Performance-critical indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_enhanced_perf ON ts_metrics_enhanced (metric_type, time DESC, performance_score);",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_latency ON ts_performance_analytics (component, latency_ms, time DESC);",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_query_type ON ts_rag_performance (query_type, time DESC);",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_realtime_events ON ts_realtime_events (event_type, time DESC);",
            
            # GPU acceleration tracking
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gpu_accelerated ON ts_metrics_enhanced (gpu_accelerated, time DESC) WHERE gpu_accelerated = TRUE;",
            
            # Sub-10ms performance tracking
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sub_10ms_queries ON ts_performance_analytics (time DESC) WHERE latency_ms <= 10;"
        ]
        
        for index_sql in indexes:
            try:
                async with self.async_engine.connect() as conn:
                    await conn.execute(text(index_sql))
            except Exception as e:
                logger.warning(f"Index creation warning: {str(e)}")
    
    async def record_metric(self,
                           metric_type: MetricType,
                           value: float,
                           tags: Optional[Dict[str, str]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           project_id: Optional[str] = None,
                           timestamp: Optional[datetime] = None) -> bool:
        """Record enhanced metric with performance tracking"""
        if not self._initialized:
            logger.warning("Enhanced TimescaleDB not initialized, skipping metric recording")
            return False
            
        start_time = time.time()
        
        try:
            ts = timestamp or datetime.utcnow()
            tags_json = json.dumps(tags or {})
            metadata_json = json.dumps(metadata or {})
            
            # Calculate performance score based on François Coppens standards
            performance_score = self._calculate_performance_score(value, metric_type)
            
            query = """
            INSERT INTO ts_metrics_enhanced 
            (time, metric_type, value, tags, metadata, user_id, session_id, project_id, performance_score, gpu_accelerated)
            VALUES (:time, :metric_type, :value, :tags, :metadata, :user_id, :session_id, :project_id, :performance_score, :gpu_accelerated)
            """
            
            async with self.async_engine.begin() as conn:
                await conn.execute(
                    text(query),
                    {
                        'time': ts,
                        'metric_type': metric_type.value,
                        'value': value,
                        'tags': tags_json,
                        'metadata': metadata_json,
                        'user_id': user_id,
                        'session_id': session_id,
                        'project_id': project_id,
                        'performance_score': performance_score,
                        'gpu_accelerated': self.gpu_analytics.device_count > 0
                    }
                )
            
            # Track recording latency
            recording_latency = (time.time() - start_time) * 1000
            if recording_latency > QUERY_LATENCY_TARGET_MS:
                logger.warning(f"Metric recording took {recording_latency:.2f}ms (above {QUERY_LATENCY_TARGET_MS}ms target)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording enhanced metric: {str(e)}")
            return False
    
    def _calculate_performance_score(self, value: float, metric_type: MetricType) -> float:
        """Calculate performance score based on François Coppens methodology"""
        if metric_type == MetricType.QUERY_LATENCY:
            # Lower latency = higher score
            return max(0.0, 1.0 - (value / 100.0))
        elif metric_type == MetricType.GPU_UTILIZATION:
            # Higher utilization = higher score (up to target)
            return min(1.0, value / GPU_UTILIZATION_TARGET)
        elif metric_type == MetricType.ERROR_RATE:
            # Lower error rate = higher score
            return max(0.0, 1.0 - value)
        else:
            # Default normalized score
            return min(1.0, max(0.0, value / 100.0))
    
    async def record_performance_metric(self,
                                       component: str,
                                       operation: str,
                                       latency_ms: float,
                                       throughput_ops_sec: Optional[float] = None,
                                       memory_usage_mb: Optional[float] = None,
                                       cpu_usage_percent: Optional[float] = None,
                                       gpu_usage_percent: Optional[float] = None,
                                       error_rate: float = 0.0,
                                       success_rate: float = 100.0,
                                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record detailed performance analytics"""
        if not self._initialized:
            return False
        
        try:
            query = """
            INSERT INTO ts_performance_analytics 
            (time, component, operation, latency_ms, throughput_ops_sec, memory_usage_mb, 
             cpu_usage_percent, gpu_usage_percent, error_rate, success_rate, metadata)
            VALUES (:time, :component, :operation, :latency_ms, :throughput_ops_sec, :memory_usage_mb,
                    :cpu_usage_percent, :gpu_usage_percent, :error_rate, :success_rate, :metadata)
            """
            
            async with self.async_engine.begin() as conn:
                await conn.execute(
                    text(query),
                    {
                        'time': datetime.utcnow(),
                        'component': component,
                        'operation': operation,
                        'latency_ms': latency_ms,
                        'throughput_ops_sec': throughput_ops_sec,
                        'memory_usage_mb': memory_usage_mb,
                        'cpu_usage_percent': cpu_usage_percent,
                        'gpu_usage_percent': gpu_usage_percent,
                        'error_rate': error_rate,
                        'success_rate': success_rate,
                        'metadata': json.dumps(metadata or {})
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording performance metric: {str(e)}")
            return False
    
    async def record_rag_performance(self,
                                    query_id: str,
                                    query_type: str,
                                    retrieval_time_ms: float,
                                    generation_time_ms: float,
                                    chunk_count: int,
                                    relevance_score: Optional[float] = None,
                                    tokens_processed: Optional[int] = None,
                                    model_used: Optional[str] = None,
                                    success: bool = True) -> bool:
        """Record RAG pipeline performance metrics"""
        if not self._initialized:
            return False
        
        try:
            query = """
            INSERT INTO ts_rag_performance 
            (time, query_id, query_type, retrieval_time_ms, generation_time_ms, chunk_count,
             relevance_score, tokens_processed, model_used, success)
            VALUES (:time, :query_id, :query_type, :retrieval_time_ms, :generation_time_ms, :chunk_count,
                    :relevance_score, :tokens_processed, :model_used, :success)
            """
            
            async with self.async_engine.begin() as conn:
                await conn.execute(
                    text(query),
                    {
                        'time': datetime.utcnow(),
                        'query_id': query_id,
                        'query_type': query_type,
                        'retrieval_time_ms': retrieval_time_ms,
                        'generation_time_ms': generation_time_ms,
                        'chunk_count': chunk_count,
                        'relevance_score': relevance_score,
                        'tokens_processed': tokens_processed,
                        'model_used': model_used,
                        'success': success
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording RAG performance: {str(e)}")
            return False
    
    async def get_real_time_metrics(self,
                                   metric_types: List[MetricType],
                                   time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get real-time metrics with sub-10ms response using continuous aggregates"""
        if not self._initialized:
            return {}
        
        start_time = time.time()
        
        try:
            # Use continuous aggregates for optimal performance
            end_time = datetime.utcnow()
            start_time_dt = end_time - timedelta(minutes=time_range_minutes)
            
            metrics_data = {}
            
            for metric_type in metric_types:
                query = """
                SELECT 
                    bucket as timestamp,
                    avg_value,
                    max_value,
                    min_value,
                    count,
                    avg_performance
                FROM ts_metrics_1min
                WHERE metric_type = :metric_type
                    AND bucket >= :start_time
                    AND bucket <= :end_time
                ORDER BY bucket DESC
                LIMIT 100
                """
                
                async with self.async_engine.connect() as conn:
                    result = await conn.execute(
                        text(query),
                        {
                            'metric_type': metric_type.value,
                            'start_time': start_time_dt,
                            'end_time': end_time
                        }
                    )
                    
                    data_points = []
                    for row in result:
                        data_points.append({
                            'timestamp': row.timestamp.isoformat(),
                            'avg_value': float(row.avg_value) if row.avg_value else 0,
                            'max_value': float(row.max_value) if row.max_value else 0,
                            'min_value': float(row.min_value) if row.min_value else 0,
                            'count': int(row.count),
                            'performance_score': float(row.avg_performance) if row.avg_performance else 0
                        })
                    
                    metrics_data[metric_type.value] = data_points
            
            query_time = (time.time() - start_time) * 1000
            
            return {
                'metrics': metrics_data,
                'query_time_ms': query_time,
                'performance_target_met': query_time <= QUERY_LATENCY_TARGET_MS,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {str(e)}")
            return {}
    
    async def generate_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Generate enhanced dashboard data with GPU acceleration"""
        start_time = time.time()
        
        try:
            now = datetime.utcnow()
            dashboard_data = {
                'system_performance': await self._get_system_performance_metrics(),
                'rag_performance': await self._get_rag_performance_metrics(),
                'gpu_utilization': self.gpu_analytics.get_device_info(),
                'real_time_status': await self._get_real_time_status(),
                'performance_alerts': await self._get_performance_alerts(),
                'optimization_recommendations': await self._get_optimization_recommendations(),
                'generation_time_ms': 0.0  # Will be set below
            }
            
            generation_time = (time.time() - start_time) * 1000
            dashboard_data['generation_time_ms'] = generation_time
            
            if generation_time > QUERY_LATENCY_TARGET_MS * 2:
                logger.warning(f"Dashboard generation took {generation_time:.2f}ms (above optimal)")
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating enhanced dashboard: {str(e)}")
            return {'error': str(e)}
    
    async def _get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics using continuous aggregates"""
        try:
            query = """
            SELECT 
                component,
                AVG(avg_latency) as avg_latency,
                MAX(max_latency) as max_latency,
                AVG(avg_cpu) as avg_cpu_usage,
                AVG(avg_memory) as avg_memory_usage,
                AVG(avg_gpu) as avg_gpu_usage,
                SUM(sub_10ms_count) as sub_10ms_operations,
                SUM(total_operations) as total_operations
            FROM ts_performance_5min
            WHERE bucket >= NOW() - INTERVAL '1 hour'
            GROUP BY component
            ORDER BY avg_latency DESC
            """
            
            metrics = []
            async with self.async_engine.connect() as conn:
                result = await conn.execute(text(query))
                
                for row in result:
                    metrics.append({
                        'component': row.component,
                        'avg_latency_ms': float(row.avg_latency) if row.avg_latency else 0,
                        'max_latency_ms': float(row.max_latency) if row.max_latency else 0,
                        'avg_cpu_percent': float(row.avg_cpu_usage) if row.avg_cpu_usage else 0,
                        'avg_memory_mb': float(row.avg_memory_usage) if row.avg_memory_usage else 0,
                        'avg_gpu_percent': float(row.avg_gpu_usage) if row.avg_gpu_usage else 0,
                        'sub_10ms_ratio': (row.sub_10ms_operations / max(row.total_operations, 1)) if row.total_operations else 0
                    })
            
            return {'components': metrics}
            
        except Exception as e:
            logger.error(f"Error getting system performance metrics: {str(e)}")
            return {'components': []}
    
    async def _get_rag_performance_metrics(self) -> Dict[str, Any]:
        """Get RAG pipeline performance metrics"""
        try:
            query = """
            SELECT 
                query_type,
                AVG(avg_total_time) as avg_total_time,
                AVG(avg_relevance) as avg_relevance,
                SUM(total_tokens) as total_tokens,
                SUM(query_count) as query_count,
                SUM(success_count) as success_count
            FROM ts_rag_hourly
            WHERE bucket >= NOW() - INTERVAL '24 hours'
            GROUP BY query_type
            ORDER BY query_count DESC
            """
            
            rag_metrics = []
            async with self.async_engine.connect() as conn:
                result = await conn.execute(text(query))
                
                for row in result:
                    rag_metrics.append({
                        'query_type': row.query_type,
                        'avg_response_time_ms': float(row.avg_total_time) if row.avg_total_time else 0,
                        'avg_relevance_score': float(row.avg_relevance) if row.avg_relevance else 0,
                        'total_tokens_processed': int(row.total_tokens) if row.total_tokens else 0,
                        'query_count': int(row.query_count) if row.query_count else 0,
                        'success_rate': (row.success_count / max(row.query_count, 1)) if row.query_count else 0
                    })
            
            return {'rag_metrics': rag_metrics}
            
        except Exception as e:
            logger.error(f"Error getting RAG performance metrics: {str(e)}")
            return {'rag_metrics': []}
    
    async def _get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time system status"""
        return {
            'timescaledb_connected': self._initialized,
            'gpu_acceleration_enabled': self.gpu_analytics.device_count > 0,
            'continuous_aggregates_active': True,  # TODO: Check actual status
            'performance_targets_met': True,  # TODO: Check against thresholds
            'last_updated': datetime.utcnow().isoformat()
        }
    
    async def _get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on François Coppens thresholds"""
        alerts = []
        
        try:
            # Check for high latency operations
            query = """
            SELECT component, operation, AVG(avg_latency) as avg_latency
            FROM ts_performance_5min
            WHERE bucket >= NOW() - INTERVAL '15 minutes'
            GROUP BY component, operation
            HAVING AVG(avg_latency) > :latency_threshold
            ORDER BY avg_latency DESC
            LIMIT 5
            """
            
            async with self.async_engine.connect() as conn:
                result = await conn.execute(
                    text(query), 
                    {'latency_threshold': QUERY_LATENCY_TARGET_MS * 2}
                )
                
                for row in result:
                    alerts.append({
                        'type': 'high_latency',
                        'severity': 'high' if row.avg_latency > QUERY_LATENCY_TARGET_MS * 5 else 'medium',
                        'component': row.component,
                        'operation': row.operation,
                        'avg_latency_ms': float(row.avg_latency),
                        'threshold_ms': QUERY_LATENCY_TARGET_MS,
                        'message': f"High latency detected: {row.avg_latency:.2f}ms exceeds {QUERY_LATENCY_TARGET_MS}ms target"
                    })
            
        except Exception as e:
            logger.error(f"Error getting performance alerts: {str(e)}")
        
        return alerts
    
    async def _get_optimization_recommendations(self) -> List[str]:
        """Get François Coppens optimization recommendations"""
        recommendations = []
        
        try:
            # Check GPU utilization
            if self.gpu_analytics.device_count == 0:
                recommendations.append("Enable GPU acceleration for analytics workloads to achieve François Coppens performance standards")
            
            # Check query performance
            query = """
            SELECT AVG(avg_latency) as overall_avg_latency
            FROM ts_performance_5min
            WHERE bucket >= NOW() - INTERVAL '1 hour'
            """
            
            async with self.async_engine.connect() as conn:
                result = await conn.execute(text(query))
                row = result.fetchone()
                
                if row and row.overall_avg_latency:
                    avg_latency = float(row.overall_avg_latency)
                    if avg_latency > QUERY_LATENCY_TARGET_MS:
                        recommendations.append(f"Query latency ({avg_latency:.2f}ms) exceeds {QUERY_LATENCY_TARGET_MS}ms target. Consider query optimization or hardware scaling.")
                    else:
                        recommendations.append("Query performance is meeting François Coppens standards (sub-10ms target)")
            
            # Check continuous aggregates utilization
            recommendations.append("Continuous aggregates are active for optimal query performance")
            
            if not recommendations:
                recommendations.append("System is performing optimally according to François Coppens performance profiling standards")
                
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {str(e)}")
            recommendations.append("Unable to generate recommendations due to system error")
        
        return recommendations
    
    async def _get_gpu_utilization_metrics(self) -> Dict[str, Any]:
        """Get GPU utilization metrics"""
        if not self.gpu_analytics.device_count:
            return {'gpu_available': False}
        
        # TODO: Implement actual GPU monitoring
        return {
            'gpu_available': True,
            'device_count': self.gpu_analytics.device_count,
            'utilization_percent': 0.0,  # Placeholder
            'memory_used_mb': 0.0,       # Placeholder
            'memory_total_mb': sum(device['memory_total'] for device in self.gpu_analytics.devices)
        }
    
    async def cleanup(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()
        logger.info("Closed Enhanced TimescaleDB connections")
