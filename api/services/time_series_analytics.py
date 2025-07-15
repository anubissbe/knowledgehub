"""
Time-Series Analytics Service using TimescaleDB.

This service provides time-series data storage and analysis for tracking
knowledge evolution, pattern trends, and temporal analytics.
"""

import logging
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
import plotly.graph_objects as go
import plotly.express as px

from ...shared.config import Config
from ...shared.logging import setup_logging

logger = setup_logging("time_series_analytics")


class MetricType(Enum):
    """Types of metrics to track"""
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


class TimeSeriesAnalyticsService:
    """
    Service for time-series analytics using TimescaleDB.
    
    Features:
    - Hypertable management
    - Time-series data ingestion
    - Temporal pattern analysis
    - Trend detection
    - Anomaly detection
    - Knowledge evolution tracking
    - Performance analytics
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the time-series analytics service.
        
        Args:
            database_url: TimescaleDB connection URL
        """
        self.config = Config()
        
        # Database configuration
        self.database_url = database_url or self.config.get(
            "TIMESCALE_URL",
            self.config.get("DATABASE_URL", "postgresql://user:pass@localhost/db")
        )
        
        # Convert to TimescaleDB URL if needed
        if "postgresql://" in self.database_url:
            self.timescale_url = self.database_url
        else:
            self.timescale_url = f"postgresql://{self.database_url}"
        
        self.engine = None
        self.async_engine = None
        self._initialized = False
        
        logger.info(f"Initialized TimeSeriesAnalyticsService")
    
    async def initialize(self):
        """Initialize TimescaleDB connection and create tables"""
        try:
            # Create async engine
            self.async_engine = create_async_engine(
                self.timescale_url,
                echo=False,
                pool_size=10,
                max_overflow=20
            )
            
            # Create sync engine for some operations
            self.engine = create_engine(self.timescale_url.replace('+asyncpg', ''))
            
            # Create tables and hypertables
            await self._create_tables()
            
            self._initialized = True
            logger.info("Successfully initialized TimescaleDB")
            
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create time-series tables and hypertables"""
        
        # Main metrics table
        create_metrics_table = """
        CREATE TABLE IF NOT EXISTS ts_metrics (
            time TIMESTAMPTZ NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            tags JSONB DEFAULT '{}',
            metadata JSONB DEFAULT '{}',
            user_id VARCHAR(50),
            session_id VARCHAR(50),
            project_id VARCHAR(50)
        );
        """
        
        # Create hypertable
        create_hypertable = """
        SELECT create_hypertable('ts_metrics', 'time', if_not_exists => TRUE);
        """
        
        # Knowledge evolution table
        create_evolution_table = """
        CREATE TABLE IF NOT EXISTS ts_knowledge_evolution (
            time TIMESTAMPTZ NOT NULL,
            entity_type VARCHAR(50) NOT NULL,
            entity_id VARCHAR(50) NOT NULL,
            change_type VARCHAR(50) NOT NULL,
            old_value JSONB,
            new_value JSONB,
            confidence DOUBLE PRECISION DEFAULT 0.0,
            impact_score DOUBLE PRECISION DEFAULT 0.0,
            user_id VARCHAR(50),
            session_id VARCHAR(50)
        );
        """
        
        create_evolution_hypertable = """
        SELECT create_hypertable('ts_knowledge_evolution', 'time', if_not_exists => TRUE);
        """
        
        # Pattern trends table
        create_patterns_table = """
        CREATE TABLE IF NOT EXISTS ts_pattern_trends (
            time TIMESTAMPTZ NOT NULL,
            pattern_type VARCHAR(50) NOT NULL,
            pattern_name VARCHAR(100) NOT NULL,
            occurrence_count INTEGER DEFAULT 1,
            effectiveness_score DOUBLE PRECISION DEFAULT 0.0,
            context JSONB DEFAULT '{}',
            project_id VARCHAR(50)
        );
        """
        
        create_patterns_hypertable = """
        SELECT create_hypertable('ts_pattern_trends', 'time', if_not_exists => TRUE);
        """
        
        # Performance metrics table
        create_performance_table = """
        CREATE TABLE IF NOT EXISTS ts_performance (
            time TIMESTAMPTZ NOT NULL,
            endpoint VARCHAR(100),
            response_time DOUBLE PRECISION,
            memory_usage DOUBLE PRECISION,
            cpu_usage DOUBLE PRECISION,
            error_count INTEGER DEFAULT 0,
            request_count INTEGER DEFAULT 1,
            user_count INTEGER DEFAULT 0
        );
        """
        
        create_performance_hypertable = """
        SELECT create_hypertable('ts_performance', 'time', if_not_exists => TRUE);
        """
        
        # Create indexes for better query performance
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_metrics_type_time ON ts_metrics (metric_type, time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_metrics_tags ON ts_metrics USING GIN (tags);",
            "CREATE INDEX IF NOT EXISTS idx_evolution_entity ON ts_knowledge_evolution (entity_type, entity_id, time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_patterns_type ON ts_pattern_trends (pattern_type, time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_performance_endpoint ON ts_performance (endpoint, time DESC);"
        ]
        
        # Create continuous aggregates for common queries
        continuous_aggregates = [
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS ts_metrics_hourly
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 hour', time) AS bucket,
                metric_type,
                AVG(value) as avg_value,
                MAX(value) as max_value,
                MIN(value) as min_value,
                COUNT(*) as count
            FROM ts_metrics
            GROUP BY bucket, metric_type;
            """,
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS ts_metrics_daily
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 day', time) AS bucket,
                metric_type,
                AVG(value) as avg_value,
                MAX(value) as max_value,
                MIN(value) as min_value,
                COUNT(*) as count,
                STDDEV(value) as stddev_value
            FROM ts_metrics
            GROUP BY bucket, metric_type;
            """
        ]
        
        async with self.async_engine.begin() as conn:
            # Create tables
            await conn.execute(text(create_metrics_table))
            await conn.execute(text(create_evolution_table))
            await conn.execute(text(create_patterns_table))
            await conn.execute(text(create_performance_table))
            
            # Create hypertables (ignore errors if already exist)
            try:
                await conn.execute(text(create_hypertable))
                await conn.execute(text(create_evolution_hypertable))
                await conn.execute(text(create_patterns_hypertable))
                await conn.execute(text(create_performance_hypertable))
            except Exception as e:
                logger.info(f"Hypertables may already exist: {str(e)}")
            
            # Create indexes
            for index_sql in create_indexes:
                try:
                    await conn.execute(text(index_sql))
                except Exception as e:
                    logger.warning(f"Index creation warning: {str(e)}")
            
            # Create continuous aggregates
            for agg_sql in continuous_aggregates:
                try:
                    await conn.execute(text(agg_sql))
                except Exception as e:
                    logger.warning(f"Continuous aggregate warning: {str(e)}")
    
    async def record_metric(self,
                           metric_type: MetricType,
                           value: float,
                           tags: Optional[Dict[str, str]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           project_id: Optional[str] = None,
                           timestamp: Optional[datetime] = None) -> bool:
        """
        Record a time-series metric.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            tags: Optional tags for grouping
            metadata: Optional metadata
            user_id: User ID
            session_id: Session ID
            project_id: Project ID
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Success status
        """
        try:
            ts = timestamp or datetime.utcnow()
            tags_json = json.dumps(tags or {})
            metadata_json = json.dumps(metadata or {})
            
            query = """
            INSERT INTO ts_metrics (time, metric_type, value, tags, metadata, user_id, session_id, project_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            async with self.async_engine.begin() as conn:
                await conn.execute(
                    text(query),
                    ts, metric_type.value, value, tags_json, metadata_json,
                    user_id, session_id, project_id
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording metric: {str(e)}")
            return False
    
    async def record_knowledge_evolution(self,
                                       entity_type: str,
                                       entity_id: str,
                                       change_type: str,
                                       old_value: Optional[Dict[str, Any]] = None,
                                       new_value: Optional[Dict[str, Any]] = None,
                                       confidence: float = 0.0,
                                       impact_score: float = 0.0,
                                       user_id: Optional[str] = None,
                                       session_id: Optional[str] = None) -> bool:
        """Record knowledge evolution event."""
        try:
            query = """
            INSERT INTO ts_knowledge_evolution 
            (time, entity_type, entity_id, change_type, old_value, new_value, 
             confidence, impact_score, user_id, session_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            
            async with self.async_engine.begin() as conn:
                await conn.execute(
                    text(query),
                    datetime.utcnow(), entity_type, entity_id, change_type,
                    json.dumps(old_value or {}), json.dumps(new_value or {}),
                    confidence, impact_score, user_id, session_id
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording evolution: {str(e)}")
            return False
    
    async def record_pattern_trend(self,
                                 pattern_type: str,
                                 pattern_name: str,
                                 occurrence_count: int = 1,
                                 effectiveness_score: float = 0.0,
                                 context: Optional[Dict[str, Any]] = None,
                                 project_id: Optional[str] = None) -> bool:
        """Record pattern trend data."""
        try:
            query = """
            INSERT INTO ts_pattern_trends 
            (time, pattern_type, pattern_name, occurrence_count, effectiveness_score, context, project_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            async with self.async_engine.begin() as conn:
                await conn.execute(
                    text(query),
                    datetime.utcnow(), pattern_type, pattern_name,
                    occurrence_count, effectiveness_score,
                    json.dumps(context or {}), project_id
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording pattern trend: {str(e)}")
            return False
    
    async def get_metric_trends(self,
                              metric_type: MetricType,
                              start_time: datetime,
                              end_time: Optional[datetime] = None,
                              window: AggregationWindow = AggregationWindow.HOUR,
                              tags_filter: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Get metric trends over time.
        
        Args:
            metric_type: Type of metric
            start_time: Start time
            end_time: End time (defaults to now)
            window: Aggregation window
            tags_filter: Optional tags filter
            
        Returns:
            List of trend data points
        """
        end_time = end_time or datetime.utcnow()
        
        # Build WHERE clause for tags
        tags_where = ""
        if tags_filter:
            conditions = []
            for key, value in tags_filter.items():
                conditions.append(f"tags->>'{key}' = '{value}'")
            tags_where = " AND " + " AND ".join(conditions)
        
        query = f"""
        SELECT 
            time_bucket('{window.value}', time) AS bucket,
            AVG(value) as avg_value,
            MAX(value) as max_value,
            MIN(value) as min_value,
            COUNT(*) as count,
            STDDEV(value) as stddev_value
        FROM ts_metrics
        WHERE metric_type = $1 
            AND time >= $2 
            AND time <= $3
            {tags_where}
        GROUP BY bucket
        ORDER BY bucket
        """
        
        trends = []
        async with self.async_engine.begin() as conn:
            result = await conn.execute(
                text(query),
                metric_type.value, start_time, end_time
            )
            
            for row in result:
                trends.append({
                    'timestamp': row.bucket,
                    'avg_value': float(row.avg_value) if row.avg_value else 0,
                    'max_value': float(row.max_value) if row.max_value else 0,
                    'min_value': float(row.min_value) if row.min_value else 0,
                    'count': int(row.count),
                    'stddev': float(row.stddev_value) if row.stddev_value else 0
                })
        
        return trends
    
    async def analyze_trends(self,
                           metric_type: MetricType,
                           start_time: datetime,
                           end_time: Optional[datetime] = None,
                           window: AggregationWindow = AggregationWindow.DAY) -> TrendAnalysis:
        """
        Analyze trends in time-series data.
        
        Args:
            metric_type: Type of metric to analyze
            start_time: Start time for analysis
            end_time: End time (defaults to now)
            window: Time window for aggregation
            
        Returns:
            Trend analysis results
        """
        trends = await self.get_metric_trends(metric_type, start_time, end_time, window)
        
        if len(trends) < 2:
            return TrendAnalysis(
                trend_direction="stable",
                trend_strength=0.0,
                change_rate=0.0,
                correlation_score=0.0,
                seasonal_patterns=[],
                anomalies=[]
            )
        
        # Convert to pandas for analysis
        df = pd.DataFrame(trends)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate trend direction and strength
        values = df['avg_value'].values
        x = np.arange(len(values))
        
        # Linear regression for trend
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        # Calculate correlation for trend strength
        correlation = np.corrcoef(x, values)[0, 1]
        trend_strength = abs(correlation)
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = "increasing"
        elif slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Calculate change rate (percentage change from first to last)
        if values[0] != 0:
            change_rate = ((values[-1] - values[0]) / values[0]) * 100
        else:
            change_rate = 0.0
        
        # Detect anomalies using statistical methods
        anomalies = self._detect_anomalies(df['avg_value'])
        
        # Analyze seasonal patterns
        seasonal_patterns = self._analyze_seasonality(df['avg_value'])
        
        return TrendAnalysis(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            change_rate=change_rate,
            correlation_score=correlation,
            seasonal_patterns=seasonal_patterns,
            anomalies=anomalies
        )
    
    def _detect_anomalies(self, series: pd.Series, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies using z-score method"""
        mean = series.mean()
        std = series.std()
        
        anomalies = []
        for idx, value in series.items():
            z_score = abs((value - mean) / std) if std > 0 else 0
            if z_score > threshold:
                anomalies.append({
                    'timestamp': idx.isoformat(),
                    'value': float(value),
                    'z_score': float(z_score),
                    'severity': 'high' if z_score > 3.0 else 'medium'
                })
        
        return anomalies
    
    def _analyze_seasonality(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Analyze seasonal patterns in the data"""
        # Simple seasonal analysis - could be enhanced with more sophisticated methods
        patterns = []
        
        if len(series) >= 7:  # Weekly pattern
            weekly_pattern = series.groupby(series.index.dayofweek).mean()
            patterns.append({
                'type': 'weekly',
                'pattern': weekly_pattern.to_dict(),
                'strength': weekly_pattern.std() / weekly_pattern.mean() if weekly_pattern.mean() > 0 else 0
            })
        
        if len(series) >= 24:  # Daily pattern (hourly data)
            daily_pattern = series.groupby(series.index.hour).mean()
            patterns.append({
                'type': 'daily',
                'pattern': daily_pattern.to_dict(),
                'strength': daily_pattern.std() / daily_pattern.mean() if daily_pattern.mean() > 0 else 0
            })
        
        return patterns
    
    async def get_knowledge_evolution_timeline(self,
                                             entity_type: Optional[str] = None,
                                             entity_id: Optional[str] = None,
                                             start_time: Optional[datetime] = None,
                                             end_time: Optional[datetime] = None,
                                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get knowledge evolution timeline."""
        
        where_conditions = []
        params = []
        param_count = 0
        
        if entity_type:
            param_count += 1
            where_conditions.append(f"entity_type = ${param_count}")
            params.append(entity_type)
        
        if entity_id:
            param_count += 1
            where_conditions.append(f"entity_id = ${param_count}")
            params.append(entity_id)
        
        if start_time:
            param_count += 1
            where_conditions.append(f"time >= ${param_count}")
            params.append(start_time)
        
        if end_time:
            param_count += 1
            where_conditions.append(f"time <= ${param_count}")
            params.append(end_time)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
        SELECT 
            time, entity_type, entity_id, change_type, 
            old_value, new_value, confidence, impact_score,
            user_id, session_id
        FROM ts_knowledge_evolution
        {where_clause}
        ORDER BY time DESC
        LIMIT {limit}
        """
        
        evolution = []
        async with self.async_engine.begin() as conn:
            result = await conn.execute(text(query), *params)
            
            for row in result:
                evolution.append({
                    'timestamp': row.time.isoformat(),
                    'entity_type': row.entity_type,
                    'entity_id': row.entity_id,
                    'change_type': row.change_type,
                    'old_value': json.loads(row.old_value) if row.old_value else {},
                    'new_value': json.loads(row.new_value) if row.new_value else {},
                    'confidence': float(row.confidence),
                    'impact_score': float(row.impact_score),
                    'user_id': row.user_id,
                    'session_id': row.session_id
                })
        
        return evolution
    
    async def get_pattern_effectiveness_trends(self,
                                             pattern_type: Optional[str] = None,
                                             start_time: Optional[datetime] = None,
                                             end_time: Optional[datetime] = None,
                                             window: AggregationWindow = AggregationWindow.DAY) -> Dict[str, List[Dict[str, Any]]]:
        """Get pattern effectiveness trends over time."""
        
        start_time = start_time or (datetime.utcnow() - timedelta(days=30))
        end_time = end_time or datetime.utcnow()
        
        where_clause = ""
        params = [start_time, end_time]
        if pattern_type:
            where_clause = "AND pattern_type = $3"
            params.append(pattern_type)
        
        query = f"""
        SELECT 
            time_bucket('{window.value}', time) AS bucket,
            pattern_type,
            pattern_name,
            AVG(effectiveness_score) as avg_effectiveness,
            SUM(occurrence_count) as total_occurrences
        FROM ts_pattern_trends
        WHERE time >= $1 AND time <= $2 {where_clause}
        GROUP BY bucket, pattern_type, pattern_name
        ORDER BY bucket, pattern_type, pattern_name
        """
        
        trends = {}
        async with self.async_engine.begin() as conn:
            result = await conn.execute(text(query), *params)
            
            for row in result:
                key = f"{row.pattern_type}:{row.pattern_name}"
                if key not in trends:
                    trends[key] = []
                
                trends[key].append({
                    'timestamp': row.bucket.isoformat(),
                    'avg_effectiveness': float(row.avg_effectiveness),
                    'total_occurrences': int(row.total_occurrences)
                })
        
        return trends
    
    async def generate_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for analytics dashboard."""
        
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        dashboard_data = {
            'overview': {},
            'trends': {},
            'patterns': {},
            'evolution': {},
            'performance': {}
        }
        
        # Overview metrics for the last 24 hours
        overview_query = """
        SELECT 
            metric_type,
            COUNT(*) as total_points,
            AVG(value) as avg_value,
            MAX(value) as max_value
        FROM ts_metrics
        WHERE time >= NOW() - INTERVAL '24 hours'
        GROUP BY metric_type
        """
        
        async with self.async_engine.begin() as conn:
            result = await conn.execute(text(overview_query))
            
            for row in result:
                dashboard_data['overview'][row.metric_type] = {
                    'total_points': int(row.total_points),
                    'avg_value': float(row.avg_value),
                    'max_value': float(row.max_value)
                }
        
        # Get trends for key metrics
        key_metrics = [
            MetricType.KNOWLEDGE_CREATION,
            MetricType.DECISION_MAKING,
            MetricType.ERROR_RATE,
            MetricType.USER_ENGAGEMENT
        ]
        
        for metric in key_metrics:
            trends = await self.get_metric_trends(
                metric, week_ago, now, AggregationWindow.HOUR
            )
            dashboard_data['trends'][metric.value] = trends[-24:] if len(trends) > 24 else trends
        
        # Pattern effectiveness
        pattern_trends = await self.get_pattern_effectiveness_trends(
            start_time=week_ago, end_time=now
        )
        dashboard_data['patterns'] = pattern_trends
        
        # Recent knowledge evolution
        evolution = await self.get_knowledge_evolution_timeline(
            start_time=week_ago, limit=50
        )
        dashboard_data['evolution'] = evolution
        
        return dashboard_data
    
    async def cleanup(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()
        logger.info("Closed TimescaleDB connections")