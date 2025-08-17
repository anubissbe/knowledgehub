"""
Temporal Analytics Service for Real-Time Decision Making
Author: Pol Verbruggen - Adaptive Quantization & Model Pruning Expert

This module provides TimescaleDB integration for analyzing temporal patterns
in decision making and recommendations, supporting time-series analytics.
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class TemporalPattern(Enum):
    """Types of temporal patterns to analyze"""
    HOURLY = "hourly"
    DAILY = "daily" 
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"
    CUSTOM = "custom"

class MetricType(Enum):
    """Types of metrics to track"""
    DECISION_LATENCY = "decision_latency"
    DECISION_CONFIDENCE = "decision_confidence" 
    RECOMMENDATION_RELEVANCE = "recommendation_relevance"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_THROUGHPUT = "system_throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"

@dataclass
class TemporalMetric:
    """Structure for temporal metrics"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

@dataclass
class PatternAnalysis:
    """Results from temporal pattern analysis"""
    pattern_type: TemporalPattern
    metric_type: MetricType
    time_range: Tuple[datetime, datetime]
    statistics: Dict[str, float]
    trends: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    insights: List[str]

class TimescaleAnalyticsService:
    """
    Service for temporal analytics using TimescaleDB
    """
    
    def __init__(self, 
                 timescale_config: Dict[str, Any],
                 batch_size: int = 1000,
                 retention_days: int = 90):
        
        self.timescale_config = timescale_config
        self.batch_size = batch_size
        self.retention_days = retention_days
        
        # Connection pool
        self.pool = None
        
        # Batch processing
        self.metric_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        # Cache for frequently accessed patterns
        self.pattern_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.performance_metrics = {
            'metrics_ingested': 0,
            'queries_executed': 0,
            'avg_query_time': 0.0,
            'cache_hits': 0
        }
        
        logger.info("TimescaleDB analytics service initialized")
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.timescale_config.get('host', 'localhost'),
                port=self.timescale_config.get('port', 5434),
                database=self.timescale_config.get('database', 'timescale_analytics'),
                user=self.timescale_config.get('user', 'postgres'),
                password=self.timescale_config.get('password', 'postgres'),
                min_size=2,
                max_size=10
            )
            
            # Create tables and hypertables
            await self._create_tables()
            
            # Start background batch processor
            asyncio.create_task(self._batch_processor())
            
            logger.info("TimescaleDB connection and tables initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {e}")
            raise
    
    async def _create_tables(self):
        """Create TimescaleDB tables and hypertables"""
        
        async with self.pool.acquire() as conn:
            # Create main metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_metrics (
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_type TEXT NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    context JSONB,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for time-series optimization
            try:
                await conn.execute("""
                    SELECT create_hypertable('decision_metrics', 'timestamp', 
                                           chunk_time_interval => INTERVAL '1 hour',
                                           if_not_exists => TRUE);
                """)
            except Exception as e:
                if "already exists" not in str(e):
                    logger.warning(f"Could not create hypertable: {e}")
            
            # Create indexes for common queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_metrics_metric_type_time 
                ON decision_metrics (metric_type, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_metrics_user_time 
                ON decision_metrics (user_id, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_metrics_session_time 
                ON decision_metrics (session_id, timestamp DESC);
            """)
            
            # Create aggregated views for common analytics
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_metrics AS
                SELECT 
                    time_bucket('1 hour', timestamp) as hour,
                    metric_type,
                    AVG(value) as avg_value,
                    MAX(value) as max_value,
                    MIN(value) as min_value,
                    COUNT(*) as count,
                    STDDEV(value) as stddev
                FROM decision_metrics
                GROUP BY hour, metric_type;
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hourly_metrics_type_hour 
                ON hourly_metrics (metric_type, hour DESC);
            """)
            
            # Create retention policy
            try:
                await conn.execute(f"""
                    SELECT add_retention_policy('decision_metrics', INTERVAL '{self.retention_days} days');
                """)
            except Exception as e:
                logger.warning(f"Could not create retention policy: {e}")
            
            # Create continuous aggregate for real-time dashboards
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS realtime_dashboard AS
                SELECT 
                    time_bucket('5 minutes', timestamp) as time_bucket,
                    metric_type,
                    AVG(value) as avg_value,
                    COUNT(*) as sample_count
                FROM decision_metrics
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY time_bucket, metric_type;
            """)
    
    async def record_metric(self, metric: TemporalMetric):
        """Record a single temporal metric"""
        async with self.buffer_lock:
            self.metric_buffer.append(metric)
            
            # Trigger immediate flush if buffer is full
            if len(self.metric_buffer) >= self.batch_size:
                await self._flush_metrics()
    
    async def record_metrics_batch(self, metrics: List[TemporalMetric]):
        """Record multiple temporal metrics"""
        async with self.buffer_lock:
            self.metric_buffer.extend(metrics)
            
            # Trigger flush if buffer is getting full
            if len(self.metric_buffer) >= self.batch_size:
                await self._flush_metrics()
    
    async def _flush_metrics(self):
        """Flush buffered metrics to TimescaleDB"""
        if not self.metric_buffer:
            return
        
        try:
            async with self.pool.acquire() as conn:
                # Prepare batch insert
                records = []
                for metric in self.metric_buffer:
                    records.append((
                        metric.timestamp,
                        metric.metric_type.value,
                        metric.value,
                        metric.user_id,
                        metric.session_id,
                        json.dumps(metric.context) if metric.context else None,
                        json.dumps(metric.metadata) if metric.metadata else None
                    ))
                
                # Batch insert
                await conn.executemany("""
                    INSERT INTO decision_metrics 
                    (timestamp, metric_type, value, user_id, session_id, context, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, records)
                
                self.performance_metrics['metrics_ingested'] += len(records)
                logger.debug(f"Flushed {len(records)} metrics to TimescaleDB")
                
                # Clear buffer
                self.metric_buffer.clear()
                
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
            # Keep metrics in buffer for retry
    
    async def _batch_processor(self):
        """Background task for periodic metric flushing"""
        while True:
            try:
                await asyncio.sleep(10)  # Flush every 10 seconds
                
                async with self.buffer_lock:
                    if self.metric_buffer:
                        await self._flush_metrics()
                        
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
    
    async def analyze_temporal_patterns(self, 
                                      metric_type: MetricType,
                                      pattern_type: TemporalPattern,
                                      time_range: Optional[Tuple[datetime, datetime]] = None,
                                      user_id: Optional[str] = None) -> PatternAnalysis:
        """Analyze temporal patterns for specific metrics"""
        
        start_time = time.perf_counter()
        
        # Generate cache key
        cache_key = f"{metric_type.value}_{pattern_type.value}_{user_id}_{time_range}"
        
        # Check cache
        if cache_key in self.pattern_cache:
            cache_entry = self.pattern_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                self.performance_metrics['cache_hits'] += 1
                return cache_entry['analysis']
        
        try:
            # Set default time range if not provided
            if time_range is None:
                end_time = datetime.now()
                if pattern_type == TemporalPattern.HOURLY:
                    start_time_range = end_time - timedelta(hours=24)
                elif pattern_type == TemporalPattern.DAILY:
                    start_time_range = end_time - timedelta(days=30)
                elif pattern_type == TemporalPattern.WEEKLY:
                    start_time_range = end_time - timedelta(weeks=12)
                else:
                    start_time_range = end_time - timedelta(days=7)
                
                time_range = (start_time_range, end_time)
            
            # Execute temporal analysis query
            analysis_result = await self._execute_pattern_query(
                metric_type, pattern_type, time_range, user_id
            )
            
            # Compute statistics
            statistics_result = await self._compute_statistics(analysis_result)
            
            # Detect trends
            trends = await self._detect_trends(analysis_result, pattern_type)
            
            # Identify anomalies
            anomalies = await self._detect_anomalies(analysis_result)
            
            # Generate predictions
            predictions = await self._generate_predictions(analysis_result, pattern_type)
            
            # Generate insights
            insights = self._generate_insights(statistics_result, trends, anomalies)
            
            # Create analysis result
            analysis = PatternAnalysis(
                pattern_type=pattern_type,
                metric_type=metric_type,
                time_range=time_range,
                statistics=statistics_result,
                trends=trends,
                anomalies=anomalies,
                predictions=predictions,
                insights=insights
            )
            
            # Cache result
            self.pattern_cache[cache_key] = {
                'analysis': analysis,
                'timestamp': time.time()
            }
            
            # Update performance metrics
            query_time = (time.perf_counter() - start_time) * 1000
            self._update_query_metrics(query_time)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            raise
    
    async def _execute_pattern_query(self,
                                   metric_type: MetricType,
                                   pattern_type: TemporalPattern,
                                   time_range: Tuple[datetime, datetime],
                                   user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute query for temporal pattern analysis"""
        
        async with self.pool.acquire() as conn:
            # Build time bucket based on pattern type
            time_bucket_map = {
                TemporalPattern.HOURLY: '1 hour',
                TemporalPattern.DAILY: '1 day',
                TemporalPattern.WEEKLY: '1 week',
                TemporalPattern.MONTHLY: '1 month'
            }
            
            time_bucket = time_bucket_map.get(pattern_type, '1 hour')
            
            # Build query
            base_query = """
                SELECT 
                    time_bucket($1, timestamp) as time_bucket,
                    AVG(value) as avg_value,
                    MAX(value) as max_value,
                    MIN(value) as min_value,
                    COUNT(*) as sample_count,
                    STDDEV(value) as stddev,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY value) as p99
                FROM decision_metrics
                WHERE metric_type = $2 
                AND timestamp BETWEEN $3 AND $4
            """
            
            params = [time_bucket, metric_type.value, time_range[0], time_range[1]]
            
            if user_id:
                base_query += " AND user_id = $5"
                params.append(user_id)
            
            base_query += """
                GROUP BY time_bucket
                ORDER BY time_bucket ASC
            """
            
            rows = await conn.fetch(base_query, *params)
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                results.append({
                    'time_bucket': row['time_bucket'],
                    'avg_value': float(row['avg_value']) if row['avg_value'] else 0.0,
                    'max_value': float(row['max_value']) if row['max_value'] else 0.0,
                    'min_value': float(row['min_value']) if row['min_value'] else 0.0,
                    'sample_count': int(row['sample_count']),
                    'stddev': float(row['stddev']) if row['stddev'] else 0.0,
                    'median': float(row['median']) if row['median'] else 0.0,
                    'p95': float(row['p95']) if row['p95'] else 0.0,
                    'p99': float(row['p99']) if row['p99'] else 0.0
                })
            
            return results
    
    async def _compute_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute statistical summary of temporal data"""
        
        if not data:
            return {}
        
        avg_values = [d['avg_value'] for d in data]
        max_values = [d['max_value'] for d in data]
        min_values = [d['min_value'] for d in data]
        sample_counts = [d['sample_count'] for d in data]
        
        return {
            'overall_mean': statistics.mean(avg_values) if avg_values else 0.0,
            'overall_median': statistics.median(avg_values) if avg_values else 0.0,
            'overall_stddev': statistics.stdev(avg_values) if len(avg_values) > 1 else 0.0,
            'global_max': max(max_values) if max_values else 0.0,
            'global_min': min(min_values) if min_values else 0.0,
            'total_samples': sum(sample_counts),
            'time_periods': len(data),
            'coefficient_of_variation': statistics.stdev(avg_values) / statistics.mean(avg_values) if avg_values and statistics.mean(avg_values) > 0 else 0.0
        }
    
    async def _detect_trends(self, data: List[Dict[str, Any]], pattern_type: TemporalPattern) -> Dict[str, float]:
        """Detect trends in temporal data"""
        
        if len(data) < 3:
            return {'trend_direction': 0.0, 'trend_strength': 0.0}
        
        values = [d['avg_value'] for d in data]
        n = len(values)
        
        # Simple linear regression for trend detection
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Calculate correlation coefficient for trend strength
        if len(values) > 1:
            correlation = np.corrcoef(x, values)[0, 1] if not np.isnan(np.corrcoef(x, values)[0, 1]) else 0.0
        else:
            correlation = 0.0
        
        # Detect cyclical patterns
        cyclical_strength = 0.0
        if pattern_type in [TemporalPattern.DAILY, TemporalPattern.WEEKLY]:
            # Simple autocorrelation check
            if len(values) >= 12:  # Need sufficient data
                autocorr = np.corrcoef(values[:-6], values[6:])[0, 1] if len(values) >= 12 else 0.0
                cyclical_strength = abs(autocorr) if not np.isnan(autocorr) else 0.0
        
        return {
            'trend_direction': slope,
            'trend_strength': abs(correlation),
            'correlation': correlation,
            'cyclical_strength': cyclical_strength,
            'volatility': statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    async def _detect_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in temporal data using statistical methods"""
        
        if len(data) < 5:  # Need sufficient data for anomaly detection
            return []
        
        values = [d['avg_value'] for d in data]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        anomalies = []
        
        # Z-score based anomaly detection
        z_threshold = 2.5
        
        for i, item in enumerate(data):
            if std_val > 0:
                z_score = (item['avg_value'] - mean_val) / std_val
                
                if abs(z_score) > z_threshold:
                    anomalies.append({
                        'timestamp': item['time_bucket'],
                        'value': item['avg_value'],
                        'z_score': z_score,
                        'anomaly_type': 'statistical_outlier',
                        'severity': 'high' if abs(z_score) > 3.0 else 'medium',
                        'description': f"Value {item['avg_value']:.3f} deviates significantly from mean {mean_val:.3f}"
                    })
        
        # IQR-based anomaly detection
        if len(values) >= 4:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for item in data:
                if item['avg_value'] < lower_bound or item['avg_value'] > upper_bound:
                    # Check if not already detected by z-score
                    if not any(a['timestamp'] == item['time_bucket'] for a in anomalies):
                        anomalies.append({
                            'timestamp': item['time_bucket'],
                            'value': item['avg_value'],
                            'bound_violated': 'lower' if item['avg_value'] < lower_bound else 'upper',
                            'anomaly_type': 'iqr_outlier',
                            'severity': 'medium',
                            'description': f"Value {item['avg_value']:.3f} outside IQR bounds [{lower_bound:.3f}, {upper_bound:.3f}]"
                        })
        
        return anomalies
    
    async def _generate_predictions(self, data: List[Dict[str, Any]], pattern_type: TemporalPattern) -> List[Dict[str, Any]]:
        """Generate simple predictions based on temporal patterns"""
        
        if len(data) < 3:
            return []
        
        values = [d['avg_value'] for d in data]
        timestamps = [d['time_bucket'] for d in data]
        
        # Simple trend-based prediction
        recent_values = values[-5:]  # Use last 5 data points
        if len(recent_values) > 1:
            trend = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
        else:
            trend = 0
        
        predictions = []
        
        # Predict next few periods
        last_timestamp = timestamps[-1]
        last_value = values[-1]
        
        # Determine time delta based on pattern
        if pattern_type == TemporalPattern.HOURLY:
            delta = timedelta(hours=1)
        elif pattern_type == TemporalPattern.DAILY:
            delta = timedelta(days=1)
        elif pattern_type == TemporalPattern.WEEKLY:
            delta = timedelta(weeks=1)
        else:
            delta = timedelta(hours=1)
        
        # Generate 3 predictions
        for i in range(1, 4):
            predicted_timestamp = last_timestamp + (delta * i)
            predicted_value = last_value + (trend * i)
            
            # Add some bounds checking
            predicted_value = max(0, predicted_value)  # Non-negative values
            
            confidence = max(0.1, 1.0 - (i * 0.2))  # Decreasing confidence
            
            predictions.append({
                'timestamp': predicted_timestamp,
                'predicted_value': predicted_value,
                'confidence': confidence,
                'method': 'linear_trend',
                'trend_component': trend * i
            })
        
        return predictions
    
    def _generate_insights(self, 
                          statistics: Dict[str, float], 
                          trends: Dict[str, float], 
                          anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate human-readable insights from analysis"""
        
        insights = []
        
        # Statistical insights
        if statistics.get('coefficient_of_variation', 0) > 0.5:
            insights.append("High variability detected - metric values are inconsistent over time")
        elif statistics.get('coefficient_of_variation', 0) < 0.1:
            insights.append("Low variability - metric values are very stable")
        
        # Trend insights
        trend_direction = trends.get('trend_direction', 0)
        trend_strength = trends.get('trend_strength', 0)
        
        if trend_strength > 0.7:
            if trend_direction > 0:
                insights.append(f"Strong upward trend detected (strength: {trend_strength:.2f})")
            elif trend_direction < 0:
                insights.append(f"Strong downward trend detected (strength: {trend_strength:.2f})")
        elif trend_strength > 0.4:
            insights.append(f"Moderate trend detected with {trend_strength:.2f} strength")
        
        # Cyclical patterns
        cyclical_strength = trends.get('cyclical_strength', 0)
        if cyclical_strength > 0.5:
            insights.append(f"Cyclical pattern detected (strength: {cyclical_strength:.2f})")
        
        # Anomaly insights
        if anomalies:
            high_severity = sum(1 for a in anomalies if a.get('severity') == 'high')
            if high_severity > 0:
                insights.append(f"{high_severity} high-severity anomalies detected")
            insights.append(f"Total of {len(anomalies)} anomalies found in the data")
        
        # Performance insights
        volatility = trends.get('volatility', 0)
        if volatility > statistics.get('overall_mean', 0) * 0.5:
            insights.append("High volatility - consider investigating causes of fluctuations")
        
        return insights
    
    def _update_query_metrics(self, query_time_ms: float):
        """Update query performance metrics"""
        self.performance_metrics['queries_executed'] += 1
        
        # Update rolling average
        alpha = 0.1
        if self.performance_metrics['avg_query_time'] == 0:
            self.performance_metrics['avg_query_time'] = query_time_ms
        else:
            current_avg = self.performance_metrics['avg_query_time']
            self.performance_metrics['avg_query_time'] = (1 - alpha) * current_avg + alpha * query_time_ms
    
    async def get_realtime_dashboard_data(self, 
                                        metric_types: List[MetricType],
                                        time_window_minutes: int = 60) -> Dict[str, List[Dict[str, Any]]]:
        """Get real-time dashboard data for specified metrics"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        dashboard_data = {}
        
        async with self.pool.acquire() as conn:
            for metric_type in metric_types:
                query = """
                    SELECT 
                        time_bucket,
                        avg_value,
                        sample_count
                    FROM realtime_dashboard
                    WHERE metric_type = $1 
                    AND time_bucket BETWEEN $2 AND $3
                    ORDER BY time_bucket ASC
                """
                
                rows = await conn.fetch(query, metric_type.value, start_time, end_time)
                
                dashboard_data[metric_type.value] = [
                    {
                        'timestamp': row['time_bucket'],
                        'value': float(row['avg_value']),
                        'sample_count': int(row['sample_count'])
                    } for row in rows
                ]
        
        return dashboard_data
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        
        # Add database-specific metrics
        async with self.pool.acquire() as conn:
            # Get table size
            table_size_result = await conn.fetch("""
                SELECT pg_size_pretty(pg_total_relation_size('decision_metrics')) as table_size
            """)
            table_size = table_size_result[0]['table_size'] if table_size_result else "unknown"
            
            # Get record count
            count_result = await conn.fetch("SELECT COUNT(*) as total_records FROM decision_metrics")
            total_records = count_result[0]['total_records'] if count_result else 0
        
        return {
            **self.performance_metrics,
            'buffer_size': len(self.metric_buffer),
            'cache_size': len(self.pattern_cache),
            'table_size': table_size,
            'total_records': total_records,
            'connection_pool_size': self.pool.get_size() if self.pool else 0
        }
    
    async def cleanup_old_data(self, days_to_keep: int = None):
        """Clean up old data beyond retention period"""
        
        if days_to_keep is None:
            days_to_keep = self.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM decision_metrics 
                WHERE timestamp < $1
            """, cutoff_date)
            
            deleted_rows = int(result.split()[-1]) if result else 0
            logger.info(f"Cleaned up {deleted_rows} old records before {cutoff_date}")
            
            return deleted_rows
    
    async def close(self):
        """Close database connections and cleanup"""
        
        # Flush any remaining metrics
        async with self.buffer_lock:
            if self.metric_buffer:
                await self._flush_metrics()
        
        # Close connection pool
        if self.pool:
            await self.pool.close()
            
        logger.info("TimescaleDB analytics service closed")


# Example usage and testing
async def test_temporal_analytics():
    """Test the temporal analytics service"""
    
    config = {
        'host': 'localhost',
        'port': 5434,
        'database': 'timescale_test',
        'user': 'postgres',
        'password': 'postgres'
    }
    
    service = TimescaleAnalyticsService(config)
    
    try:
        await service.initialize()
        
        # Record some test metrics
        test_metrics = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(100):
            timestamp = base_time + timedelta(minutes=i*15)  # Every 15 minutes
            
            # Simulate decision latency with some trends and noise
            latency = 50 + 20 * np.sin(i / 10) + np.random.normal(0, 10)
            latency = max(10, latency)  # Ensure positive
            
            metric = TemporalMetric(
                timestamp=timestamp,
                metric_type=MetricType.DECISION_LATENCY,
                value=latency,
                user_id=f"user_{i % 10}",
                session_id=f"session_{i // 10}",
                context={'test': True, 'batch': i // 20},
                metadata={'synthetic': True}
            )
            test_metrics.append(metric)
        
        # Record metrics
        await service.record_metrics_batch(test_metrics)
        
        # Wait for flush
        await asyncio.sleep(2)
        
        # Analyze patterns
        analysis = await service.analyze_temporal_patterns(
            MetricType.DECISION_LATENCY,
            TemporalPattern.HOURLY,
            user_id="user_1"
        )
        
        print("=== TEMPORAL ANALYSIS RESULTS ===")
        print(f"Pattern: {analysis.pattern_type.value}")
        print(f"Metric: {analysis.metric_type.value}")
        print(f"Time range: {analysis.time_range[0]} to {analysis.time_range[1]}")
        print(f"Statistics: {analysis.statistics}")
        print(f"Trends: {analysis.trends}")
        print(f"Anomalies: {len(analysis.anomalies)}")
        print(f"Predictions: {len(analysis.predictions)}")
        print(f"Insights: {analysis.insights}")
        
        # Test dashboard data
        dashboard_data = await service.get_realtime_dashboard_data([MetricType.DECISION_LATENCY])
        print(f"\n=== DASHBOARD DATA ===")
        print(f"Data points: {len(dashboard_data.get('decision_latency', []))}")
        
        # Performance metrics
        perf_metrics = await service.get_performance_metrics()
        print(f"\n=== PERFORMANCE METRICS ===")
        for key, value in perf_metrics.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error testing temporal analytics: {e}")
    finally:
        await service.close()

if __name__ == "__main__":
    asyncio.run(test_temporal_analytics())
