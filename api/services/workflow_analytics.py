#\!/usr/bin/env python3
"""
TimescaleDB Workflow Analytics Service
Provides time-series analytics for workflow optimization patterns
Author: Joke Verhelst - FPGA Acceleration & Unified Memory Specialist
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncpg
import json
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WorkflowMetric:
    """Workflow performance metric"""
    workflow_id: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    metric_unit: str
    metadata: Dict[str, Any]

@dataclass
class PerformancePattern:
    """Identified performance pattern"""
    pattern_id: str
    pattern_type: str
    confidence_score: float
    description: str
    optimization_suggestions: List[str]
    affected_workflows: List[str]
    time_range: Tuple[datetime, datetime]

class WorkflowAnalyticsService:
    """
    TimescaleDB-based Workflow Analytics Service
    Provides real-time and historical performance analytics
    """
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or self._get_default_connection()
        self.connection_pool = None
        self.analytics_cache = {}
        self.pattern_detection_models = {}
        
        logger.info("Workflow Analytics Service initialized")
    
    def _get_default_connection(self) -> str:
        """Get default TimescaleDB connection string"""
        return "postgresql://knowledgehub:knowledgehub123@localhost:5434/knowledgehub_analytics"
    
    async def initialize(self):
        """Initialize TimescaleDB connection and create tables"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            await self._create_tables()
            logger.info("TimescaleDB workflow analytics initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow analytics: {e}")
            raise
    
    async def _create_tables(self):
        """Create TimescaleDB hypertables for workflow analytics"""
        async with self.connection_pool.acquire() as conn:
            # Workflow metrics hypertable
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_metrics (
                    timestamp TIMESTAMPTZ NOT NULL,
                    workflow_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value DOUBLE PRECISION NOT NULL,
                    metric_unit TEXT NOT NULL,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, workflow_id, metric_name)
                )
            """)
            
            # Create hypertable if not exists
            try:
                await conn.execute("""
                    SELECT create_hypertable('workflow_metrics', 'timestamp', 
                                           if_not_exists => TRUE)
                """)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Hypertable creation warning: {e}")
            
            # FPGA utilization tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS fpga_utilization (
                    timestamp TIMESTAMPTZ NOT NULL,
                    device_id INTEGER NOT NULL,
                    utilization_percent DOUBLE PRECISION NOT NULL,
                    memory_used_mb INTEGER NOT NULL,
                    memory_total_mb INTEGER NOT NULL,
                    temperature_c INTEGER,
                    power_usage_w INTEGER,
                    PRIMARY KEY (timestamp, device_id)
                )
            """)
            
            try:
                await conn.execute("""
                    SELECT create_hypertable('fpga_utilization', 'timestamp', 
                                           if_not_exists => TRUE)
                """)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"FPGA hypertable creation warning: {e}")
            
            # Memory usage patterns
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_usage_patterns (
                    timestamp TIMESTAMPTZ NOT NULL,
                    pool_name TEXT NOT NULL,
                    allocated_mb INTEGER NOT NULL,
                    peak_usage_mb INTEGER NOT NULL,
                    fragmentation_percent DOUBLE PRECISION,
                    cache_hit_rate DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, pool_name)
                )
            """)
            
            try:
                await conn.execute("""
                    SELECT create_hypertable('memory_usage_patterns', 'timestamp', 
                                           if_not_exists => TRUE)
                """)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Memory hypertable creation warning: {e}")
            
            # Performance anomalies
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_anomalies (
                    timestamp TIMESTAMPTZ NOT NULL,
                    anomaly_id TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    affected_workflows TEXT[],
                    detection_confidence DOUBLE PRECISION,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, anomaly_id)
                )
            """)
            
            try:
                await conn.execute("""
                    SELECT create_hypertable('performance_anomalies', 'timestamp', 
                                           if_not_exists => TRUE)
                """)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Anomaly hypertable creation warning: {e}")
            
            # Create indexes for better query performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_metrics_name 
                ON workflow_metrics (metric_name, timestamp DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fpga_device 
                ON fpga_utilization (device_id, timestamp DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_pool 
                ON memory_usage_patterns (pool_name, timestamp DESC)
            """)
            
            logger.info("TimescaleDB tables and indexes created successfully")
    
    async def record_workflow_metric(self, metric: WorkflowMetric):
        """Record a workflow performance metric"""
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO workflow_metrics 
                    (timestamp, workflow_id, metric_name, metric_value, metric_unit, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, metric.timestamp, metric.workflow_id, metric.metric_name,
                    metric.metric_value, metric.metric_unit, json.dumps(metric.metadata))
                
        except Exception as e:
            logger.error(f"Failed to record workflow metric: {e}")
    
    async def record_fpga_utilization(self, device_id: int, utilization_data: Dict[str, Any]):
        """Record FPGA utilization metrics"""
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO fpga_utilization 
                    (timestamp, device_id, utilization_percent, memory_used_mb, 
                     memory_total_mb, temperature_c, power_usage_w)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, 
                datetime.utcnow(),
                device_id,
                utilization_data.get('utilization_percent', 0),
                utilization_data.get('memory_used_mb', 0),
                utilization_data.get('memory_total_mb', 0),
                utilization_data.get('temperature_c'),
                utilization_data.get('power_usage_w')
                )
                
        except Exception as e:
            logger.error(f"Failed to record FPGA utilization: {e}")
    
    async def record_memory_usage(self, pool_name: str, usage_data: Dict[str, Any]):
        """Record memory usage patterns"""
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO memory_usage_patterns 
                    (timestamp, pool_name, allocated_mb, peak_usage_mb, 
                     fragmentation_percent, cache_hit_rate)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                datetime.utcnow(),
                pool_name,
                usage_data.get('allocated_mb', 0),
                usage_data.get('peak_usage_mb', 0),
                usage_data.get('fragmentation_percent', 0.0),
                usage_data.get('cache_hit_rate', 0.0)
                )
                
        except Exception as e:
            logger.error(f"Failed to record memory usage: {e}")
    
    async def get_workflow_performance_trends(self, 
                                            workflow_id: str = None,
                                            time_range_hours: int = 24) -> Dict[str, Any]:
        """Get workflow performance trends over time"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            async with self.connection_pool.acquire() as conn:
                # Base query
                where_clause = "WHERE timestamp >= $1"
                params = [start_time]
                
                if workflow_id:
                    where_clause += " AND workflow_id = $2"
                    params.append(workflow_id)
                
                # Get aggregated metrics
                query = f"""
                    SELECT 
                        metric_name,
                        time_bucket('1 hour', timestamp) as bucket,
                        AVG(metric_value) as avg_value,
                        MIN(metric_value) as min_value,
                        MAX(metric_value) as max_value,
                        COUNT(*) as measurement_count
                    FROM workflow_metrics 
                    {where_clause}
                    GROUP BY metric_name, bucket
                    ORDER BY bucket DESC, metric_name
                """
                
                rows = await conn.fetch(query, *params)
                
                # Organize data by metric
                trends = {}
                for row in rows:
                    metric_name = row['metric_name']
                    if metric_name not in trends:
                        trends[metric_name] = {
                            'timestamps': [],
                            'avg_values': [],
                            'min_values': [],
                            'max_values': [],
                            'measurement_counts': []
                        }
                    
                    trends[metric_name]['timestamps'].append(row['bucket'])
                    trends[metric_name]['avg_values'].append(float(row['avg_value']))
                    trends[metric_name]['min_values'].append(float(row['min_value']))
                    trends[metric_name]['max_values'].append(float(row['max_value']))
                    trends[metric_name]['measurement_counts'].append(row['measurement_count'])
                
                return {
                    'workflow_id': workflow_id,
                    'time_range_hours': time_range_hours,
                    'trends': trends,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return {'error': str(e)}
    
    async def detect_performance_anomalies(self, 
                                         sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis"""
        try:
            # Look at the last 24 hours
            start_time = datetime.utcnow() - timedelta(hours=24)
            
            async with self.connection_pool.acquire() as conn:
                # Get recent metrics with statistical analysis
                query = """
                    WITH metric_stats AS (
                        SELECT 
                            metric_name,
                            workflow_id,
                            timestamp,
                            metric_value,
                            AVG(metric_value) OVER (
                                PARTITION BY metric_name 
                                ORDER BY timestamp 
                                ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
                            ) as moving_avg,
                            STDDEV(metric_value) OVER (
                                PARTITION BY metric_name 
                                ORDER BY timestamp 
                                ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
                            ) as moving_stddev
                        FROM workflow_metrics 
                        WHERE timestamp >= $1
                    ),
                    anomalies AS (
                        SELECT 
                            *,
                            ABS(metric_value - moving_avg) / NULLIF(moving_stddev, 0) as z_score
                        FROM metric_stats
                        WHERE moving_stddev > 0
                    )
                    SELECT *
                    FROM anomalies 
                    WHERE z_score > $2
                    ORDER BY timestamp DESC, z_score DESC
                    LIMIT 100
                """
                
                rows = await conn.fetch(query, start_time, sensitivity)
                
                anomalies = []
                for row in rows:
                    anomaly = {
                        'timestamp': row['timestamp'].isoformat(),
                        'workflow_id': row['workflow_id'],
                        'metric_name': row['metric_name'],
                        'metric_value': float(row['metric_value']),
                        'expected_value': float(row['moving_avg']),
                        'z_score': float(row['z_score']),
                        'severity': self._classify_anomaly_severity(float(row['z_score'])),
                        'description': f"Unusual {row['metric_name']} value detected"
                    }
                    anomalies.append(anomaly)
                
                # Group related anomalies
                grouped_anomalies = self._group_related_anomalies(anomalies)
                
                return grouped_anomalies
                
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return []
    
    def _classify_anomaly_severity(self, z_score: float) -> str:
        """Classify anomaly severity based on z-score"""
        if z_score > 4.0:
            return "critical"
        elif z_score > 3.0:
            return "high"
        elif z_score > 2.0:
            return "medium"
        else:
            return "low"
    
    def _group_related_anomalies(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group related anomalies by time and workflow"""
        grouped = {}
        
        for anomaly in anomalies:
            # Create a key based on workflow and time window (5 minutes)
            timestamp = datetime.fromisoformat(anomaly['timestamp'].replace('T', ' '))
            time_bucket = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
            key = f"{anomaly['workflow_id']}_{time_bucket}"
            
            if key not in grouped:
                grouped[key] = {
                    'group_id': key,
                    'workflow_id': anomaly['workflow_id'],
                    'time_window': time_bucket.isoformat(),
                    'anomalies': [],
                    'max_severity': 'low',
                    'affected_metrics': set()
                }
            
            grouped[key]['anomalies'].append(anomaly)
            grouped[key]['affected_metrics'].add(anomaly['metric_name'])
            
            # Update severity to highest in group
            severities = ['low', 'medium', 'high', 'critical']
            current_severity_idx = severities.index(grouped[key]['max_severity'])
            anomaly_severity_idx = severities.index(anomaly['severity'])
            
            if anomaly_severity_idx > current_severity_idx:
                grouped[key]['max_severity'] = anomaly['severity']
        
        # Convert to list and add metadata
        result = []
        for group in grouped.values():
            group['affected_metrics'] = list(group['affected_metrics'])
            group['anomaly_count'] = len(group['anomalies'])
            result.append(group)
        
        # Sort by severity and time
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        result.sort(key=lambda x: (severity_order[x['max_severity']], x['time_window']), reverse=True)
        
        return result
    
    async def get_fpga_utilization_analysis(self, 
                                          device_id: int = None,
                                          hours: int = 24) -> Dict[str, Any]:
        """Get FPGA utilization analysis and trends"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            async with self.connection_pool.acquire() as conn:
                where_clause = "WHERE timestamp >= $1"
                params = [start_time]
                
                if device_id is not None:
                    where_clause += " AND device_id = $2"
                    params.append(device_id)
                
                # Get aggregated FPGA utilization data
                query = f"""
                    SELECT 
                        device_id,
                        time_bucket('15 minutes', timestamp) as bucket,
                        AVG(utilization_percent) as avg_utilization,
                        MAX(utilization_percent) as peak_utilization,
                        AVG(memory_used_mb) as avg_memory_used,
                        MAX(memory_used_mb) as peak_memory_used,
                        AVG(memory_total_mb) as memory_total,
                        AVG(temperature_c) as avg_temperature,
                        MAX(temperature_c) as peak_temperature,
                        AVG(power_usage_w) as avg_power
                    FROM fpga_utilization
                    {where_clause}
                    GROUP BY device_id, bucket
                    ORDER BY device_id, bucket DESC
                """
                
                rows = await conn.fetch(query, *params)
                
                # Organize by device
                devices = {}
                for row in rows:
                    dev_id = row['device_id']
                    if dev_id not in devices:
                        devices[dev_id] = {
                            'device_id': dev_id,
                            'timestamps': [],
                            'utilization': [],
                            'memory_usage': [],
                            'temperature': [],
                            'power_usage': []
                        }
                    
                    devices[dev_id]['timestamps'].append(row['bucket'])
                    devices[dev_id]['utilization'].append(float(row['avg_utilization']))
                    devices[dev_id]['memory_usage'].append({
                        'used_mb': float(row['avg_memory_used']),
                        'total_mb': float(row['memory_total']),
                        'usage_percent': (float(row['avg_memory_used']) / float(row['memory_total'])) * 100
                    })
                    
                    if row['avg_temperature']:
                        devices[dev_id]['temperature'].append(float(row['avg_temperature']))
                    if row['avg_power']:
                        devices[dev_id]['power_usage'].append(float(row['avg_power']))
                
                # Calculate summary statistics
                summary = {}
                for dev_id, data in devices.items():
                    if data['utilization']:
                        summary[dev_id] = {
                            'avg_utilization': np.mean(data['utilization']),
                            'peak_utilization': np.max(data['utilization']),
                            'utilization_efficiency': self._calculate_efficiency_score(data['utilization']),
                            'memory_efficiency': np.mean([m['usage_percent'] for m in data['memory_usage']]),
                            'thermal_status': 'normal' if not data['temperature'] or np.max(data['temperature']) < 80 else 'high'
                        }
                
                return {
                    'device_analysis': devices,
                    'summary_statistics': summary,
                    'analysis_period': f"{hours} hours",
                    'generated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get FPGA utilization analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_efficiency_score(self, utilization_values: List[float]) -> float:
        """Calculate FPGA efficiency score based on utilization consistency"""
        if not utilization_values:
            return 0.0
        
        mean_util = np.mean(utilization_values)
        std_util = np.std(utilization_values)
        
        # Efficiency score: high average utilization with low variance is better
        consistency_score = max(0, 1 - (std_util / 100))  # Normalize std by max possible
        utilization_score = mean_util / 100  # Normalize to 0-1
        
        # Weighted combination: 60% utilization, 40% consistency
        efficiency = (utilization_score * 0.6) + (consistency_score * 0.4)
        
        return min(1.0, efficiency)
    
    async def generate_optimization_insights(self) -> Dict[str, Any]:
        """Generate AI-powered optimization insights from analytics data"""
        try:
            insights = {
                'performance_insights': [],
                'resource_optimization': [],
                'predictive_recommendations': [],
                'cost_analysis': [],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Analyze workflow performance patterns
            performance_trends = await self.get_workflow_performance_trends(time_range_hours=168)  # 1 week
            
            for metric_name, trend_data in performance_trends.get('trends', {}).items():
                if not trend_data['avg_values']:
                    continue
                
                # Calculate trend direction
                recent_avg = np.mean(trend_data['avg_values'][:24])  # Last 24 hours
                historical_avg = np.mean(trend_data['avg_values'][24:])  # Previous period
                
                if historical_avg > 0:
                    trend_change = ((recent_avg - historical_avg) / historical_avg) * 100
                    
                    if abs(trend_change) > 10:  # Significant change
                        insights['performance_insights'].append({
                            'metric': metric_name,
                            'trend_change_percent': trend_change,
                            'trend_direction': 'improving' if trend_change < 0 and 'time' in metric_name.lower() else 'degrading',
                            'significance': 'high' if abs(trend_change) > 25 else 'medium',
                            'recommendation': self._generate_metric_recommendation(metric_name, trend_change)
                        })
            
            # FPGA utilization analysis
            fpga_analysis = await self.get_fpga_utilization_analysis(hours=72)  # 3 days
            
            for device_id, stats in fpga_analysis.get('summary_statistics', {}).items():
                if stats['avg_utilization'] < 60:
                    insights['resource_optimization'].append({
                        'resource_type': 'fpga',
                        'device_id': device_id,
                        'issue': 'low_utilization',
                        'current_utilization': stats['avg_utilization'],
                        'potential_improvement': f"{100 - stats['avg_utilization']:.1f}% underutilized",
                        'recommendation': 'Increase FPGA workload allocation or consider workload consolidation'
                    })
                
                if stats['efficiency_score'] < 0.7:
                    insights['resource_optimization'].append({
                        'resource_type': 'fpga',
                        'device_id': device_id,
                        'issue': 'inefficient_usage',
                        'efficiency_score': stats['efficiency_score'],
                        'recommendation': 'Optimize task scheduling to reduce utilization variance'
                    })
            
            # Detect recent anomalies for predictive recommendations
            recent_anomalies = await self.detect_performance_anomalies(sensitivity=2.5)
            
            if recent_anomalies:
                # Group anomalies by type for pattern recognition
                anomaly_patterns = {}
                for anomaly_group in recent_anomalies:
                    for anomaly in anomaly_group['anomalies']:
                        metric = anomaly['metric_name']
                        if metric not in anomaly_patterns:
                            anomaly_patterns[metric] = []
                        anomaly_patterns[metric].append(anomaly)
                
                for metric, anomalies in anomaly_patterns.items():
                    if len(anomalies) >= 3:  # Pattern detected
                        insights['predictive_recommendations'].append({
                            'pattern_type': 'recurring_anomaly',
                            'metric': metric,
                            'frequency': len(anomalies),
                            'severity': max([a['severity'] for a in anomalies]),
                            'prediction': f'Likely to see continued {metric} anomalies',
                            'recommendation': self._generate_anomaly_recommendation(metric, anomalies)
                        })
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate optimization insights: {e}")
            return {'error': str(e)}
    
    def _generate_metric_recommendation(self, metric_name: str, trend_change: float) -> str:
        """Generate recommendation based on metric trend"""
        metric_lower = metric_name.lower()
        
        if 'time' in metric_lower or 'duration' in metric_lower:
            if trend_change > 0:
                return "Processing times are increasing. Consider optimizing algorithms or increasing resources."
            else:
                return "Processing times are improving. Current optimizations are effective."
        
        elif 'memory' in metric_lower:
            if trend_change > 0:
                return "Memory usage is increasing. Monitor for memory leaks and consider memory optimization."
            else:
                return "Memory usage is decreasing. Memory optimizations are working well."
        
        elif 'throughput' in metric_lower:
            if trend_change > 0:
                return "Throughput is improving. Consider scaling to handle more workload."
            else:
                return "Throughput is declining. Investigate bottlenecks and scaling needs."
        
        else:
            if trend_change > 0:
                return f"{metric_name} values are increasing. Monitor for potential issues."
            else:
                return f"{metric_name} values are decreasing. Current trend appears positive."
    
    def _generate_anomaly_recommendation(self, metric_name: str, anomalies: List[Dict[str, Any]]) -> str:
        """Generate recommendation for recurring anomalies"""
        avg_z_score = np.mean([a['z_score'] for a in anomalies])
        
        if avg_z_score > 4:
            return f"Critical {metric_name} anomalies detected. Immediate investigation required."
        elif avg_z_score > 3:
            return f"Significant {metric_name} variations. Review recent changes and optimize accordingly."
        else:
            return f"Monitor {metric_name} trends. Consider adjusting thresholds or implementing gradual optimizations."
    
    async def close(self):
        """Close database connections"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Workflow analytics service closed")

# Global analytics service instance
analytics_service = None

async def get_analytics_service() -> WorkflowAnalyticsService:
    """Get global analytics service instance"""
    global analytics_service
    if analytics_service is None:
        analytics_service = WorkflowAnalyticsService()
        await analytics_service.initialize()
    return analytics_service

logger.info("Workflow Analytics Service module loaded")
