"""
Real-Time AI Intelligence System
Created by Annelies Claes - Expert in Lottery Ticket Hypothesis & Real-Time AI Systems

This system provides real-time AI intelligence features including user behavior analysis,
content quality monitoring, and proactive recommendations using efficient quantized models.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import numpy as np
from threading import Lock
import websockets
from contextlib import asynccontextmanager

from .lottery_ticket_pattern_engine import LotteryTicketPatternEngine, PatternMatch
from .quantized_ai_service import QuantizedAIService
from .advanced_semantic_analysis import AdvancedSemanticAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class RealTimeEvent:
    """Represents a real-time event in the system."""
    event_id: str
    user_id: str
    event_type: str  # 'content_view', 'search', 'edit', 'pattern_detected'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False

@dataclass
class UserSession:
    """Represents a user session with real-time tracking."""
    user_id: str
    session_id: str
    start_time: datetime
    last_activity: datetime
    events: deque = field(default_factory=lambda: deque(maxlen=1000))
    behavior_profile: Dict[str, Any] = field(default_factory=dict)
    anomaly_score: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class IntelligenceAlert:
    """Represents an intelligence alert from real-time analysis."""
    alert_id: str
    alert_type: str  # 'security', 'quality', 'behavior', 'performance'
    severity: str  # 'critical', 'high', 'medium', 'low'
    user_id: Optional[str]
    content: str
    confidence: float
    recommendations: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class RealTimeIntelligenceEngine:
    """
    Real-Time AI Intelligence Engine with quantized neural networks.
    
    Features:
    - Real-time pattern detection using Lottery Ticket Hypothesis
    - User behavior analysis and anomaly detection
    - Content quality monitoring
    - Proactive recommendations
    - WebSocket-based real-time updates
    - Adaptive learning from user interactions
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        analysis_window: int = 300,  # 5 minutes
        alert_threshold: float = 0.7,
        max_concurrent_analysis: int = 50
    ):
        self.buffer_size = buffer_size
        self.analysis_window = analysis_window
        self.alert_threshold = alert_threshold
        self.max_concurrent_analysis = max_concurrent_analysis
        
        # Core AI components
        self.pattern_engine: Optional[LotteryTicketPatternEngine] = None
        self.semantic_analyzer: Optional[AdvancedSemanticAnalyzer] = None
        self.ai_service: Optional[QuantizedAIService] = None
        
        # Real-time data structures
        self.event_buffer = deque(maxlen=buffer_size)
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.alert_queue = deque(maxlen=1000)
        
        # Thread safety
        self.buffer_lock = Lock()
        self.session_lock = Lock()
        
        # Performance metrics
        self.processing_stats = {
            'events_processed': 0,
            'alerts_generated': 0,
            'avg_processing_time': 0.0,
            'patterns_detected': 0,
            'anomalies_detected': 0
        }
        
        # WebSocket connections
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Background tasks
        self.analysis_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("RealTimeIntelligenceEngine initialized")

    async def initialize(self):
        """Initialize the real-time intelligence engine."""
        try:
            # Initialize AI components
            self.pattern_engine = LotteryTicketPatternEngine(
                sparsity_target=0.15,  # More sparse for real-time performance
                quantization_bits=4    # More aggressive quantization for speed
            )
            await self.pattern_engine.initialize_embedding_model()
            
            self.semantic_analyzer = AdvancedSemanticAnalyzer(
                use_quantization=True,
                quantization_bits=8
            )
            await self.semantic_analyzer.initialize()
            
            self.ai_service = QuantizedAIService()
            await self.ai_service.initialize()
            
            # Start background processing tasks
            self.analysis_task = asyncio.create_task(self._continuous_analysis())
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info("RealTimeIntelligenceEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RealTimeIntelligenceEngine: {e}")
            raise

    async def process_event(self, event: RealTimeEvent) -> List[IntelligenceAlert]:
        """Process a single real-time event and generate alerts if necessary."""
        alerts = []
        
        try:
            start_time = time.time()
            
            # Add to buffer
            with self.buffer_lock:
                self.event_buffer.append(event)
            
            # Update user session
            await self._update_user_session(event)
            
            # Immediate analysis for critical patterns
            if event.event_type in ['content_edit', 'security_action']:
                immediate_alerts = await self._immediate_analysis(event)
                alerts.extend(immediate_alerts)
            
            # Update processing stats
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Event processing failed: {e}")
            return []

    async def _update_user_session(self, event: RealTimeEvent):
        """Update user session with new event."""
        with self.session_lock:
            session_key = f"{event.user_id}:{event.metadata.get('session_id', 'default')}"
            
            if session_key not in self.active_sessions:
                self.active_sessions[session_key] = UserSession(
                    user_id=event.user_id,
                    session_id=event.metadata.get('session_id', 'default'),
                    start_time=event.timestamp,
                    last_activity=event.timestamp
                )
            
            session = self.active_sessions[session_key]
            session.events.append(event)
            session.last_activity = event.timestamp
            
            # Update behavior profile
            self._update_behavior_profile(session, event)

    def _update_behavior_profile(self, session: UserSession, event: RealTimeEvent):
        """Update user behavior profile based on new event."""
        profile = session.behavior_profile
        
        # Track event types
        event_types = profile.get('event_types', defaultdict(int))
        event_types[event.event_type] += 1
        profile['event_types'] = dict(event_types)
        
        # Track timing patterns
        if len(session.events) > 1:
            prev_event = session.events[-2]
            time_diff = (event.timestamp - prev_event.timestamp).total_seconds()
            
            intervals = profile.get('intervals', [])
            intervals.append(time_diff)
            if len(intervals) > 100:  # Keep last 100 intervals
                intervals = intervals[-100:]
            profile['intervals'] = intervals
            
            # Calculate average interval
            profile['avg_interval'] = np.mean(intervals) if intervals else 0
            profile['interval_std'] = np.std(intervals) if len(intervals) > 1 else 0
        
        # Track content patterns
        content_lengths = profile.get('content_lengths', [])
        content_lengths.append(len(event.content))
        if len(content_lengths) > 50:  # Keep last 50 lengths
            content_lengths = content_lengths[-50:]
        profile['content_lengths'] = content_lengths
        profile['avg_content_length'] = np.mean(content_lengths) if content_lengths else 0

    async def _immediate_analysis(self, event: RealTimeEvent) -> List[IntelligenceAlert]:
        """Perform immediate analysis for critical events."""
        alerts = []
        
        try:
            # Pattern detection
            patterns = await self.pattern_engine.analyze_content(
                content=event.content,
                content_type="text"
            )
            
            # Generate alerts for critical patterns
            for pattern in patterns:
                if pattern.severity in ['critical', 'high'] and pattern.confidence > self.alert_threshold:
                    alert = IntelligenceAlert(
                        alert_id=f"immediate_{int(time.time())}_{pattern.pattern_id}",
                        alert_type='security' if 'security' in pattern.pattern_name.lower() else 'quality',
                        severity=pattern.severity,
                        user_id=event.user_id,
                        content=event.content[:200] + "..." if len(event.content) > 200 else event.content,
                        confidence=pattern.confidence,
                        recommendations=[
                            f"Address {pattern.pattern_name} pattern detected in content",
                            "Review security implications if applicable",
                            "Consider content quality improvements"
                        ],
                        timestamp=datetime.utcnow(),
                        metadata={
                            'pattern_id': pattern.pattern_id,
                            'pattern_type': pattern.metadata.get('pattern_type'),
                            'event_id': event.event_id
                        }
                    )
                    alerts.append(alert)
            
            # Store alerts
            with self.buffer_lock:
                self.alert_queue.extend(alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Immediate analysis failed: {e}")
            return []

    async def _continuous_analysis(self):
        """Continuous background analysis of buffered events."""
        logger.info("Starting continuous analysis task")
        
        while True:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds
                
                # Get events from buffer
                events_to_process = []
                with self.buffer_lock:
                    # Get unprocessed events
                    events_to_process = [e for e in self.event_buffer if not e.processed]
                    # Mark as processed
                    for event in events_to_process:
                        event.processed = True
                
                if not events_to_process:
                    continue
                
                logger.debug(f"Processing {len(events_to_process)} events in background")
                
                # Batch analysis
                alerts = await self._batch_analysis(events_to_process)
                
                # Store generated alerts
                if alerts:
                    with self.buffer_lock:
                        self.alert_queue.extend(alerts)
                    
                    # Broadcast alerts via WebSocket
                    await self._broadcast_alerts(alerts)
                
                # Update performance stats
                self.processing_stats['events_processed'] += len(events_to_process)
                self.processing_stats['alerts_generated'] += len(alerts)
                
            except Exception as e:
                logger.error(f"Continuous analysis error: {e}")
                await asyncio.sleep(30)  # Back off on error

    async def _batch_analysis(self, events: List[RealTimeEvent]) -> List[IntelligenceAlert]:
        """Perform batch analysis on multiple events."""
        alerts = []
        
        try:
            # Group events by user for behavior analysis
            user_events = defaultdict(list)
            for event in events:
                user_events[event.user_id].append(event)
            
            # Analyze each user's events
            for user_id, user_event_list in user_events.items():
                user_alerts = await self._analyze_user_events(user_id, user_event_list)
                alerts.extend(user_alerts)
            
            # Content quality analysis
            content_alerts = await self._analyze_content_quality(events)
            alerts.extend(content_alerts)
            
            # Pattern correlation analysis
            pattern_alerts = await self._analyze_pattern_correlations(events)
            alerts.extend(pattern_alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return []

    async def _analyze_user_events(
        self, 
        user_id: str, 
        events: List[RealTimeEvent]
    ) -> List[IntelligenceAlert]:
        """Analyze events for a specific user to detect behavioral patterns."""
        alerts = []
        
        try:
            if len(events) < 3:  # Need minimum events for analysis
                return alerts
            
            # Get user session
            session_key = f"{user_id}:{events[0].metadata.get('session_id', 'default')}"
            session = self.active_sessions.get(session_key)
            
            if not session:
                return alerts
            
            # Anomaly detection
            anomalies = await self._detect_user_anomalies(session, events)
            
            for anomaly in anomalies:
                alert = IntelligenceAlert(
                    alert_id=f"behavior_{int(time.time())}_{user_id}",
                    alert_type='behavior',
                    severity='medium' if anomaly['severity'] > 0.7 else 'low',
                    user_id=user_id,
                    content=f"Behavioral anomaly detected: {anomaly['description']}",
                    confidence=anomaly['confidence'],
                    recommendations=[
                        "Monitor user behavior for potential security concerns",
                        "Check for automated or bot-like activity",
                        "Verify user identity if necessary"
                    ],
                    timestamp=datetime.utcnow(),
                    metadata={
                        'anomaly_type': anomaly['type'],
                        'session_id': session.session_id,
                        'event_count': len(events)
                    }
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"User event analysis failed: {e}")
            return []

    async def _detect_user_anomalies(
        self, 
        session: UserSession, 
        recent_events: List[RealTimeEvent]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in user behavior."""
        anomalies = []
        
        try:
            profile = session.behavior_profile
            
            # Rapid fire detection
            if len(recent_events) > 10:
                time_span = (recent_events[-1].timestamp - recent_events[0].timestamp).total_seconds()
                if time_span < 30:  # More than 10 events in 30 seconds
                    anomalies.append({
                        'type': 'rapid_fire',
                        'description': f'Rapid activity: {len(recent_events)} events in {time_span:.1f} seconds',
                        'severity': 0.8,
                        'confidence': 0.9
                    })
            
            # Unusual interval patterns
            intervals = profile.get('intervals', [])
            if len(intervals) > 10:
                recent_intervals = intervals[-10:]
                avg_recent = np.mean(recent_intervals)
                historical_avg = profile.get('avg_interval', avg_recent)
                
                if historical_avg > 0 and avg_recent < historical_avg * 0.1:  # 10x faster than usual
                    anomalies.append({
                        'type': 'unusual_speed',
                        'description': f'Unusually fast activity pattern detected',
                        'severity': 0.7,
                        'confidence': 0.8
                    })
            
            # Content pattern anomalies
            content_lengths = [len(event.content) for event in recent_events]
            if content_lengths:
                avg_recent_length = np.mean(content_lengths)
                historical_avg_length = profile.get('avg_content_length', avg_recent_length)
                
                if historical_avg_length > 0 and avg_recent_length < historical_avg_length * 0.1:
                    anomalies.append({
                        'type': 'unusual_content',
                        'description': 'Unusually short content pattern detected',
                        'severity': 0.6,
                        'confidence': 0.7
                    })
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return []

    async def _analyze_content_quality(self, events: List[RealTimeEvent]) -> List[IntelligenceAlert]:
        """Analyze content quality across events."""
        alerts = []
        
        try:
            # Extract content from events
            content_samples = [event.content for event in events if len(event.content) > 50]
            
            if len(content_samples) < 5:  # Need minimum samples
                return alerts
            
            # Quality analysis using semantic analyzer
            quality_scores = []
            for content in content_samples[:10]:  # Limit to 10 samples for performance
                try:
                    analysis = await self.semantic_analyzer.analyze_document(
                        document_id=f"temp_{hash(content)}",
                        content=content
                    )
                    quality_scores.append(analysis.quality_score)
                except:
                    continue
            
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                
                if avg_quality < 0.4:  # Poor quality threshold
                    alert = IntelligenceAlert(
                        alert_id=f"quality_{int(time.time())}",
                        alert_type='quality',
                        severity='medium',
                        user_id=None,  # System-wide alert
                        content=f"Low content quality detected across {len(quality_scores)} samples",
                        confidence=0.8,
                        recommendations=[
                            "Review content creation guidelines",
                            "Implement content quality checks",
                            "Provide user education on content standards"
                        ],
                        timestamp=datetime.utcnow(),
                        metadata={
                            'avg_quality_score': avg_quality,
                            'samples_analyzed': len(quality_scores),
                            'quality_threshold': 0.4
                        }
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Content quality analysis failed: {e}")
            return []

    async def _analyze_pattern_correlations(self, events: List[RealTimeEvent]) -> List[IntelligenceAlert]:
        """Analyze correlations between different patterns."""
        alerts = []
        
        try:
            # Group events by pattern types detected
            pattern_events = defaultdict(list)
            
            for event in events:
                # Get patterns for each event (simplified)
                if 'security' in event.content.lower():
                    pattern_events['security'].append(event)
                if 'error' in event.content.lower():
                    pattern_events['error'].append(event)
                if len(event.content) > 1000:
                    pattern_events['long_content'].append(event)
            
            # Check for suspicious correlations
            if len(pattern_events['security']) > 3 and len(pattern_events['error']) > 3:
                # Security events correlated with errors
                alert = IntelligenceAlert(
                    alert_id=f"correlation_{int(time.time())}",
                    alert_type='security',
                    severity='high',
                    user_id=None,
                    content="Correlation detected between security events and system errors",
                    confidence=0.9,
                    recommendations=[
                        "Investigate potential security breach",
                        "Check system logs for related incidents",
                        "Review security monitoring systems"
                    ],
                    timestamp=datetime.utcnow(),
                    metadata={
                        'security_events': len(pattern_events['security']),
                        'error_events': len(pattern_events['error']),
                        'correlation_type': 'security_error'
                    }
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Pattern correlation analysis failed: {e}")
            return []

    async def _broadcast_alerts(self, alerts: List[IntelligenceAlert]):
        """Broadcast alerts to connected WebSocket clients."""
        if not self.websocket_clients:
            return
        
        try:
            for alert in alerts:
                message = {
                    'type': 'intelligence_alert',
                    'alert': {
                        'alert_id': alert.alert_id,
                        'alert_type': alert.alert_type,
                        'severity': alert.severity,
                        'user_id': alert.user_id,
                        'content': alert.content,
                        'confidence': alert.confidence,
                        'recommendations': alert.recommendations,
                        'timestamp': alert.timestamp.isoformat(),
                        'metadata': alert.metadata
                    }
                }
                
                # Send to all connected clients
                disconnected_clients = set()
                for client in self.websocket_clients:
                    try:
                        await client.send(json.dumps(message))
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)
                    except Exception as e:
                        logger.warning(f"Failed to send alert to client: {e}")
                        disconnected_clients.add(client)
                
                # Remove disconnected clients
                self.websocket_clients -= disconnected_clients
                
        except Exception as e:
            logger.error(f"Alert broadcast failed: {e}")

    async def _periodic_cleanup(self):
        """Periodic cleanup of old sessions and data."""
        logger.info("Starting periodic cleanup task")
        
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                current_time = datetime.utcnow()
                
                # Clean up old sessions
                with self.session_lock:
                    sessions_to_remove = []
                    for session_key, session in self.active_sessions.items():
                        time_since_activity = current_time - session.last_activity
                        if time_since_activity > timedelta(hours=1):  # 1 hour timeout
                            sessions_to_remove.append(session_key)
                    
                    for session_key in sessions_to_remove:
                        del self.active_sessions[session_key]
                    
                    if sessions_to_remove:
                        logger.debug(f"Cleaned up {len(sessions_to_remove)} inactive sessions")
                
                # Clean up old alerts
                with self.buffer_lock:
                    old_alerts = []
                    for i, alert in enumerate(self.alert_queue):
                        time_since_alert = current_time - alert.timestamp
                        if time_since_alert > timedelta(hours=24):  # 24 hour retention
                            old_alerts.append(i)
                    
                    # Remove old alerts (from end to preserve indices)
                    for i in reversed(old_alerts):
                        del self.alert_queue[i]
                
            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")

    def _update_processing_stats(self, processing_time: float):
        """Update processing statistics."""
        stats = self.processing_stats
        current_avg = stats['avg_processing_time']
        total_processed = stats['events_processed']
        
        # Update running average
        stats['avg_processing_time'] = (
            (current_avg * total_processed + processing_time) / (total_processed + 1)
        )

    async def add_websocket_client(self, websocket):
        """Add a WebSocket client for real-time updates."""
        self.websocket_clients.add(websocket)
        logger.debug(f"Added WebSocket client. Total: {len(self.websocket_clients)}")

    async def remove_websocket_client(self, websocket):
        """Remove a WebSocket client."""
        self.websocket_clients.discard(websocket)
        logger.debug(f"Removed WebSocket client. Total: {len(self.websocket_clients)}")

    async def get_recent_alerts(
        self, 
        limit: int = 50, 
        severity_filter: Optional[str] = None,
        user_id_filter: Optional[str] = None
    ) -> List[IntelligenceAlert]:
        """Get recent intelligence alerts with optional filtering."""
        alerts = []
        
        with self.buffer_lock:
            for alert in reversed(self.alert_queue):  # Most recent first
                if severity_filter and alert.severity != severity_filter:
                    continue
                if user_id_filter and alert.user_id != user_id_filter:
                    continue
                
                alerts.append(alert)
                if len(alerts) >= limit:
                    break
        
        return alerts

    async def get_user_behavior_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user behavior patterns."""
        summary = {
            'user_id': user_id,
            'active_sessions': 0,
            'total_events': 0,
            'behavior_profile': {},
            'anomaly_score': 0.0,
            'recent_patterns': [],
            'risk_assessment': 'low'
        }
        
        try:
            with self.session_lock:
                user_sessions = [
                    session for session in self.active_sessions.values()
                    if session.user_id == user_id
                ]
                
                summary['active_sessions'] = len(user_sessions)
                
                if user_sessions:
                    # Aggregate behavior data
                    total_events = sum(len(session.events) for session in user_sessions)
                    summary['total_events'] = total_events
                    
                    # Get most recent session's behavior profile
                    latest_session = max(user_sessions, key=lambda s: s.last_activity)
                    summary['behavior_profile'] = latest_session.behavior_profile.copy()
                    summary['anomaly_score'] = latest_session.anomaly_score
                    
                    # Risk assessment
                    if latest_session.anomaly_score > 0.8:
                        summary['risk_assessment'] = 'high'
                    elif latest_session.anomaly_score > 0.6:
                        summary['risk_assessment'] = 'medium'
            
            # Get recent alerts for this user
            recent_user_alerts = await self.get_recent_alerts(
                limit=10, 
                user_id_filter=user_id
            )
            
            summary['recent_alerts'] = len(recent_user_alerts)
            summary['recent_patterns'] = [
                alert.alert_type for alert in recent_user_alerts
            ]
            
        except Exception as e:
            logger.error(f"User behavior summary failed: {e}")
        
        return summary

    async def get_system_intelligence_status(self) -> Dict[str, Any]:
        """Get comprehensive system intelligence status."""
        status = {
            'service': 'RealTimeIntelligenceEngine',
            'status': 'operational',
            'timestamp': datetime.utcnow().isoformat(),
            'processing_stats': self.processing_stats.copy(),
            'active_components': {},
            'real_time_metrics': {}
        }
        
        try:
            # Component status
            status['active_components'] = {
                'pattern_engine': self.pattern_engine is not None,
                'semantic_analyzer': self.semantic_analyzer is not None,
                'ai_service': self.ai_service is not None,
                'analysis_task_running': self.analysis_task and not self.analysis_task.done(),
                'cleanup_task_running': self.cleanup_task and not self.cleanup_task.done()
            }
            
            # Real-time metrics
            with self.buffer_lock:
                buffer_size = len(self.event_buffer)
                alert_count = len(self.alert_queue)
            
            with self.session_lock:
                active_session_count = len(self.active_sessions)
            
            status['real_time_metrics'] = {
                'buffer_size': buffer_size,
                'buffer_capacity': self.buffer_size,
                'active_sessions': active_session_count,
                'pending_alerts': alert_count,
                'websocket_clients': len(self.websocket_clients),
                'events_per_second': self._calculate_events_per_second()
            }
            
            # Health assessment
            if not all(status['active_components'].values()):
                status['status'] = 'degraded'
            
            if buffer_size > self.buffer_size * 0.9:
                status['status'] = 'overloaded'
            
        except Exception as e:
            status['status'] = 'error'
            status['error'] = str(e)
        
        return status

    def _calculate_events_per_second(self) -> float:
        """Calculate current events per second rate."""
        try:
            if self.processing_stats['events_processed'] == 0:
                return 0.0
            
            # Simple approximation based on recent processing
            recent_time_window = 60  # 1 minute
            recent_events = sum(
                1 for event in self.event_buffer
                if (datetime.utcnow() - event.timestamp).total_seconds() <= recent_time_window
            )
            
            return recent_events / recent_time_window
            
        except Exception:
            return 0.0

    async def shutdown(self):
        """Graceful shutdown of the intelligence engine."""
        logger.info("Shutting down RealTimeIntelligenceEngine")
        
        try:
            # Cancel background tasks
            if self.analysis_task and not self.analysis_task.done():
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Close WebSocket connections
            for client in list(self.websocket_clients):
                try:
                    await client.close()
                except Exception:
                    pass
            
            self.websocket_clients.clear()
            
            logger.info("RealTimeIntelligenceEngine shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

