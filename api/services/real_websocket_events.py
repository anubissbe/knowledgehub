"""
Real WebSocket Events System for KnowledgeHub.

This service provides real-time event broadcasting for:
- Memory operations (create, update, search)
- Session management (start, handoff, restore)
- Error learning (pattern detected, solution applied)
- Decision tracking (choice made, outcome recorded)
- Performance alerts and system status
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from ..websocket.manager import websocket_manager, WebSocketMessage, MessageType
from ..services.cache import redis_client
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("websocket_events")


class EventType(str, Enum):
    """WebSocket event types for KnowledgeHub."""
    # Memory events
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_CLUSTERED = "memory_clustered"
    
    # Session events
    SESSION_STARTED = "session_started"
    SESSION_RESUMED = "session_resumed"
    SESSION_HANDOFF = "session_handoff"
    SESSION_COMPLETED = "session_completed"
    
    # Error learning events
    ERROR_DETECTED = "error_detected"
    ERROR_PATTERN_LEARNED = "error_pattern_learned"
    SOLUTION_APPLIED = "solution_applied"
    SOLUTION_VALIDATED = "solution_validated"
    
    # Decision events
    DECISION_MADE = "decision_made"
    DECISION_OUTCOME = "decision_outcome"
    ALTERNATIVE_EVALUATED = "alternative_evaluated"
    
    # AI Intelligence events
    PATTERN_RECOGNIZED = "pattern_recognized"
    PREDICTION_MADE = "prediction_made"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    LEARNING_ADAPTED = "learning_adapted"
    
    # Performance events
    PERFORMANCE_ALERT = "performance_alert"
    METRIC_THRESHOLD_EXCEEDED = "metric_threshold_exceeded"
    SYSTEM_STATUS_CHANGED = "system_status_changed"
    
    # Search events
    SEARCH_EXECUTED = "search_executed"
    SEMANTIC_MATCH_FOUND = "semantic_match_found"
    
    # Code evolution events
    CODE_CHANGED = "code_changed"
    REFACTORING_SUGGESTED = "refactoring_suggested"
    QUALITY_IMPROVED = "quality_improved"


@dataclass
class EventData:
    """Base event data structure."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}


class RealWebSocketEvents:
    """
    Real-time WebSocket events system for KnowledgeHub.
    
    Features:
    - Type-safe event broadcasting
    - Channel-based subscriptions
    - Event filtering and routing
    - Performance monitoring
    - Event persistence and replay
    - Integration with Redis pub/sub
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Event channels
        self.channels = {
            # User-specific channels
            "user": "user.{user_id}",
            "session": "session.{session_id}",
            "project": "project.{project_id}",
            
            # Global channels
            "memories": "memories",
            "errors": "errors",
            "decisions": "decisions",
            "ai": "ai_intelligence",
            "performance": "performance",
            "system": "system",
            "search": "search",
            "code": "code_evolution",
            
            # Admin channels
            "admin": "admin",
            "monitoring": "monitoring",
            "alerts": "alerts"
        }
        
        # Event statistics
        self.stats = {
            "events_published": 0,
            "events_delivered": 0,
            "channels_active": 0,
            "subscribers_total": 0,
            "delivery_failures": 0,
            "processing_time_total": 0.0
        }
        
        # Redis pub/sub for cross-service events
        self._redis_pubsub = None
        self._redis_listener_task = None
        
        # Event persistence
        self.persist_events = True
        self.event_retention_hours = 24
        
        # Running state
        self._running = False
        
        logger.info("Initialized RealWebSocketEvents")
    
    async def start(self):
        """Start the WebSocket events system."""
        if self._running:
            logger.warning("WebSocket events already running")
            return
        
        try:
            # Start WebSocket manager if not running
            await websocket_manager.start()
            
            # Initialize Redis for pub/sub
            await redis_client.initialize()
            
            # Set up Redis pub/sub listener
            await self._setup_redis_pubsub()
            
            self._running = True
            
            logger.info("Real WebSocket events system started")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket events: {e}")
            self._running = False
            raise
    
    async def stop(self):
        """Stop the WebSocket events system."""
        logger.info("Stopping WebSocket events system")
        self._running = False
        
        # Stop Redis listener
        if self._redis_listener_task:
            self._redis_listener_task.cancel()
        
        # Close Redis pub/sub
        if self._redis_pubsub:
            await self._redis_pubsub.close()
        
        logger.info("WebSocket events system stopped")
    
    # Memory Events
    
    async def publish_memory_created(
        self,
        memory_id: str,
        user_id: str,
        session_id: str,
        content: str,
        memory_type: str,
        relevance_score: float = 1.0,
        **kwargs
    ):
        """Publish memory created event."""
        event = EventData(
            event_type=EventType.MEMORY_CREATED,
            user_id=user_id,
            session_id=session_id,
            project_id=kwargs.get("project_id"),
            data={
                "memory_id": memory_id,
                "content": content[:200],  # Truncate for broadcasting
                "memory_type": memory_type,
                "relevance_score": relevance_score,
                "full_content_available": len(content) > 200
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["memories", "user", "session"])
    
    async def publish_memory_accessed(
        self,
        memory_id: str,
        user_id: str,
        access_method: str,
        search_query: Optional[str] = None,
        **kwargs
    ):
        """Publish memory accessed event."""
        event = EventData(
            event_type=EventType.MEMORY_ACCESSED,
            user_id=user_id,
            data={
                "memory_id": memory_id,
                "access_method": access_method,
                "search_query": search_query,
                "access_count": kwargs.get("access_count", 1)
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["memories", "user"])
    
    async def publish_memory_clustered(
        self,
        memory_ids: List[str],
        cluster_id: str,
        cluster_name: str,
        similarity_score: float,
        **kwargs
    ):
        """Publish memory clustering event."""
        event = EventData(
            event_type=EventType.MEMORY_CLUSTERED,
            data={
                "memory_ids": memory_ids,
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "similarity_score": similarity_score,
                "cluster_size": len(memory_ids)
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["memories", "ai"])
    
    # Session Events
    
    async def publish_session_started(
        self,
        session_id: str,
        user_id: str,
        project_id: Optional[str] = None,
        session_type: str = "interactive",
        **kwargs
    ):
        """Publish session started event."""
        event = EventData(
            event_type=EventType.SESSION_STARTED,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            data={
                "session_type": session_type,
                "start_time": datetime.utcnow().isoformat(),
                "context_window_size": kwargs.get("context_window_size", 0)
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["session", "user", "project"])
    
    async def publish_session_handoff(
        self,
        source_session_id: str,
        target_session_id: str,
        user_id: str,
        handoff_reason: str,
        context_data: Dict[str, Any],
        **kwargs
    ):
        """Publish session handoff event."""
        event = EventData(
            event_type=EventType.SESSION_HANDOFF,
            user_id=user_id,
            session_id=target_session_id,
            data={
                "source_session_id": source_session_id,
                "target_session_id": target_session_id,
                "handoff_reason": handoff_reason,
                "context_summary": context_data.get("summary", ""),
                "memory_count": len(context_data.get("memory_ids", []))
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["session", "user"])
    
    # Error Learning Events
    
    async def publish_error_detected(
        self,
        error_type: str,
        error_message: str,
        user_id: str,
        session_id: str,
        context: Dict[str, Any],
        **kwargs
    ):
        """Publish error detected event."""
        event = EventData(
            event_type=EventType.ERROR_DETECTED,
            user_id=user_id,
            session_id=session_id,
            data={
                "error_type": error_type,
                "error_message": error_message[:500],  # Truncate
                "context": context,
                "timestamp": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["errors", "user", "session"])
    
    async def publish_error_pattern_learned(
        self,
        pattern_id: str,
        error_type: str,
        solution: str,
        confidence_score: float,
        occurrences: int,
        **kwargs
    ):
        """Publish error pattern learned event."""
        event = EventData(
            event_type=EventType.ERROR_PATTERN_LEARNED,
            data={
                "pattern_id": pattern_id,
                "error_type": error_type,
                "solution": solution[:300],  # Truncate
                "confidence_score": confidence_score,
                "occurrences": occurrences,
                "learning_timestamp": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["errors", "ai"])
    
    async def publish_solution_applied(
        self,
        solution_id: str,
        error_pattern_id: str,
        user_id: str,
        session_id: str,
        success: bool,
        **kwargs
    ):
        """Publish solution applied event."""
        event = EventData(
            event_type=EventType.SOLUTION_APPLIED,
            user_id=user_id,
            session_id=session_id,
            data={
                "solution_id": solution_id,
                "error_pattern_id": error_pattern_id,
                "success": success,
                "application_time": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["errors", "user", "session"])
    
    # Decision Events
    
    async def publish_decision_made(
        self,
        decision_id: str,
        user_id: str,
        session_id: str,
        decision_type: str,
        choice: str,
        alternatives: List[str],
        reasoning: str,
        **kwargs
    ):
        """Publish decision made event."""
        event = EventData(
            event_type=EventType.DECISION_MADE,
            user_id=user_id,
            session_id=session_id,
            data={
                "decision_id": decision_id,
                "decision_type": decision_type,
                "choice": choice,
                "alternatives": alternatives,
                "reasoning": reasoning[:300],  # Truncate
                "alternative_count": len(alternatives)
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["decisions", "user", "session"])
    
    async def publish_decision_outcome(
        self,
        decision_id: str,
        outcome: str,
        effectiveness_score: float,
        lessons_learned: str,
        **kwargs
    ):
        """Publish decision outcome event."""
        event = EventData(
            event_type=EventType.DECISION_OUTCOME,
            data={
                "decision_id": decision_id,
                "outcome": outcome,
                "effectiveness_score": effectiveness_score,
                "lessons_learned": lessons_learned[:200],
                "evaluation_time": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["decisions", "ai"])
    
    # AI Intelligence Events
    
    async def publish_pattern_recognized(
        self,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        confidence: float,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Publish pattern recognition event."""
        event = EventData(
            event_type=EventType.PATTERN_RECOGNIZED,
            user_id=user_id,
            data={
                "pattern_type": pattern_type,
                "pattern_data": pattern_data,
                "confidence": confidence,
                "recognition_time": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["ai", "user"] if user_id else ["ai"])
    
    async def publish_prediction_made(
        self,
        prediction_type: str,
        prediction: str,
        confidence: float,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Publish AI prediction event."""
        event = EventData(
            event_type=EventType.PREDICTION_MADE,
            user_id=user_id,
            data={
                "prediction_type": prediction_type,
                "prediction": prediction,
                "confidence": confidence,
                "context": context,
                "prediction_time": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["ai", "user"] if user_id else ["ai"])
    
    # Performance Events
    
    async def publish_performance_alert(
        self,
        alert_type: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        severity: str = "warning",
        **kwargs
    ):
        """Publish performance alert event."""
        event = EventData(
            event_type=EventType.PERFORMANCE_ALERT,
            data={
                "alert_type": alert_type,
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold": threshold,
                "severity": severity,
                "alert_time": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["performance", "alerts", "monitoring"])
    
    async def publish_system_status_changed(
        self,
        component: str,
        old_status: str,
        new_status: str,
        details: Dict[str, Any],
        **kwargs
    ):
        """Publish system status change event."""
        event = EventData(
            event_type=EventType.SYSTEM_STATUS_CHANGED,
            data={
                "component": component,
                "old_status": old_status,
                "new_status": new_status,
                "details": details,
                "change_time": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["system", "monitoring"])
    
    # Search Events
    
    async def publish_search_executed(
        self,
        search_query: str,
        search_type: str,
        results_count: int,
        user_id: str,
        processing_time: float,
        **kwargs
    ):
        """Publish search executed event."""
        event = EventData(
            event_type=EventType.SEARCH_EXECUTED,
            user_id=user_id,
            data={
                "search_query": search_query,
                "search_type": search_type,
                "results_count": results_count,
                "processing_time": processing_time,
                "search_time": datetime.utcnow().isoformat()
            },
            metadata=kwargs
        )
        
        await self._publish_event(event, ["search", "user"])
    
    # Utility Methods
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Get WebSocket events statistics."""
        return {
            **self.stats,
            "channels_available": list(self.channels.keys()),
            "websocket_connections": websocket_manager.get_manager_stats()["active_connections"],
            "running": self._running
        }
    
    async def subscribe_to_channel(
        self,
        connection_id: str,
        channel: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Subscribe connection to event channel."""
        return await websocket_manager.subscribe(connection_id, channel, filters)
    
    async def unsubscribe_from_channel(
        self,
        connection_id: str,
        channel: str
    ) -> bool:
        """Unsubscribe connection from event channel."""
        return await websocket_manager.unsubscribe(connection_id, channel)
    
    # Private Methods
    
    async def _publish_event(
        self,
        event: EventData,
        channel_names: List[str]
    ):
        """Publish event to specified channels."""
        start_time = time.time()
        
        try:
            # Create WebSocket message
            ws_message = WebSocketMessage(
                type=MessageType.DATA,
                data={
                    "event": asdict(event),
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat()
                }
            )
            
            total_sent = 0
            
            # Broadcast to each channel
            for channel_name in channel_names:
                channel = self._resolve_channel(channel_name, event)
                if channel:
                    sent_count = await websocket_manager.broadcast_to_channel(
                        channel, ws_message
                    )
                    total_sent += sent_count
            
            # Persist event if enabled
            if self.persist_events:
                await self._persist_event(event)
            
            # Publish to Redis for cross-service communication
            await self._publish_to_redis(event)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["events_published"] += 1
            self.stats["events_delivered"] += total_sent
            self.stats["processing_time_total"] += processing_time
            
            logger.debug(
                f"Published {event.event_type.value} to {len(channel_names)} "
                f"channels, delivered to {total_sent} connections"
            )
            
        except Exception as e:
            self.stats["delivery_failures"] += 1
            logger.error(f"Failed to publish event {event.event_type.value}: {e}")
    
    def _resolve_channel(self, channel_name: str, event: EventData) -> Optional[str]:
        """Resolve channel name with event context."""
        if channel_name not in self.channels:
            return channel_name  # Use as-is if not templated
        
        channel_template = self.channels[channel_name]
        
        try:
            # Replace placeholders with event data
            channel = channel_template.format(
                user_id=event.user_id,
                session_id=event.session_id,
                project_id=event.project_id
            )
            return channel
        except (KeyError, AttributeError):
            # Return base channel if template variables missing
            return channel_name
    
    async def _persist_event(self, event: EventData):
        """Persist event to Redis for replay."""
        try:
            event_key = f"event:{event.event_id}"
            event_data = asdict(event)
            event_data["timestamp"] = event.timestamp.isoformat()
            
            # Store with expiration
            expire_seconds = self.event_retention_hours * 3600
            await redis_client.setex(
                event_key,
                expire_seconds,
                json.dumps(event_data)
            )
            
        except Exception as e:
            logger.debug(f"Event persistence failed: {e}")
    
    async def _publish_to_redis(self, event: EventData):
        """Publish event to Redis pub/sub for cross-service communication."""
        try:
            redis_channel = f"knowledgehub:events:{event.event_type.value}"
            event_data = asdict(event)
            event_data["timestamp"] = event.timestamp.isoformat()
            
            await redis_client.publish(redis_channel, json.dumps(event_data))
            
        except Exception as e:
            logger.debug(f"Redis publish failed: {e}")
    
    async def _setup_redis_pubsub(self):
        """Set up Redis pub/sub listener for external events."""
        try:
            self._redis_pubsub = redis_client.pubsub()
            await self._redis_pubsub.subscribe("knowledgehub:events:*")
            
            # Start listener task
            self._redis_listener_task = asyncio.create_task(
                self._redis_listener()
            )
            
        except Exception as e:
            logger.warning(f"Redis pub/sub setup failed: {e}")
    
    async def _redis_listener(self):
        """Listen for Redis pub/sub events."""
        while self._running:
            try:
                message = await self._redis_pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    # Process external event
                    await self._process_external_event(message)
                    
            except Exception as e:
                logger.debug(f"Redis listener error: {e}")
                await asyncio.sleep(1)
    
    async def _process_external_event(self, message):
        """Process event from external service."""
        try:
            event_data = json.loads(message["data"])
            logger.debug(f"Received external event: {event_data.get('event_type')}")
            
            # Forward to WebSocket clients if relevant
            # Implementation depends on external event format
            
        except Exception as e:
            logger.debug(f"External event processing failed: {e}")


# Global WebSocket events instance
real_websocket_events = RealWebSocketEvents()


# Convenience functions

async def start_websocket_events():
    """Start the real WebSocket events system."""
    await real_websocket_events.start()


async def stop_websocket_events():
    """Stop the real WebSocket events system."""
    await real_websocket_events.stop()


# Event publishing shortcuts

async def notify_memory_created(memory_id: str, user_id: str, content: str, **kwargs):
    """Shortcut to notify memory creation."""
    await real_websocket_events.publish_memory_created(
        memory_id, user_id, kwargs.get("session_id", ""), content, 
        kwargs.get("memory_type", "general"), **kwargs
    )


async def notify_error_learned(error_type: str, solution: str, confidence: float, **kwargs):
    """Shortcut to notify error pattern learning."""
    await real_websocket_events.publish_error_pattern_learned(
        str(uuid.uuid4()), error_type, solution, confidence, 
        kwargs.get("occurrences", 1), **kwargs
    )


async def notify_decision_made(user_id: str, choice: str, reasoning: str, **kwargs):
    """Shortcut to notify decision making."""
    await real_websocket_events.publish_decision_made(
        str(uuid.uuid4()), user_id, kwargs.get("session_id", ""),
        kwargs.get("decision_type", "general"), choice, 
        kwargs.get("alternatives", []), reasoning, **kwargs
    )


async def notify_performance_issue(metric: str, value: float, threshold: float, **kwargs):
    """Shortcut to notify performance issues."""
    await real_websocket_events.publish_performance_alert(
        "threshold_exceeded", metric, value, threshold, 
        kwargs.get("severity", "warning"), **kwargs
    )