"""
Real-time Event Streaming System for WebSocket Communications.

This system provides:
- Event-driven architecture
- Real-time data streaming
- Event filtering and routing
- Performance monitoring
- Integration with metrics and alerts
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from .manager import (
    websocket_manager, WebSocketMessage, MessageType,
    broadcast_metric_update, broadcast_alert, broadcast_system_status
)
from ..services.metrics_service import metrics_service
from ..workers.metrics_collector import metrics_collector_worker
from ..models.memory import MemoryItem
from ..models.session import Session
from ..models.error_pattern import ErrorOccurrence
from ..models.workflow import WorkflowExecution
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("websocket_events")


class EventType(str, Enum):
    """Types of real-time events."""
    # Metrics events
    METRIC_UPDATE = "metric_update"
    METRIC_THRESHOLD = "metric_threshold"
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_RESOLVED = "alert_resolved"
    
    # System events
    SYSTEM_STATUS = "system_status"
    SERVICE_STATUS = "service_status"
    PERFORMANCE_UPDATE = "performance_update"
    
    # Memory system events
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    
    # Session events
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    SESSION_UPDATED = "session_updated"
    
    # Error events
    ERROR_OCCURRED = "error_occurred"
    ERROR_RESOLVED = "error_resolved"
    
    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    
    # Analytics events
    DASHBOARD_UPDATE = "dashboard_update"
    TREND_CHANGE = "trend_change"
    REPORT_READY = "report_ready"
    
    # User events
    USER_ACTIVITY = "user_activity"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StreamEvent:
    """Real-time stream event."""
    event_type: EventType
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    event_id: str = None
    source: str = "system"
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        if self.tags is None:
            self.tags = {}


@dataclass
class EventSubscription:
    """Event subscription configuration."""
    channel: str
    event_types: Set[EventType]
    filters: Dict[str, Any] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


class EventStreamer:
    """
    Real-time event streaming system.
    
    Features:
    - Event publishing and routing
    - Real-time WebSocket streaming
    - Event filtering and subscriptions
    - Performance monitoring
    - Integration with all system components
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Event management
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        self._event_filters: Dict[str, Callable] = {}
        self._event_queue = asyncio.Queue()
        self._processed_events = {}
        
        # Subscription management
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._channel_subscriptions: Dict[str, List[str]] = {}
        
        # Performance tracking
        self._stats = {
            "events_published": 0,
            "events_streamed": 0,
            "active_subscriptions": 0,
            "processing_errors": 0
        }
        
        # Configuration
        self.max_queue_size = 10000
        self.event_retention_hours = 24
        self.batch_size = 100
        self.processing_interval = 0.1
        
        # Background tasks
        self._running = False
        self._processing_task = None
        self._cleanup_task = None
        
        logger.info("Initialized EventStreamer")
    
    async def start(self):
        """Start the event streaming system."""
        if self._running:
            logger.warning("Event streamer already running")
            return
        
        try:
            self._running = True
            
            # Start background tasks
            self._processing_task = asyncio.create_task(self._event_processing_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Register default event handlers
            await self._register_default_handlers()
            
            logger.info("Event streamer started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start event streamer: {e}")
            self._running = False
            raise
    
    async def stop(self):
        """Stop the event streaming system."""
        logger.info("Stopping event streamer")
        self._running = False
        
        # Cancel background tasks
        if self._processing_task:
            self._processing_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Process remaining events
        remaining = self._event_queue.qsize()
        if remaining > 0:
            logger.info(f"Processing {remaining} remaining events")
            while not self._event_queue.empty():
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    await self._process_event(event)
                except asyncio.TimeoutError:
                    break
        
        logger.info("Event streamer stopped")
    
    async def publish_event(self, event: StreamEvent):
        """Publish an event to the streaming system."""
        try:
            # Check queue size
            if self._event_queue.qsize() >= self.max_queue_size:
                logger.warning("Event queue full, dropping event")
                return
            
            # Add to queue
            await self._event_queue.put(event)
            
            # Update stats
            self._stats["events_published"] += 1
            
            # Log high priority events immediately
            if event.priority in [EventPriority.HIGH, EventPriority.CRITICAL]:
                logger.info(f"High priority event published: {event.event_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
    
    async def subscribe_to_events(
        self,
        subscription_id: str,
        channel: str,
        event_types: List[EventType],
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """Subscribe to specific events."""
        try:
            subscription = EventSubscription(
                channel=channel,
                event_types=set(event_types),
                filters=filters or {},
                user_id=user_id,
                project_id=project_id
            )
            
            self._subscriptions[subscription_id] = subscription
            
            # Add to channel subscriptions
            if channel not in self._channel_subscriptions:
                self._channel_subscriptions[channel] = []
            self._channel_subscriptions[channel].append(subscription_id)
            
            self._stats["active_subscriptions"] = len(self._subscriptions)
            
            logger.info(f"Event subscription created: {subscription_id} -> {channel}")
            
        except Exception as e:
            logger.error(f"Failed to create event subscription: {e}")
    
    async def unsubscribe_from_events(self, subscription_id: str):
        """Unsubscribe from events."""
        try:
            if subscription_id not in self._subscriptions:
                return
            
            subscription = self._subscriptions[subscription_id]
            
            # Remove from channel subscriptions
            if subscription.channel in self._channel_subscriptions:
                self._channel_subscriptions[subscription.channel] = [
                    sid for sid in self._channel_subscriptions[subscription.channel]
                    if sid != subscription_id
                ]
                
                # Remove empty channel
                if not self._channel_subscriptions[subscription.channel]:
                    del self._channel_subscriptions[subscription.channel]
            
            # Remove subscription
            del self._subscriptions[subscription_id]
            
            self._stats["active_subscriptions"] = len(self._subscriptions)
            
            logger.info(f"Event subscription removed: {subscription_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove event subscription: {e}")
    
    def register_event_handler(
        self,
        event_type: EventType,
        handler: Callable[[StreamEvent], None]
    ):
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered event handler for: {event_type.value}")
    
    def register_event_filter(
        self,
        filter_name: str,
        filter_func: Callable[[StreamEvent], bool]
    ):
        """Register an event filter."""
        self._event_filters[filter_name] = filter_func
        logger.info(f"Registered event filter: {filter_name}")
    
    async def get_event_stats(self) -> Dict[str, Any]:
        """Get event streaming statistics."""
        return {
            **self._stats,
            "queue_size": self._event_queue.qsize(),
            "active_handlers": sum(len(handlers) for handlers in self._event_handlers.values()),
            "active_filters": len(self._event_filters),
            "channels": list(self._channel_subscriptions.keys())
        }
    
    # Event publishing convenience methods
    
    async def publish_metric_event(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None
    ):
        """Publish a metric update event."""
        event = StreamEvent(
            event_type=EventType.METRIC_UPDATE,
            data={
                "metric_name": metric_name,
                "value": value,
                "tags": tags or {}
            },
            user_id=user_id,
            source="metrics_service"
        )
        await self.publish_event(event)
    
    async def publish_alert_event(
        self,
        alert_data: Dict[str, Any],
        priority: EventPriority = EventPriority.HIGH
    ):
        """Publish an alert event."""
        event = StreamEvent(
            event_type=EventType.ALERT_TRIGGERED,
            data=alert_data,
            priority=priority,
            source="alerts_service"
        )
        await self.publish_event(event)
    
    async def publish_system_event(
        self,
        status_data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL
    ):
        """Publish a system status event."""
        event = StreamEvent(
            event_type=EventType.SYSTEM_STATUS,
            data=status_data,
            priority=priority,
            source="system_monitor"
        )
        await self.publish_event(event)
    
    async def publish_memory_event(
        self,
        event_type: EventType,
        memory_item: MemoryItem,
        user_id: str
    ):
        """Publish a memory system event."""
        event = StreamEvent(
            event_type=event_type,
            data={
                "memory_id": str(memory_item.id),
                "memory_type": memory_item.memory_type,
                "content_preview": memory_item.content[:100] + "..." if len(memory_item.content) > 100 else memory_item.content,
                "created_at": memory_item.created_at.isoformat()
            },
            user_id=user_id,
            project_id=memory_item.project_id,
            source="memory_service"
        )
        await self.publish_event(event)
    
    async def publish_session_event(
        self,
        event_type: EventType,
        session: Session,
        user_id: str
    ):
        """Publish a session event."""
        event = StreamEvent(
            event_type=event_type,
            data={
                "session_id": str(session.id),
                "session_type": session.session_type,
                "started_at": session.started_at.isoformat(),
                "is_active": session.is_active
            },
            user_id=user_id,
            session_id=str(session.id),
            source="session_service"
        )
        await self.publish_event(event)
    
    async def publish_error_event(
        self,
        error_occurrence: ErrorOccurrence,
        priority: EventPriority = EventPriority.HIGH
    ):
        """Publish an error event."""
        event = StreamEvent(
            event_type=EventType.ERROR_OCCURRED,
            data={
                "error_id": str(error_occurrence.id),
                "error_type": error_occurrence.error_type,
                "severity": error_occurrence.severity,
                "message": error_occurrence.error_message,
                "occurred_at": error_occurrence.occurred_at.isoformat()
            },
            user_id=error_occurrence.user_id,
            priority=priority,
            source="error_tracking"
        )
        await self.publish_event(event)
    
    async def publish_workflow_event(
        self,
        event_type: EventType,
        workflow_execution: WorkflowExecution,
        user_id: str
    ):
        """Publish a workflow event."""
        event = StreamEvent(
            event_type=event_type,
            data={
                "execution_id": str(workflow_execution.id),
                "execution_name": workflow_execution.execution_name,
                "template_id": str(workflow_execution.template_id),
                "success": workflow_execution.success,
                "execution_time": workflow_execution.execution_time,
                "created_at": workflow_execution.created_at.isoformat()
            },
            user_id=user_id,
            project_id=workflow_execution.project_id,
            session_id=workflow_execution.session_id,
            source="workflow_service"
        )
        await self.publish_event(event)
    
    async def publish_dashboard_update(
        self,
        dashboard_data: Dict[str, Any],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """Publish a dashboard update event."""
        event = StreamEvent(
            event_type=EventType.DASHBOARD_UPDATE,
            data=dashboard_data,
            user_id=user_id,
            project_id=project_id,
            source="analytics_service"
        )
        await self.publish_event(event)
    
    # Internal methods
    
    async def _event_processing_loop(self):
        """Main event processing loop."""
        while self._running:
            try:
                # Process events in batches
                events = []
                deadline = asyncio.get_event_loop().time() + self.processing_interval
                
                while (len(events) < self.batch_size and 
                       asyncio.get_event_loop().time() < deadline):
                    try:
                        event = await asyncio.wait_for(
                            self._event_queue.get(),
                            timeout=0.01
                        )
                        events.append(event)
                    except asyncio.TimeoutError:
                        break
                
                # Process collected events
                for event in events:
                    await self._process_event(event)
                
                # Small delay if no events
                if not events:
                    await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                self._stats["processing_errors"] += 1
                await asyncio.sleep(1)
    
    async def _process_event(self, event: StreamEvent):
        """Process a single event."""
        try:
            # Store event for retention
            self._processed_events[event.event_id] = {
                "event": event,
                "processed_at": datetime.utcnow()
            }
            
            # Apply event filters
            for filter_name, filter_func in self._event_filters.items():
                if not filter_func(event):
                    logger.debug(f"Event filtered out by {filter_name}: {event.event_type.value}")
                    return
            
            # Call event handlers
            if event.event_type in self._event_handlers:
                for handler in self._event_handlers[event.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Event handler failed: {e}")
            
            # Stream to WebSocket subscribers
            await self._stream_to_websockets(event)
            
            # Update stats
            self._stats["events_streamed"] += 1
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id}: {e}")
            self._stats["processing_errors"] += 1
    
    async def _stream_to_websockets(self, event: StreamEvent):
        """Stream event to WebSocket subscribers."""
        try:
            # Create WebSocket message
            ws_message = WebSocketMessage(
                type=MessageType.DATA,
                data={
                    "event_type": event.event_type.value,
                    "event_id": event.event_id,
                    "data": event.data,
                    "priority": event.priority.value,
                    "source": event.source,
                    "timestamp": event.timestamp.isoformat(),
                    "tags": event.tags
                }
            )
            
            # Broadcast to relevant channels
            channels_to_broadcast = self._get_relevant_channels(event)
            
            for channel in channels_to_broadcast:
                await websocket_manager.broadcast_to_channel(channel, ws_message)
            
            # Broadcast to user-specific channels
            if event.user_id:
                user_channel = f"user.{event.user_id}"
                await websocket_manager.broadcast_to_channel(user_channel, ws_message)
            
            # Broadcast to project-specific channels
            if event.project_id:
                project_channel = f"project.{event.project_id}"
                await websocket_manager.broadcast_to_channel(project_channel, ws_message)
            
        except Exception as e:
            logger.error(f"Failed to stream event to WebSockets: {e}")
    
    def _get_relevant_channels(self, event: StreamEvent) -> List[str]:
        """Get relevant channels for an event."""
        channels = []
        
        # Event type channels
        if event.event_type in [EventType.METRIC_UPDATE, EventType.METRIC_THRESHOLD]:
            channels.append("metrics")
        
        if event.event_type in [EventType.ALERT_TRIGGERED, EventType.ALERT_RESOLVED]:
            channels.append("alerts")
        
        if event.event_type in [EventType.SYSTEM_STATUS, EventType.SERVICE_STATUS]:
            channels.append("system")
        
        if event.event_type in [EventType.MEMORY_CREATED, EventType.MEMORY_UPDATED, EventType.MEMORY_DELETED]:
            channels.append("memory")
        
        if event.event_type in [EventType.SESSION_STARTED, EventType.SESSION_ENDED, EventType.SESSION_UPDATED]:
            channels.append("sessions")
        
        if event.event_type in [EventType.ERROR_OCCURRED, EventType.ERROR_RESOLVED]:
            channels.append("errors")
        
        if event.event_type in [EventType.WORKFLOW_STARTED, EventType.WORKFLOW_COMPLETED, EventType.WORKFLOW_FAILED]:
            channels.append("workflows")
        
        if event.event_type in [EventType.DASHBOARD_UPDATE, EventType.TREND_CHANGE]:
            channels.append("analytics")
        
        # General events channel
        channels.append("events")
        
        return channels
    
    async def _register_default_handlers(self):
        """Register default event handlers."""
        # Metrics events
        self.register_event_handler(
            EventType.METRIC_UPDATE,
            self._handle_metric_update
        )
        
        # Alert events
        self.register_event_handler(
            EventType.ALERT_TRIGGERED,
            self._handle_alert_triggered
        )
        
        # System events
        self.register_event_handler(
            EventType.SYSTEM_STATUS,
            self._handle_system_status
        )
        
        # Error events
        self.register_event_handler(
            EventType.ERROR_OCCURRED,
            self._handle_error_occurred
        )
    
    async def _handle_metric_update(self, event: StreamEvent):
        """Handle metric update events."""
        try:
            # Broadcast to metrics channel
            await broadcast_metric_update(
                event.data["metric_name"],
                event.data["value"],
                event.data.get("tags", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to handle metric update event: {e}")
    
    async def _handle_alert_triggered(self, event: StreamEvent):
        """Handle alert triggered events."""
        try:
            # Broadcast alert
            await broadcast_alert(event.data)
            
            # Log critical alerts
            if event.priority == EventPriority.CRITICAL:
                logger.critical(f"Critical alert triggered: {event.data}")
            
        except Exception as e:
            logger.error(f"Failed to handle alert event: {e}")
    
    async def _handle_system_status(self, event: StreamEvent):
        """Handle system status events."""
        try:
            # Broadcast system status
            await broadcast_system_status(event.data)
            
        except Exception as e:
            logger.error(f"Failed to handle system status event: {e}")
    
    async def _handle_error_occurred(self, event: StreamEvent):
        """Handle error occurrence events."""
        try:
            # Log errors based on severity
            if event.data.get("severity") == "critical":
                logger.critical(f"Critical error occurred: {event.data['message']}")
            elif event.data.get("severity") == "high":
                logger.error(f"High severity error: {event.data['message']}")
            else:
                logger.warning(f"Error occurred: {event.data['message']}")
            
        except Exception as e:
            logger.error(f"Failed to handle error event: {e}")
    
    async def _cleanup_loop(self):
        """Background loop for event cleanup."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old processed events
                cutoff = datetime.utcnow() - timedelta(hours=self.event_retention_hours)
                
                to_remove = []
                for event_id, event_data in self._processed_events.items():
                    if event_data["processed_at"] < cutoff:
                        to_remove.append(event_id)
                
                for event_id in to_remove:
                    del self._processed_events[event_id]
                
                logger.debug(f"Cleaned up {len(to_remove)} old events")
                
            except Exception as e:
                logger.error(f"Event cleanup error: {e}")


# Global event streamer instance
event_streamer = EventStreamer()


# Convenience functions

async def start_event_streaming():
    """Start the event streaming system."""
    await event_streamer.start()


async def stop_event_streaming():
    """Stop the event streaming system."""
    await event_streamer.stop()


async def publish_metric_update(
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None,
    user_id: Optional[str] = None
):
    """Convenience function to publish metric updates."""
    await event_streamer.publish_metric_event(metric_name, value, tags, user_id)


async def publish_alert(alert_data: Dict[str, Any], priority: EventPriority = EventPriority.HIGH):
    """Convenience function to publish alerts."""
    await event_streamer.publish_alert_event(alert_data, priority)


async def publish_system_status(status_data: Dict[str, Any]):
    """Convenience function to publish system status."""
    await event_streamer.publish_system_event(status_data)


# Integration with existing services

async def integrate_with_metrics_service():
    """Integrate event streaming with metrics service."""
    try:
        # Override metrics service broadcast functions to use event streaming
        original_record_metric = metrics_service.record_metric
        
        async def enhanced_record_metric(name, value, metric_type, tags=None, metadata=None):
            # Record metric normally
            result = await original_record_metric(name, value, metric_type, tags, metadata)
            
            # Publish event
            await publish_metric_update(name, value, tags)
            
            return result
        
        # Replace the method
        metrics_service.record_metric = enhanced_record_metric
        
        logger.info("Integrated event streaming with metrics service")
        
    except Exception as e:
        logger.error(f"Failed to integrate with metrics service: {e}")


async def integrate_with_alert_system():
    """Integrate event streaming with alert system."""
    try:
        # This would integrate with the alert evaluation in metrics service
        # For now, we'll add a placeholder
        logger.info("Alert system integration ready")
        
    except Exception as e:
        logger.error(f"Failed to integrate with alert system: {e}")