"""
Real-time Learning Pipeline with Redis Streams
Provides immediate context updates and streaming event processing
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime, timedelta
import redis.asyncio as redis
from pydantic import BaseModel, Field
import hashlib
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events in the learning pipeline"""
    CODE_CHANGE = "code_change"
    MEMORY_CREATED = "memory_created"
    DECISION_MADE = "decision_made"
    ERROR_OCCURRED = "error_occurred"
    PATTERN_DETECTED = "pattern_detected"
    CONTEXT_UPDATED = "context_updated"
    SESSION_EVENT = "session_event"
    LEARNING_INSIGHT = "learning_insight"


class StreamEvent(BaseModel):
    """Base event model for the pipeline"""
    event_id: str = Field(default_factory=lambda: hashlib.md5(str(datetime.utcnow()).encode()).hexdigest())
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RealtimeLearningPipeline:
    """
    Real-time learning pipeline using Redis Streams for:
    - Immediate event processing
    - Context updates in real-time
    - Pattern detection as events occur
    - Distributed event handling
    """
    
    def __init__(self, redis_url: str = None):
        # Use environment variable or default to container name
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://knowledgehub-redis:6379")
        self.redis_client: Optional[redis.Redis] = None
        self.consumer_group = "knowledgehub-learners"
        self.consumer_name = f"learner-{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}"
        
        # Stream names
        self.streams = {
            "events": "knowledgehub:events",
            "insights": "knowledgehub:insights",
            "patterns": "knowledgehub:patterns",
            "context": "knowledgehub:context"
        }
        
        # Processing state
        self.processing = False
        self.processors = {}
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis for real-time pipeline")
            
            # Create consumer groups
            for stream_name in self.streams.values():
                try:
                    await self.redis_client.xgroup_create(
                        stream_name, 
                        self.consumer_group, 
                        id="0",
                        mkstream=True  # Create stream if it doesn't exist
                    )
                    logger.info(f"Created consumer group for {stream_name}")
                except redis.ResponseError as e:
                    if "BUSYGROUP" in str(e):
                        logger.info(f"Consumer group already exists for {stream_name}")
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def publish_event(self, event: StreamEvent):
        """Publish an event to the appropriate stream"""
        if not self.redis_client:
            await self.connect()
            
        try:
            # Serialize event
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "session_id": event.session_id or "",
                "user_id": event.user_id or "",
                "data": json.dumps(event.data),
                "metadata": json.dumps(event.metadata)
            }
            
            # Publish to main event stream
            stream_id = await self.redis_client.xadd(
                self.streams["events"],
                event_data
            )
            
            # Also publish to type-specific streams for specialized processing
            type_stream = f"{self.streams['events']}:{event.event_type.value}"
            await self.redis_client.xadd(type_stream, event_data)
            
            logger.debug(f"Published event {event.event_id} to stream {stream_id}")
            
            # Trigger immediate processing for critical events
            if event.event_type in [EventType.ERROR_OCCURRED, EventType.DECISION_MADE]:
                await self._process_critical_event(event)
                
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            raise
            
    async def _process_critical_event(self, event: StreamEvent):
        """Process critical events immediately"""
        if event.event_type == EventType.ERROR_OCCURRED:
            # Extract error patterns
            insight = await self._analyze_error_pattern(event.data)
            if insight:
                await self._publish_insight(insight)
                
        elif event.event_type == EventType.DECISION_MADE:
            # Update decision context
            await self._update_decision_context(event.data)
            
    async def consume_events(self) -> AsyncIterator[StreamEvent]:
        """Consume events from the stream"""
        if not self.redis_client:
            await self.connect()
            
        self.processing = True
        
        try:
            while self.processing:
                # Read from multiple streams
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.streams["events"]: ">"},
                    count=10,
                    block=1000  # Block for 1 second
                )
                
                for stream, stream_messages in messages:
                    for message_id, data in stream_messages:
                        try:
                            # Deserialize event
                            event = StreamEvent(
                                event_id=data.get("event_id"),
                                event_type=EventType(data.get("event_type")),
                                timestamp=datetime.fromisoformat(data.get("timestamp")),
                                session_id=data.get("session_id") or None,
                                user_id=data.get("user_id") or None,
                                data=json.loads(data.get("data", "{}")),
                                metadata=json.loads(data.get("metadata", "{}"))
                            )
                            
                            yield event
                            
                            # Acknowledge message
                            await self.redis_client.xack(
                                stream,
                                self.consumer_group,
                                message_id
                            )
                            
                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            
        except Exception as e:
            logger.error(f"Error consuming events: {e}")
            raise
        finally:
            self.processing = False
            
    async def process_event_stream(self):
        """Main event processing loop"""
        async for event in self.consume_events():
            try:
                # Route to appropriate processor
                processor = self.processors.get(event.event_type)
                if processor:
                    await processor(event)
                else:
                    await self._default_processor(event)
                    
            except Exception as e:
                logger.error(f"Error processing event {event.event_id}: {e}")
                
    async def _default_processor(self, event: StreamEvent):
        """Default event processor"""
        logger.debug(f"Processing {event.event_type} event: {event.event_id}")
        
        # Extract patterns
        if event.event_type == EventType.CODE_CHANGE:
            patterns = await self._detect_code_patterns(event.data)
            if patterns:
                await self._publish_patterns(patterns)
                
        # Update context
        await self._update_context(event)
        
    async def _detect_code_patterns(self, code_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect patterns in code changes"""
        patterns = []
        
        # Simple pattern detection (would use ML models in production)
        if "changes" in code_data:
            for change in code_data.get("changes", []):
                # Detect common patterns
                if "import" in change.get("text", ""):
                    patterns.append({
                        "type": "import_pattern",
                        "description": "New import added",
                        "confidence": 0.9
                    })
                elif "async def" in change.get("text", ""):
                    patterns.append({
                        "type": "async_pattern",
                        "description": "Async function defined",
                        "confidence": 0.95
                    })
                    
        return patterns
        
    async def _analyze_error_pattern(self, error_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze error patterns for insights"""
        error_type = error_data.get("error_type", "")
        error_message = error_data.get("error_message", "")
        
        # Check for known patterns
        if "ImportError" in error_type:
            return {
                "type": "missing_dependency",
                "suggestion": "Check if all required packages are installed",
                "confidence": 0.85
            }
        elif "AttributeError" in error_type:
            return {
                "type": "api_change",
                "suggestion": "API might have changed, check documentation",
                "confidence": 0.75
            }
            
        return None
        
    async def _update_context(self, event: StreamEvent):
        """Update real-time context"""
        context_key = f"context:{event.session_id or 'global'}"
        
        # Update context in Redis
        context_data = {
            "last_event_type": event.event_type.value,
            "last_event_time": event.timestamp.isoformat(),
            "event_count": 0
        }
        
        # Get existing context
        existing = await self.redis_client.hgetall(context_key)
        if existing:
            context_data["event_count"] = int(existing.get("event_count", 0)) + 1
            
        # Save updated context
        await self.redis_client.hset(context_key, mapping=context_data)
        await self.redis_client.expire(context_key, 3600)  # Expire after 1 hour
        
    async def _update_decision_context(self, decision_data: Dict[str, Any]):
        """Update decision-making context"""
        decision_key = f"decisions:{decision_data.get('category', 'general')}"
        
        # Store decision for pattern analysis
        await self.redis_client.lpush(
            decision_key,
            json.dumps({
                "decision": decision_data.get("decision"),
                "context": decision_data.get("context"),
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        # Keep only recent decisions
        await self.redis_client.ltrim(decision_key, 0, 99)
        
    async def _publish_patterns(self, patterns: List[Dict[str, Any]]):
        """Publish detected patterns"""
        for pattern in patterns:
            await self.redis_client.xadd(
                self.streams["patterns"],
                {
                    "pattern_type": pattern.get("type", "unknown"),
                    "description": pattern.get("description", ""),
                    "confidence": str(pattern.get("confidence", 0)),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    async def _publish_insight(self, insight: Dict[str, Any]):
        """Publish learning insights"""
        await self.redis_client.xadd(
            self.streams["insights"],
            {
                "insight_type": insight.get("type", "unknown"),
                "suggestion": insight.get("suggestion", ""),
                "confidence": str(insight.get("confidence", 0)),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    async def get_real_time_context(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current real-time context"""
        context_key = f"context:{session_id or 'global'}"
        context = await self.redis_client.hgetall(context_key)
        
        # Get recent patterns
        patterns = await self.redis_client.xrevrange(
            self.streams["patterns"],
            count=10
        )
        
        # Get recent insights
        insights = await self.redis_client.xrevrange(
            self.streams["insights"],
            count=5
        )
        
        return {
            "session_context": context,
            "recent_patterns": [p[1] for p in patterns],
            "recent_insights": [i[1] for i in insights],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def register_processor(self, event_type: EventType, processor):
        """Register a custom event processor"""
        self.processors[event_type] = processor
        logger.info(f"Registered processor for {event_type}")
        
    async def start_background_processing(self):
        """Start background event processing"""
        asyncio.create_task(self.process_event_stream())
        logger.info("Started background event processing")
        
    async def stop_processing(self):
        """Stop event processing"""
        self.processing = False
        logger.info("Stopping event processing")


# Singleton instance
_pipeline_instance: Optional[RealtimeLearningPipeline] = None


async def get_learning_pipeline() -> RealtimeLearningPipeline:
    """Get or create the learning pipeline instance"""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = RealtimeLearningPipeline()
        await _pipeline_instance.connect()
        await _pipeline_instance.start_background_processing()
        
    return _pipeline_instance