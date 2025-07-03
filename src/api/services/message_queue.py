"""Message queue service using Redis Streams"""

import redis.asyncio as redis
import json
import asyncio
from typing import Optional, Dict, Any, List, Callable
import logging
from datetime import datetime
import uuid

from ..config import settings

logger = logging.getLogger(__name__)


class MessageQueue:
    """Redis Streams based message queue"""
    
    def __init__(self, url: str):
        self.url = url
        self.client: Optional[redis.Redis] = None
        self.consumer_group = "knowledge-hub"
        self.consumer_name = f"worker-{uuid.uuid4().hex[:8]}"
        self.handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._tasks = []
    
    async def initialize(self):
        """Initialize Redis connection and consumer groups"""
        try:
            self.client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            if self.client is None:
                raise Exception("Redis client not initialized")
            await self.client.ping()
            logger.info("Message queue connection established")
            
            # Create consumer groups for known streams
            streams = ["scraping_jobs", "rag_processing", "notifications"]
            for stream in streams:
                try:
                    if self.client is None:
                        raise Exception("Redis client not initialized")
                    await self.client.xgroup_create(
                        stream, self.consumer_group, id="0", mkstream=True
                    )
                    logger.info(f"Created consumer group for stream: {stream}")
                except redis.ResponseError as e:
                    # Group already exists
                    if "BUSYGROUP" in str(e):
                        logger.debug(f"Consumer group already exists for stream: {stream}")
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"Failed to initialize message queue: {e}")
            raise
    
    async def publish(self, stream: str, message: str) -> str:
        """Publish message to a stream"""
        try:
            # Add metadata
            data = {
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "publisher": self.consumer_name
            }
            
            # Add to stream
            if self.client is None:
                raise Exception("Redis client not initialized")
            message_id = await self.client.xadd(stream, data)
            logger.debug(f"Published to {stream}: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise
    
    async def subscribe(self, stream: str, handler: Callable):
        """Subscribe to a stream with a handler function"""
        if stream not in self.handlers:
            self.handlers[stream] = []
        self.handlers[stream].append(handler)
        logger.info(f"Subscribed to stream: {stream}")
    
    async def start_consumers(self):
        """Start consuming messages from subscribed streams"""
        if not self.handlers:
            logger.warning("No handlers registered, not starting consumers")
            return
        
        self._running = True
        
        # Start a consumer task for each stream
        for stream in self.handlers:
            task = asyncio.create_task(self._consume_stream(stream))
            self._tasks.append(task)
        
        logger.info(f"Started {len(self._tasks)} consumer tasks")
    
    async def _consume_stream(self, stream: str):
        """Consume messages from a single stream"""
        logger.info(f"Starting consumer for stream: {stream}")
        
        while self._running:
            try:
                # Read messages from stream
                if self.client is None:
                    raise Exception("Redis client not initialized")
                messages = await self.client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {stream: ">"},  # Read only new messages
                    count=10,
                    block=1000  # Block for 1 second
                )
                
                if not messages:
                    continue
                
                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        try:
                            # Extract message content
                            message_content = data.get("message", "{}")
                            
                            # Call handlers
                            for handler in self.handlers[stream]:
                                await handler(message_content)
                            
                            # Acknowledge message
                            if self.client is not None:
                                await self.client.xack(stream, self.consumer_group, message_id)
                            
                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            # Message will be redelivered
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error consuming from {stream}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def get(self, stream: str) -> Optional[str]:
        """Get a single message from a stream (for simple consumers)"""
        try:
            if self.client is None:
                raise Exception("Redis client not initialized")
            messages = await self.client.xreadgroup(
                self.consumer_group,
                self.consumer_name,
                {stream: ">"},
                count=1,
                block=1000
            )
            
            if not messages:
                return None
            
            for stream_name, stream_messages in messages:
                for message_id, data in stream_messages:
                    # Acknowledge immediately
                    if self.client is not None:
                        await self.client.xack(stream, self.consumer_group, message_id)
                    return data.get("message")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get message from {stream}: {e}")
            return None
    
    async def close(self):
        """Close message queue and stop consumers"""
        self._running = False
        
        # Cancel all consumer tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.client:
            if self.client is not None:
                await self.client.close()
        
        logger.info("Message queue closed")


# Global message queue instance
message_queue = MessageQueue(settings.REDIS_URL)