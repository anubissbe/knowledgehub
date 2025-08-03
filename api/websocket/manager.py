"""
WebSocket Connection Manager for Real-time Communication.

This manager provides:
- WebSocket connection lifecycle management
- Connection pooling and scaling
- Real-time event broadcasting
- Subscription management
- Authentication and authorization
- Connection health monitoring
"""

import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import jwt

from ..services.cache import redis_client
from ..services.auth import verify_token
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("websocket_manager")


class ConnectionState(str, Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(str, Enum):
    """WebSocket message types."""
    AUTH = "auth"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    HEARTBEAT = "heartbeat"
    NOTIFICATION = "notification"


@dataclass
class WebSocketConnection:
    """WebSocket connection metadata."""
    id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    state: ConnectionState = ConnectionState.CONNECTING
    connected_at: datetime = None
    last_heartbeat: datetime = None
    subscriptions: Set[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.connected_at is None:
            self.connected_at = datetime.utcnow()
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()
        if self.subscriptions is None:
            self.subscriptions = set()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: MessageType
    data: Dict[str, Any]
    target: Optional[str] = None
    source: Optional[str] = None
    timestamp: datetime = None
    message_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())


@dataclass
class Subscription:
    """Subscription metadata."""
    channel: str
    connection_id: str
    user_id: Optional[str] = None
    filters: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.filters is None:
            self.filters = {}


class WebSocketManager:
    """
    Comprehensive WebSocket connection manager.
    
    Features:
    - Connection lifecycle management
    - Authentication and authorization
    - Subscription-based messaging
    - Connection pooling and load balancing
    - Health monitoring and auto-reconnection
    - Message routing and broadcasting
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Connection management
        self._connections: Dict[str, WebSocketConnection] = {}
        self._user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self._subscriptions: Dict[str, List[Subscription]] = {}  # channel -> subscriptions
        self._connection_pool = weakref.WeakValueDictionary()
        
        # Message handling
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._broadcast_queue = asyncio.Queue()
        self._pending_messages: Dict[str, List[WebSocketMessage]] = {}
        
        # Performance and monitoring
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "subscriptions_total": 0,
            "errors_count": 0
        }
        
        # Configuration
        self.max_connections = 1000
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # 5 minutes
        self.max_message_size = 1024 * 1024  # 1MB
        self.auth_timeout = 30  # seconds
        
        # Background tasks
        self._running = False
        self._cleanup_task = None
        self._heartbeat_task = None
        self._broadcast_task = None
        
        # Register default message handlers
        self._register_default_handlers()
        
        logger.info("Initialized WebSocketManager")
    
    async def start(self):
        """Start the WebSocket manager."""
        if self._running:
            logger.warning("WebSocket manager already running")
            return
        
        try:
            await redis_client.initialize()
            
            self._running = True
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())
            
            logger.info("WebSocket manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket manager: {e}")
            self._running = False
            raise
    
    async def stop(self):
        """Stop the WebSocket manager."""
        logger.info("Stopping WebSocket manager")
        self._running = False
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._broadcast_task:
            self._broadcast_task.cancel()
        
        # Close all connections
        await self._close_all_connections()
        
        logger.info("WebSocket manager stopped")
    
    async def connect(self, websocket: WebSocket) -> str:
        """Handle new WebSocket connection."""
        try:
            await websocket.accept()
            
            # Check connection limits
            if len(self._connections) >= self.max_connections:
                await websocket.close(code=1008, reason="Connection limit exceeded")
                raise Exception("Connection limit exceeded")
            
            # Create connection object
            connection_id = str(uuid.uuid4())
            connection = WebSocketConnection(
                id=connection_id,
                websocket=websocket,
                state=ConnectionState.CONNECTED
            )
            
            self._connections[connection_id] = connection
            self._stats["total_connections"] += 1
            self._stats["active_connections"] = len(self._connections)
            
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Start authentication timeout
            asyncio.create_task(self._auth_timeout_handler(connection_id))
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
    
    async def disconnect(self, connection_id: str, reason: str = "Client disconnect"):
        """Handle WebSocket disconnection."""
        try:
            if connection_id not in self._connections:
                return
            
            connection = self._connections[connection_id]
            connection.state = ConnectionState.DISCONNECTING
            
            # Remove from user connections
            if connection.user_id:
                if connection.user_id in self._user_connections:
                    self._user_connections[connection.user_id].discard(connection_id)
                    if not self._user_connections[connection.user_id]:
                        del self._user_connections[connection.user_id]
            
            # Remove subscriptions
            await self._remove_connection_subscriptions(connection_id)
            
            # Close WebSocket if still open
            if connection.websocket.client_state != WebSocketState.DISCONNECTED:
                try:
                    await connection.websocket.close()
                except:
                    pass
            
            # Remove from connections
            del self._connections[connection_id]
            self._stats["active_connections"] = len(self._connections)
            
            logger.info(f"WebSocket connection closed: {connection_id} ({reason})")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def authenticate(self, connection_id: str, token: str) -> bool:
        """Authenticate a WebSocket connection."""
        try:
            if connection_id not in self._connections:
                return False
            
            connection = self._connections[connection_id]
            
            # Verify token
            user_data = await verify_token(token)
            if not user_data:
                return False
            
            # Update connection with user data
            connection.user_id = user_data.get("user_id")
            connection.session_id = user_data.get("session_id")
            connection.project_id = user_data.get("project_id")
            connection.state = ConnectionState.AUTHENTICATED
            
            # Add to user connections
            if connection.user_id:
                if connection.user_id not in self._user_connections:
                    self._user_connections[connection.user_id] = set()
                self._user_connections[connection.user_id].add(connection_id)
            
            # Send authentication success
            await self.send_message(connection_id, WebSocketMessage(
                type=MessageType.AUTH,
                data={"status": "authenticated", "user_id": connection.user_id}
            ))
            
            logger.info(f"WebSocket connection authenticated: {connection_id} (user: {connection.user_id})")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed for connection {connection_id}: {e}")
            return False
    
    async def subscribe(
        self,
        connection_id: str,
        channel: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Subscribe connection to a channel."""
        try:
            if connection_id not in self._connections:
                return False
            
            connection = self._connections[connection_id]
            
            # Check if authenticated for protected channels
            if channel.startswith("user.") and connection.state != ConnectionState.AUTHENTICATED:
                return False
            
            # Create subscription
            subscription = Subscription(
                channel=channel,
                connection_id=connection_id,
                user_id=connection.user_id,
                filters=filters or {}
            )
            
            # Add to subscriptions
            if channel not in self._subscriptions:
                self._subscriptions[channel] = []
            self._subscriptions[channel].append(subscription)
            
            # Add to connection subscriptions
            connection.subscriptions.add(channel)
            
            self._stats["subscriptions_total"] = sum(
                len(subs) for subs in self._subscriptions.values()
            )
            
            # Send subscription confirmation
            await self.send_message(connection_id, WebSocketMessage(
                type=MessageType.SUBSCRIBE,
                data={"channel": channel, "status": "subscribed"}
            ))
            
            logger.debug(f"Connection {connection_id} subscribed to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False
    
    async def unsubscribe(self, connection_id: str, channel: str) -> bool:
        """Unsubscribe connection from a channel."""
        try:
            if connection_id not in self._connections:
                return False
            
            connection = self._connections[connection_id]
            
            # Remove from channel subscriptions
            if channel in self._subscriptions:
                self._subscriptions[channel] = [
                    sub for sub in self._subscriptions[channel]
                    if sub.connection_id != connection_id
                ]
                
                # Remove empty channel
                if not self._subscriptions[channel]:
                    del self._subscriptions[channel]
            
            # Remove from connection subscriptions
            connection.subscriptions.discard(channel)
            
            self._stats["subscriptions_total"] = sum(
                len(subs) for subs in self._subscriptions.values()
            )
            
            # Send unsubscribe confirmation
            await self.send_message(connection_id, WebSocketMessage(
                type=MessageType.UNSUBSCRIBE,
                data={"channel": channel, "status": "unsubscribed"}
            ))
            
            logger.debug(f"Connection {connection_id} unsubscribed from {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            return False
    
    async def send_message(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send message to a specific connection."""
        try:
            if connection_id not in self._connections:
                # Store for later delivery if connection is temporarily unavailable
                if connection_id not in self._pending_messages:
                    self._pending_messages[connection_id] = []
                self._pending_messages[connection_id].append(message)
                return False
            
            connection = self._connections[connection_id]
            
            if connection.websocket.client_state != WebSocketState.CONNECTED:
                return False
            
            # Convert message to JSON
            message_data = {
                "type": message.type.value,
                "data": message.data,
                "timestamp": message.timestamp.isoformat(),
                "message_id": message.message_id
            }
            
            if message.source:
                message_data["source"] = message.source
            
            # Send message
            await connection.websocket.send_text(json.dumps(message_data))
            
            self._stats["messages_sent"] += 1
            
            logger.debug(f"Message sent to {connection_id}: {message.type.value}")
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id, "Connection lost")
            return False
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self._stats["errors_count"] += 1
            return False
    
    async def broadcast_to_channel(
        self,
        channel: str,
        message: WebSocketMessage,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Broadcast message to all subscribers of a channel."""
        try:
            if channel not in self._subscriptions:
                return 0
            
            sent_count = 0
            subscriptions = self._subscriptions[channel]
            
            for subscription in subscriptions:
                # Apply filters if specified
                if filters and not self._matches_filters(subscription, filters):
                    continue
                
                # Send message
                success = await self.send_message(subscription.connection_id, message)
                if success:
                    sent_count += 1
            
            logger.debug(f"Broadcast to {channel}: {sent_count} messages sent")
            return sent_count
            
        except Exception as e:
            logger.error(f"Broadcast failed for channel {channel}: {e}")
            return 0
    
    async def broadcast_to_user(
        self,
        user_id: str,
        message: WebSocketMessage
    ) -> int:
        """Broadcast message to all connections of a user."""
        try:
            if user_id not in self._user_connections:
                return 0
            
            sent_count = 0
            connection_ids = list(self._user_connections[user_id])
            
            for connection_id in connection_ids:
                success = await self.send_message(connection_id, message)
                if success:
                    sent_count += 1
            
            logger.debug(f"Broadcast to user {user_id}: {sent_count} messages sent")
            return sent_count
            
        except Exception as e:
            logger.error(f"User broadcast failed for {user_id}: {e}")
            return 0
    
    async def broadcast_to_all(self, message: WebSocketMessage) -> int:
        """Broadcast message to all connected clients."""
        try:
            sent_count = 0
            connection_ids = list(self._connections.keys())
            
            for connection_id in connection_ids:
                success = await self.send_message(connection_id, message)
                if success:
                    sent_count += 1
            
            logger.debug(f"Broadcast to all: {sent_count} messages sent")
            return sent_count
            
        except Exception as e:
            logger.error(f"Global broadcast failed: {e}")
            return 0
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """Handle incoming WebSocket message."""
        try:
            if connection_id not in self._connections:
                return
            
            connection = self._connections[connection_id]
            
            # Parse message
            try:
                message_data = json.loads(raw_message)
            except json.JSONDecodeError:
                await self.send_error(connection_id, "Invalid JSON message")
                return
            
            # Validate message structure
            if "type" not in message_data:
                await self.send_error(connection_id, "Missing message type")
                return
            
            message_type = MessageType(message_data["type"])
            message = WebSocketMessage(
                type=message_type,
                data=message_data.get("data", {}),
                source=connection_id
            )
            
            # Update stats
            self._stats["messages_received"] += 1
            connection.last_heartbeat = datetime.utcnow()
            
            # Handle message based on type
            if message_type in self._message_handlers:
                await self._message_handlers[message_type](connection_id, message)
            else:
                await self.send_error(connection_id, f"Unknown message type: {message_type}")
            
        except Exception as e:
            logger.error(f"Message handling failed for {connection_id}: {e}")
            await self.send_error(connection_id, "Message processing error")
    
    async def send_error(self, connection_id: str, error_message: str):
        """Send error message to connection."""
        error_msg = WebSocketMessage(
            type=MessageType.ERROR,
            data={"error": error_message}
        )
        await self.send_message(connection_id, error_msg)
    
    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[str, WebSocketMessage], None]
    ):
        """Register a message handler for a specific message type."""
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type.value}")
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information."""
        if connection_id not in self._connections:
            return None
        
        connection = self._connections[connection_id]
        return {
            "id": connection.id,
            "user_id": connection.user_id,
            "session_id": connection.session_id,
            "project_id": connection.project_id,
            "state": connection.state.value,
            "connected_at": connection.connected_at.isoformat(),
            "last_heartbeat": connection.last_heartbeat.isoformat(),
            "subscriptions": list(connection.subscriptions),
            "metadata": connection.metadata
        }
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            **self._stats,
            "channels": list(self._subscriptions.keys()),
            "channel_subscription_counts": {
                channel: len(subs) for channel, subs in self._subscriptions.items()
            },
            "authenticated_connections": len([
                conn for conn in self._connections.values()
                if conn.state == ConnectionState.AUTHENTICATED
            ]),
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
        }
    
    # Internal methods
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self._message_handlers[MessageType.AUTH] = self._handle_auth
        self._message_handlers[MessageType.SUBSCRIBE] = self._handle_subscribe
        self._message_handlers[MessageType.UNSUBSCRIBE] = self._handle_unsubscribe
        self._message_handlers[MessageType.PING] = self._handle_ping
        self._message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
    
    async def _handle_auth(self, connection_id: str, message: WebSocketMessage):
        """Handle authentication message."""
        token = message.data.get("token")
        if not token:
            await self.send_error(connection_id, "Missing authentication token")
            return
        
        success = await self.authenticate(connection_id, token)
        if not success:
            await self.send_error(connection_id, "Authentication failed")
            # Close connection after failed auth
            await self.disconnect(connection_id, "Authentication failed")
    
    async def _handle_subscribe(self, connection_id: str, message: WebSocketMessage):
        """Handle subscription message."""
        channel = message.data.get("channel")
        if not channel:
            await self.send_error(connection_id, "Missing channel name")
            return
        
        filters = message.data.get("filters")
        success = await self.subscribe(connection_id, channel, filters)
        if not success:
            await self.send_error(connection_id, f"Subscription to {channel} failed")
    
    async def _handle_unsubscribe(self, connection_id: str, message: WebSocketMessage):
        """Handle unsubscription message."""
        channel = message.data.get("channel")
        if not channel:
            await self.send_error(connection_id, "Missing channel name")
            return
        
        await self.unsubscribe(connection_id, channel)
    
    async def _handle_ping(self, connection_id: str, message: WebSocketMessage):
        """Handle ping message."""
        pong_msg = WebSocketMessage(
            type=MessageType.PONG,
            data={"timestamp": datetime.utcnow().isoformat()}
        )
        await self.send_message(connection_id, pong_msg)
    
    async def _handle_heartbeat(self, connection_id: str, message: WebSocketMessage):
        """Handle heartbeat message."""
        if connection_id in self._connections:
            self._connections[connection_id].last_heartbeat = datetime.utcnow()
    
    async def _auth_timeout_handler(self, connection_id: str):
        """Handle authentication timeout."""
        await asyncio.sleep(self.auth_timeout)
        
        if connection_id in self._connections:
            connection = self._connections[connection_id]
            if connection.state == ConnectionState.CONNECTED:
                await self.disconnect(connection_id, "Authentication timeout")
    
    async def _remove_connection_subscriptions(self, connection_id: str):
        """Remove all subscriptions for a connection."""
        for channel in list(self._subscriptions.keys()):
            self._subscriptions[channel] = [
                sub for sub in self._subscriptions[channel]
                if sub.connection_id != connection_id
            ]
            
            # Remove empty channels
            if not self._subscriptions[channel]:
                del self._subscriptions[channel]
    
    def _matches_filters(
        self,
        subscription: Subscription,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if subscription matches broadcast filters."""
        for key, value in filters.items():
            if key == "user_id" and subscription.user_id != value:
                return False
            if key in subscription.filters and subscription.filters[key] != value:
                return False
        
        return True
    
    async def _cleanup_loop(self):
        """Background loop for connection cleanup."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self._connections.items():
                    # Check for stale connections
                    if (current_time - connection.last_heartbeat).total_seconds() > self.connection_timeout:
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id, "Connection timeout")
                
                # Clear old pending messages
                for connection_id in list(self._pending_messages.keys()):
                    if connection_id not in self._connections:
                        messages = self._pending_messages[connection_id]
                        # Keep only recent messages (last 5 minutes)
                        cutoff = current_time - timedelta(minutes=5)
                        self._pending_messages[connection_id] = [
                            msg for msg in messages
                            if msg.timestamp > cutoff
                        ]
                        
                        # Remove if empty
                        if not self._pending_messages[connection_id]:
                            del self._pending_messages[connection_id]
                
                logger.debug(f"Cleanup completed: {len(stale_connections)} stale connections removed")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _heartbeat_loop(self):
        """Background loop for sending heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                heartbeat_msg = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={"timestamp": datetime.utcnow().isoformat()}
                )
                
                # Send to all authenticated connections
                for connection_id, connection in self._connections.items():
                    if connection.state == ConnectionState.AUTHENTICATED:
                        await self.send_message(connection_id, heartbeat_msg)
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
    
    async def _broadcast_loop(self):
        """Background loop for processing broadcast queue."""
        while self._running:
            try:
                # Process broadcast queue
                # This could be used for queued broadcasts from external services
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")
    
    async def _close_all_connections(self):
        """Close all active connections."""
        connection_ids = list(self._connections.keys())
        
        for connection_id in connection_ids:
            await self.disconnect(connection_id, "Server shutdown")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Convenience functions

async def start_websocket_manager():
    """Start the WebSocket manager."""
    await websocket_manager.start()


async def stop_websocket_manager():
    """Stop the WebSocket manager."""
    await websocket_manager.stop()


async def broadcast_metric_update(metric_name: str, value: float, tags: Dict[str, str] = None):
    """Broadcast metric update to subscribers."""
    message = WebSocketMessage(
        type=MessageType.DATA,
        data={
            "event": "metric_update",
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    await websocket_manager.broadcast_to_channel("metrics", message)


async def broadcast_alert(alert_data: Dict[str, Any]):
    """Broadcast alert to subscribers."""
    message = WebSocketMessage(
        type=MessageType.NOTIFICATION,
        data={
            "event": "alert",
            "alert": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    await websocket_manager.broadcast_to_channel("alerts", message)


async def broadcast_system_status(status_data: Dict[str, Any]):
    """Broadcast system status update."""
    message = WebSocketMessage(
        type=MessageType.DATA,
        data={
            "event": "system_status",
            "status": status_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    await websocket_manager.broadcast_to_channel("system", message)