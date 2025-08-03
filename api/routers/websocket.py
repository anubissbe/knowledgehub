"""Real WebSocket router using comprehensive WebSocket manager"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
import json
import logging
from typing import Dict, Set, Optional
import asyncio

from ..websocket.manager import websocket_manager, WebSocketMessage, MessageType
from ..services.real_websocket_events import real_websocket_events
from ..services.real_embeddings_service import real_embeddings_service
from .websocket_auth import get_websocket_user, validate_websocket_origin

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/notifications")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """Real-time WebSocket endpoint using comprehensive manager"""
    connection_id = None
    user = None
    
    try:
        # Validate origin first
        if not await validate_websocket_origin(websocket):
            await websocket.close(code=1008, reason="Origin not allowed")
            return
        
        # Connect using WebSocket manager
        connection_id = await websocket_manager.connect(websocket)
        
        # Get user from token (if provided)
        user = await get_websocket_user(websocket, token)
        
        # Authenticate if token provided
        if token and user and user.get("id") != "anonymous":
            auth_success = await websocket_manager.authenticate(connection_id, token)
            if not auth_success:
                logger.warning(f"Authentication failed for connection {connection_id}")
                await websocket_manager.disconnect(connection_id, "Authentication failed")
                return
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user.get('id') if user else 'anonymous'})")
        
        # Auto-subscribe to relevant channels
        if user and user.get("id") != "anonymous":
            user_id = user["id"]
            await websocket_manager.subscribe(connection_id, f"user.{user_id}")
            await websocket_manager.subscribe(connection_id, "system")
            
            # Subscribe to session channel if session_id available
            session_id = user.get("session_id")
            if session_id:
                await websocket_manager.subscribe(connection_id, f"session.{session_id}")
            
            # Subscribe to project channel if project_id available  
            project_id = user.get("project_id")
            if project_id:
                await websocket_manager.subscribe(connection_id, f"project.{project_id}")
        else:
            # Anonymous users get limited channels
            await websocket_manager.subscribe(connection_id, "system")
        
        # Message handling loop
        while True:
            try:
                # Receive message from client
                raw_message = await websocket.receive_text()
                
                # Let WebSocket manager handle the message
                await websocket_manager.handle_message(connection_id, raw_message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Message handling error for {connection_id}: {e}")
                await websocket_manager.send_error(connection_id, "Message processing error")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup via manager
        if connection_id:
            await websocket_manager.disconnect(connection_id, "Client disconnect")


@router.websocket("/realtime")
async def realtime_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    channels: Optional[str] = Query(None)
):
    """Enhanced real-time endpoint with channel subscription"""
    connection_id = None
    
    try:
        # Connect using WebSocket manager
        connection_id = await websocket_manager.connect(websocket)
        
        # Authenticate if token provided
        if token:
            auth_success = await websocket_manager.authenticate(connection_id, token)
            if not auth_success:
                await websocket_manager.disconnect(connection_id, "Authentication required")
                return
        
        # Subscribe to requested channels
        if channels:
            channel_list = [ch.strip() for ch in channels.split(",")]
            for channel in channel_list:
                await websocket_manager.subscribe(connection_id, channel)
        
        # Keep connection alive
        while True:
            try:
                raw_message = await websocket.receive_text()
                await websocket_manager.handle_message(connection_id, raw_message)
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"Realtime endpoint error: {e}")
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id, "Client disconnect")


async def broadcast_job_update(job_id: str, update: Dict):
    """Broadcast job update using WebSocket manager"""
    message = WebSocketMessage(
        type=MessageType.DATA,
        data={
            "event": "job_update",
            "job_id": job_id,
            "update": update
        }
    )
    
    # Broadcast to jobs channel
    await websocket_manager.broadcast_to_channel(f"jobs.{job_id}", message)


async def broadcast_to_all(message: Dict):
    """Broadcast message to all connected clients using WebSocket manager"""
    ws_message = WebSocketMessage(
        type=MessageType.DATA,
        data=message
    )
    
    await websocket_manager.broadcast_to_all(ws_message)


async def broadcast_memory_update(memory_id: str, user_id: str, update_type: str, data: Dict):
    """Broadcast memory update event"""
    await real_websocket_events.publish_memory_updated(
        memory_id=memory_id,
        user_id=user_id,
        update_type=update_type,
        update_data=data
    )


async def broadcast_search_results(user_id: str, search_query: str, results: list[dict]):
    """Broadcast search results event"""
    await real_websocket_events.publish_search_executed(
        search_query=search_query,
        search_type="semantic",
        results_count=len(results),
        user_id=user_id,
        processing_time=0.0,  # Add actual timing
        results_preview=results[:3]  # Send preview of results
    )


async def broadcast_ai_insight(insight_type: str, insight: str, confidence: float, user_id: Optional[str] = None):
    """Broadcast AI insight event"""
    await real_websocket_events.publish_prediction_made(
        prediction_type=insight_type,
        prediction=insight,
        confidence=confidence,
        context={},
        user_id=user_id
    )


# WebSocket manager status endpoints

@router.get("/websocket/status")
async def get_websocket_status():
    """Get WebSocket manager status"""
    return {
        "manager_stats": websocket_manager.get_manager_stats(),
        "events_stats": real_websocket_events.get_event_stats(),
        "embeddings_stats": real_embeddings_service.get_embedding_stats()
    }


@router.get("/websocket/connections")
async def get_websocket_connections():
    """Get active WebSocket connections (admin only)"""
    stats = websocket_manager.get_manager_stats()
    return {
        "active_connections": stats["active_connections"],
        "authenticated_connections": stats["authenticated_connections"],
        "channels": stats["channels"],
        "channel_subscriptions": stats["channel_subscription_counts"]
    }