"""WebSocket router for real-time updates"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from typing import Dict, Set
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()

# Connected clients
clients: Dict[str, WebSocket] = {}
# Client subscriptions
subscriptions: Dict[str, Set[str]] = {}


@router.websocket("/notifications")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time notifications"""
    client_id = None
    try:
        await websocket.accept()
        # Create unique client ID
        import uuid
        client_id = str(uuid.uuid4())
        clients[client_id] = websocket
        subscriptions[client_id] = set()
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "message": "WebSocket connection established"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle subscription requests
            if message.get("type") == "subscribe":
                job_id = message.get("job_id")
                if job_id:
                    subscriptions[client_id].add(job_id)
                    await websocket.send_json({
                        "type": "subscribed",
                        "job_id": job_id
                    })
            
            elif message.get("type") == "unsubscribe":
                job_id = message.get("job_id")
                if job_id and job_id in subscriptions[client_id]:
                    subscriptions[client_id].remove(job_id)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "job_id": job_id
                    })
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        # Cleanup
        if client_id:
            if client_id in clients:
                del clients[client_id]
            if client_id in subscriptions:
                del subscriptions[client_id]
            logger.info(f"WebSocket client cleaned up: {client_id}")


async def broadcast_job_update(job_id: str, update: Dict):
    """Broadcast job update to subscribed clients"""
    # Find clients subscribed to this job
    for client_id, client_subs in subscriptions.items():
        if job_id in client_subs:
            websocket = clients.get(client_id)
            if websocket:
                try:
                    await websocket.send_json({
                        "type": "job_update",
                        "job_id": job_id,
                        "update": update
                    })
                except Exception as e:
                    logger.error(f"Failed to send update to {client_id}: {e}")


async def broadcast_to_all(message: Dict):
    """Broadcast message to all connected clients"""
    disconnected_clients = []
    for client_id, websocket in clients.items():
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to {client_id}: {e}")
            disconnected_clients.append(client_id)
    
    # Clean up disconnected clients
    for client_id in disconnected_clients:
        if client_id in clients:
            del clients[client_id]
        if client_id in subscriptions:
            del subscriptions[client_id]