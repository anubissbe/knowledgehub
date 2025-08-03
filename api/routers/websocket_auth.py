"""WebSocket authentication handler"""

from fastapi import WebSocket, WebSocketDisconnect, Query, HTTPException
from typing import Optional
import logging

logger = logging.getLogger(__name__)


async def get_websocket_user(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
) -> Optional[dict]:
    """
    Get user from WebSocket connection
    
    WebSocket connections can't use regular HTTP headers for auth,
    so we accept token as query parameter
    """
    
    # For now, allow all connections (public WebSocket)
    # In production, you would validate the token here
    
    if token:
        # TODO: Validate token and return user info
        logger.info(f"WebSocket connection with token: {token[:10]}...")
        return {"id": "authenticated_user", "token": token}
    
    # Allow anonymous connections
    logger.info("Anonymous WebSocket connection")
    return {"id": "anonymous", "token": None}


async def validate_websocket_origin(websocket: WebSocket) -> bool:
    """
    Validate WebSocket connection origin
    
    Prevents CSRF attacks on WebSocket connections
    """
    headers = dict(websocket.headers)
    origin = headers.get("origin", "")
    
    # Allow connections without origin header (e.g., from non-browser clients)
    if not origin:
        logger.info("WebSocket connection without origin header allowed (non-browser client)")
        return True
    
    # In development, allow localhost origins
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:3100",
        "http://192.168.1.25:3100",
        "http://192.168.1.24:3000"
    ]
    
    if origin in allowed_origins:
        return True
    
    # Allow same-origin connections
    host = headers.get("host", "")
    if origin == f"http://{host}" or origin == f"https://{host}":
        return True
    
    logger.warning(f"WebSocket connection from unauthorized origin: {origin}")
    return False