"""MCP Protocol implementation"""

import json
from typing import Dict, Any, Callable, Optional
from dataclasses import field
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 Request"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = field(default_factory=dict)
    id: Optional[Any] = None

@dataclass 
class JSONRPCResponse:
    """JSON-RPC 2.0 Response"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None


class MCPProtocol:
    """Handles MCP JSON-RPC 2.0 protocol"""
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.notifications: Dict[str, Callable] = {}
    
    def method(self, name: str):
        """Decorator to register a method handler"""
        def decorator(func: Callable):
            self.handlers[name] = func
            return func
        return decorator
    
    def notification(self, name: str):
        """Decorator to register a notification handler"""
        def decorator(func: Callable):
            self.notifications[name] = func
            return func
        return decorator
    
    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle a JSON-RPC 2.0 request"""
        # Validate JSON-RPC format
        if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
            return self._error_response(
                -32600, "Invalid Request", request.get("id")
            )
        
        method = request.get("method")
        if not method:
            return self._error_response(
                -32600, "Invalid Request", request.get("id")
            )
        
        params = request.get("params", {})
        request_id = request.get("id")
        
        # Check if it's a notification (no id)
        if request_id is None:
            handler = self.notifications.get(method)
            if handler:
                try:
                    await handler(params)
                except Exception as e:
                    logger.error(f"Error handling notification {method}: {e}")
            return None  # No response for notifications
        
        # Handle method call
        handler = self.handlers.get(method)
        if not handler:
            return self._error_response(
                -32601, f"Method not found: {method}", request_id
            )
        
        try:
            result = await handler(params)
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        except Exception as e:
            logger.error(f"Error handling method {method}: {e}")
            return self._error_response(
                -32603, f"Internal error: {str(e)}", request_id
            )
    
    def _error_response(self, code: int, message: str, request_id: Any) -> Dict[str, Any]:
        """Create an error response"""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "id": request_id
        }