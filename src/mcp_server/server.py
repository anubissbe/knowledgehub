"""Main MCP Server implementation"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

import websockets
from websockets.server import WebSocketServerProtocol

from .protocol import MCPProtocol
from .tools import KnowledgeTools
from .resources import KnowledgeResources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeHubMCPServer:
    """MCP Server for AI Knowledge Hub"""
    
    def __init__(self, knowledge_api_url: str = None, port: int = None):
        self.api_url = knowledge_api_url or os.getenv("API_URL", "http://localhost:3000")
        self.port = port or int(os.getenv("MCP_SERVER_PORT", "3002"))
        self.protocol = MCPProtocol()
        self.tools = KnowledgeTools(self.api_url)
        self.resources = KnowledgeResources(self.api_url)
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.setup_handlers()
    
    def setup_handlers(self):
        """Register all MCP method handlers"""
        
        # Initialize handler
        @self.protocol.method("initialize")
        async def handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
            """Handle initialization request"""
            return {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": "AI Knowledge Hub MCP Server",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "tools": True,
                    "resources": True,
                    "prompts": False,
                    "logging": True
                }
            }
        
        # List tools handler
        @self.protocol.method("tools/list")
        async def handle_list_tools(params: Dict[str, Any]) -> Dict[str, Any]:
            """List available tools"""
            return {
                "tools": [
                    {
                        "name": "search_knowledge",
                        "description": "Search the knowledge base using hybrid search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language search query"
                                },
                                "source_filter": {
                                    "type": "string",
                                    "description": "Optional source name to filter results"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results",
                                    "default": 10
                                },
                                "include_metadata": {
                                    "type": "boolean",
                                    "description": "Include document metadata",
                                    "default": True
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "store_memory",
                        "description": "Store information in the knowledge base",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Text content to store"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Tags for categorization"
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional metadata"
                                }
                            },
                            "required": ["content"]
                        }
                    },
                    {
                        "name": "get_relevant_context",
                        "description": "Retrieve relevant context for a query",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Context query"
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "description": "Maximum tokens to return",
                                    "default": 4000
                                },
                                "recency_weight": {
                                    "type": "number",
                                    "description": "Weight for recent documents (0-1)",
                                    "default": 0.1
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "list_knowledge_sources",
                        "description": "List all available knowledge sources",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                ]
            }
        
        # Call tool handler
        @self.protocol.method("tools/call")
        async def handle_call_tool(params: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a tool call"""
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            logger.info(f"Calling tool: {tool_name} with args: {arguments}")
            
            try:
                if tool_name == "search_knowledge":
                    result = await self.tools.search(
                        query=arguments["query"],
                        source_filter=arguments.get("source_filter"),
                        limit=arguments.get("limit", 10),
                        include_metadata=arguments.get("include_metadata", True)
                    )
                    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
                
                elif tool_name == "store_memory":
                    memory_id = await self.tools.store_memory(
                        content=arguments["content"],
                        tags=arguments.get("tags", []),
                        metadata=arguments.get("metadata", {})
                    )
                    return {"content": [{"type": "text", "text": f"Memory stored with ID: {memory_id}"}]}
                
                elif tool_name == "get_relevant_context":
                    context = await self.tools.get_context(
                        query=arguments["query"],
                        max_tokens=arguments.get("max_tokens", 4000),
                        recency_weight=arguments.get("recency_weight", 0.1)
                    )
                    return {"content": [{"type": "text", "text": context}]}
                
                elif tool_name == "list_knowledge_sources":
                    sources = await self.tools.list_sources()
                    return {"content": [{"type": "text", "text": json.dumps(sources, indent=2)}]}
                
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
                    
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return {"error": {"code": -32603, "message": str(e)}}
        
        # List resources handler
        @self.protocol.method("resources/list")
        async def handle_list_resources(params: Dict[str, Any]) -> Dict[str, Any]:
            """List available resources"""
            return {
                "resources": [
                    {
                        "uri": "knowledge://sources",
                        "name": "Knowledge Sources",
                        "description": "Detailed information about all knowledge sources",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "knowledge://stats",
                        "name": "System Statistics",
                        "description": "System statistics and metrics",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "knowledge://health",
                        "name": "System Health",
                        "description": "System health status",
                        "mimeType": "application/json"
                    }
                ]
            }
        
        # Read resource handler
        @self.protocol.method("resources/read")
        async def handle_read_resource(params: Dict[str, Any]) -> Dict[str, Any]:
            """Read a specific resource"""
            uri = params.get("uri")
            
            try:
                if uri == "knowledge://sources":
                    content = await self.resources.get_sources_details()
                elif uri == "knowledge://stats":
                    content = await self.resources.get_system_stats()
                elif uri == "knowledge://health":
                    content = await self.resources.get_health_status()
                else:
                    raise ValueError(f"Unknown resource: {uri}")
                
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": content
                    }]
                }
                
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                return {"error": {"code": -32603, "message": str(e)}}
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        self.clients[client_id] = websocket
        
        try:
            async for message in websocket:
                try:
                    # Parse JSON-RPC request
                    request = json.loads(message)
                    logger.debug(f"Received request: {request}")
                    
                    # Process request
                    response = await self.protocol.handle_request(request)
                    
                    # Send response
                    if response:
                        await websocket.send(json.dumps(response))
                        logger.debug(f"Sent response: {response}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {client_id}: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        },
                        "id": None
                    }
                    await websocket.send(json.dumps(error_response))
                    
                except Exception as e:
                    logger.error(f"Error handling request from {client_id}: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": "Internal error",
                            "data": str(e)
                        },
                        "id": request.get("id") if isinstance(request, dict) else None
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error in client handler for {client_id}: {e}")
        finally:
            del self.clients[client_id]
    
    async def start(self):
        """Start the MCP server"""
        logger.info(f"Starting MCP server on port {self.port}")
        
        # Start WebSocket server
        async with websockets.serve(self.handle_client, "0.0.0.0", self.port):
            logger.info(f"MCP server listening on ws://0.0.0.0:{self.port}")
            
            # Keep server running
            await asyncio.Future()  # Run forever


async def main():
    """Main entry point"""
    server = KnowledgeHubMCPServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())