"""
Model Context Protocol (MCP) Server for KnowledgeHub.

This server provides Claude Code with direct access to KnowledgeHub's
AI-enhanced memory and intelligence systems through the MCP protocol.

Features:
- Memory operations (create, search, retrieve, update)
- Session management and continuity
- AI intelligence features
- Real-time analytics access
- Context synchronization
- Learning and adaptation
"""

import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime
import traceback

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    LoggingLevel, CallToolResult, ListResourcesResult, ListToolsResult,
    ReadResourceResult
)
import mcp.types as types

from .tools import (
    KnowledgeHubTools, ToolRegistry, ToolError,
    # Tool categories
    MEMORY_TOOLS, SESSION_TOOLS, AI_TOOLS, ANALYTICS_TOOLS, UTILITY_TOOLS
)
from .handlers import (
    MemoryHandler, SessionHandler, AIHandler, AnalyticsHandler,
    ContextSynchronizer, ResponseFormatter
)
from .performance_monitor import performance_monitor, monitor_tool_execution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeHubMCPServer:
    """
    MCP Server for KnowledgeHub integration with Claude Code.
    
    This server exposes KnowledgeHub's AI capabilities through the Model Context Protocol,
    allowing Claude Code to seamlessly access memory systems, AI features, and analytics.
    """
    
    def __init__(self):
        self.server = Server("knowledgehub")
        self.tools = KnowledgeHubTools()
        self.tool_registry = ToolRegistry()
        
        # Initialize handlers
        self.memory_handler = MemoryHandler()
        self.session_handler = SessionHandler()
        self.ai_handler = AIHandler()
        self.analytics_handler = AnalyticsHandler()
        self.context_sync = ContextSynchronizer()
        self.response_formatter = ResponseFormatter()
        
        # Server state
        self.initialized = False
        self.client_info = {}
        self.active_sessions = {}
        
        # Setup server handlers
        self._setup_server_handlers()
        
        logger.info("KnowledgeHub MCP Server initialized")
    
    def _setup_server_handlers(self):
        """Setup MCP server event handlers."""
        
        @self.server.list_resources()
        async def handle_list_resources() -> ListResourcesResult:
            """List available resources."""
            try:
                resources = []
                
                # Memory resources
                resources.extend([
                    Resource(
                        uri="knowledgehub://memories",
                        name="Memory System",
                        description="Access to AI-enhanced memory storage and retrieval",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="knowledgehub://sessions",
                        name="Session Management",
                        description="Session continuity and context management",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="knowledgehub://ai-features",
                        name="AI Intelligence",
                        description="Advanced AI features and learning systems",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="knowledgehub://analytics",
                        name="Real-time Analytics",
                        description="Performance metrics and system analytics",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="knowledgehub://context",
                        name="Context Synchronization",
                        description="Real-time context and state synchronization",
                        mimeType="application/json"
                    )
                ])
                
                return ListResourcesResult(resources=resources)
                
            except Exception as e:
                logger.error(f"Error listing resources: {e}")
                return ListResourcesResult(resources=[])
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> ReadResourceResult:
            """Read a specific resource."""
            try:
                if uri == "knowledgehub://memories":
                    content = await self.memory_handler.get_resource_info()
                elif uri == "knowledgehub://sessions":
                    content = await self.session_handler.get_resource_info()
                elif uri == "knowledgehub://ai-features":
                    content = await self.ai_handler.get_resource_info()
                elif uri == "knowledgehub://analytics":
                    content = await self.analytics_handler.get_resource_info()
                elif uri == "knowledgehub://context":
                    content = await self.context_sync.get_resource_info()
                else:
                    content = {"error": f"Unknown resource: {uri}"}
                
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps(content, indent=2)
                        )
                    ]
                )
                
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps({"error": str(e)}, indent=2)
                        )
                    ]
                )
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools."""
            try:
                tools = []
                
                # Register all tool categories
                await self._register_all_tools()
                
                # Get tools from registry
                for tool_id, tool_info in self.tool_registry.get_all_tools().items():
                    tools.append(Tool(
                        name=tool_id,
                        description=tool_info["description"],
                        inputSchema=tool_info["input_schema"]
                    ))
                
                logger.info(f"Listing {len(tools)} available tools")
                return ListToolsResult(tools=tools)
                
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                return ListToolsResult(tools=[])
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool execution with performance monitoring."""
            start_time = time.time()
            execution_success = False
            error_message = None
            
            try:
                logger.info(f"Executing tool: {name} with args: {arguments}")
                
                # Get tool info
                tool_info = self.tool_registry.get_tool(name)
                if not tool_info:
                    raise ToolError(f"Unknown tool: {name}")
                
                # Execute tool
                result = await self._execute_tool(name, arguments, tool_info)
                execution_success = result.get("success", False) if isinstance(result, dict) else True
                
                # Format response
                execution_time = time.time() - start_time
                formatted_result = await self.response_formatter.format_tool_result(
                    tool_name=name,
                    result=result,
                    execution_time=execution_time
                )
                
                # Record performance metrics
                performance_monitor.record_tool_execution(
                    tool_name=name,
                    execution_time=execution_time * 1000,  # Convert to milliseconds
                    success=execution_success,
                    user_context=f"args: {len(arguments)} params"
                )
                
                # Synchronize context if needed
                if tool_info.get("sync_context", False):
                    await self.context_sync.sync_after_tool_execution(
                        tool_name=name,
                        arguments=arguments,
                        result=result
                    )
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(formatted_result, indent=2)
                        )
                    ]
                )
                
            except ToolError as e:
                error_message = str(e)
                logger.error(f"Tool error in {name}: {e}")
                
                # Record failed execution
                execution_time = time.time() - start_time
                performance_monitor.record_tool_execution(
                    tool_name=name,
                    execution_time=execution_time * 1000,
                    success=False,
                    error_message=error_message,
                    user_context=f"args: {len(arguments)} params"
                )
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "error": str(e),
                                "tool": name,
                                "type": "tool_error"
                            }, indent=2)
                        )
                    ]
                )
            except Exception as e:
                error_message = f"Internal server error: {str(e)}"
                logger.error(f"Unexpected error in {name}: {e}")
                logger.error(traceback.format_exc())
                
                # Record failed execution
                execution_time = time.time() - start_time
                performance_monitor.record_tool_execution(
                    tool_name=name,
                    execution_time=execution_time * 1000,
                    success=False,
                    error_message=error_message,
                    user_context=f"args: {len(arguments)} params"
                )
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "error": error_message,
                                "tool": name,
                                "type": "server_error"
                            }, indent=2)
                        )
                    ]
                )
        
        @self.server.notification(types.NotificationType.Initialized)
        async def handle_initialized(params):
            """Handle client initialization."""
            logger.info("Client initialized")
            self.initialized = True
            
            # Store client info
            if hasattr(params, 'clientInfo'):
                self.client_info = params.clientInfo
                logger.info(f"Client: {self.client_info.get('name', 'Unknown')}")
    
    async def _register_all_tools(self):
        """Register all available tools."""
        try:
            # Register memory tools
            for tool_id, tool_config in MEMORY_TOOLS.items():
                self.tool_registry.register_tool(
                    tool_id=tool_id,
                    handler=self.memory_handler,
                    **tool_config
                )
            
            # Register session tools
            for tool_id, tool_config in SESSION_TOOLS.items():
                self.tool_registry.register_tool(
                    tool_id=tool_id,
                    handler=self.session_handler,
                    **tool_config
                )
            
            # Register AI tools
            for tool_id, tool_config in AI_TOOLS.items():
                self.tool_registry.register_tool(
                    tool_id=tool_id,
                    handler=self.ai_handler,
                    **tool_config
                )
            
            # Register analytics tools
            for tool_id, tool_config in ANALYTICS_TOOLS.items():
                self.tool_registry.register_tool(
                    tool_id=tool_id,
                    handler=self.analytics_handler,
                    **tool_config
                )
            
            # Register utility tools
            for tool_id, tool_config in UTILITY_TOOLS.items():
                self.tool_registry.register_tool(
                    tool_id=tool_id,
                    handler=self.context_sync,
                    **tool_config
                )
            
            logger.info(f"Registered {len(self.tool_registry.get_all_tools())} tools")
            
        except Exception as e:
            logger.error(f"Error registering tools: {e}")
    
    async def _execute_tool(self, name: str, arguments: Dict[str, Any], tool_info: Dict[str, Any]) -> Any:
        """Execute a tool with the appropriate handler."""
        try:
            handler = tool_info["handler"]
            method_name = tool_info["method"]
            
            # Get the method from the handler
            if hasattr(handler, method_name):
                method = getattr(handler, method_name)
                
                # Execute the method
                if asyncio.iscoroutinefunction(method):
                    result = await method(**arguments)
                else:
                    result = method(**arguments)
                
                return result
            else:
                raise ToolError(f"Method {method_name} not found in handler")
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            raise ToolError(f"Tool execution failed: {str(e)}")
    
    async def run(self, transport_type: str = "stdio"):
        """Run the MCP server."""
        try:
            logger.info(f"Starting KnowledgeHub MCP Server with {transport_type} transport")
            
            # Start performance monitoring
            performance_monitor.start_monitoring()
            
            if transport_type == "stdio":
                # Run with stdio transport (standard for MCP)
                from mcp.server.stdio import stdio_server
                
                async with stdio_server() as (read_stream, write_stream):
                    await self.server.run(
                        read_stream,
                        write_stream,
                        InitializationOptions(
                            server_name="knowledgehub",
                            server_version="1.0.0",
                            capabilities=self.server.get_capabilities(
                                notification_options=NotificationOptions(),
                                experimental_capabilities={}
                            )
                        )
                    )
            else:
                raise ValueError(f"Unsupported transport type: {transport_type}")
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Stop performance monitoring
            await performance_monitor.stop_monitoring()
    
    async def shutdown(self):
        """Gracefully shutdown the server."""
        logger.info("Shutting down KnowledgeHub MCP Server")
        
        # Close active sessions
        for session_id in list(self.active_sessions.keys()):
            try:
                await self.session_handler.close_session(session_id)
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
        
        # Cleanup handlers
        await self.memory_handler.cleanup()
        await self.session_handler.cleanup()
        await self.ai_handler.cleanup()
        await self.analytics_handler.cleanup()
        await self.context_sync.cleanup()
        
        logger.info("Server shutdown complete")


# CLI interface
async def main():
    """Main entry point for the MCP server."""
    server = KnowledgeHubMCPServer()
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        await server.shutdown()


if __name__ == "__main__":
    # Run the server
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)