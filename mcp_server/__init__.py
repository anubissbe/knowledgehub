"""
KnowledgeHub MCP Server Package.

This package provides a Model Context Protocol (MCP) server that allows
Claude Code to directly integrate with KnowledgeHub's AI-enhanced systems.

Main Components:
- server.py: Main MCP server implementation
- tools.py: Tool registry and definitions
- handlers.py: Tool implementation handlers

Usage:
    python -m mcp_server.server

Features:
- 25+ AI-enhanced tools across 5 categories
- Real-time context synchronization
- Memory operations with semantic search
- Session management and continuity
- AI intelligence features
- Analytics and metrics access
"""

from .server import KnowledgeHubMCPServer, main
from .tools import (
    KnowledgeHubTools, ToolRegistry, ToolError,
    MEMORY_TOOLS, SESSION_TOOLS, AI_TOOLS, 
    ANALYTICS_TOOLS, UTILITY_TOOLS,
    get_total_tool_count, get_tool_by_name, list_all_tool_names
)
from .handlers import (
    MemoryHandler, SessionHandler, AIHandler, 
    AnalyticsHandler, ContextSynchronizer, ResponseFormatter
)

__version__ = "1.0.0"
__author__ = "KnowledgeHub AI Team"

# Export main components
__all__ = [
    # Server
    "KnowledgeHubMCPServer",
    "main",
    
    # Tools
    "KnowledgeHubTools",
    "ToolRegistry", 
    "ToolError",
    "MEMORY_TOOLS",
    "SESSION_TOOLS", 
    "AI_TOOLS",
    "ANALYTICS_TOOLS",
    "UTILITY_TOOLS",
    "get_total_tool_count",
    "get_tool_by_name",
    "list_all_tool_names",
    
    # Handlers
    "MemoryHandler",
    "SessionHandler",
    "AIHandler",
    "AnalyticsHandler", 
    "ContextSynchronizer",
    "ResponseFormatter"
]

# Server configuration
SERVER_INFO = {
    "name": "knowledgehub",
    "version": __version__,
    "description": "AI-enhanced MCP server for KnowledgeHub integration",
    "capabilities": [
        "memory_operations",
        "session_management", 
        "ai_intelligence",
        "real_time_analytics",
        "context_synchronization"
    ],
    "tool_count": None  # Will be set dynamically
}

def get_server_info():
    """Get server information including dynamic tool count."""
    info = SERVER_INFO.copy()
    info["tool_count"] = get_total_tool_count()
    return info