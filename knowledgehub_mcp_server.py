#!/usr/bin/env python3
"""
KnowledgeHub MCP Server for Claude Code
Provides code intelligence capabilities via KnowledgeHub API
"""

import asyncio
import json
import sys
import httpx
import os
from typing import Any, Dict, List, Optional
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    LoggingLevel
)
import mcp.types as types
from pydantic import AnyUrl
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledgehub-mcp")

# Configuration
KNOWLEDGEHUB_API_URL = os.getenv("KNOWLEDGEHUB_API_URL", "http://192.168.1.25:3000")
KNOWLEDGEHUB_PROJECT_PATH = os.getenv("KNOWLEDGEHUB_PROJECT_PATH", "/opt/projects/knowledgehub")

# Create the server instance
server = Server("knowledgehub")

# HTTP client for KnowledgeHub API
http_client = httpx.AsyncClient(timeout=30.0)


@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available KnowledgeHub resources"""
    return [
        Resource(
            uri=AnyUrl("knowledgehub://project/current"),
            name="Current Project Context",
            description="Active project information and context",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("knowledgehub://symbols/overview"),
            name="Project Symbols Overview", 
            description="Overview of all symbols in the current project",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("knowledgehub://memories/list"),
            name="Project Memories",
            description="List of saved project memories",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read KnowledgeHub resources"""
    
    if uri.scheme != "knowledgehub":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")
    
    try:
        if str(uri) == "knowledgehub://project/current":
            # Get current project context
            response = await http_client.post(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/activate-project",
                json={"project_path": KNOWLEDGEHUB_PROJECT_PATH}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
            
        elif str(uri) == "knowledgehub://symbols/overview":
            # Get symbols overview
            response = await http_client.get(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/symbols/overview",
                params={"project_path": KNOWLEDGEHUB_PROJECT_PATH}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
            
        elif str(uri) == "knowledgehub://memories/list":
            # Get project memories
            response = await http_client.get(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/memory/list",
                params={"project_path": KNOWLEDGEHUB_PROJECT_PATH}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
            
        else:
            raise ValueError(f"Unknown resource: {uri}")
            
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        return json.dumps({"error": str(e)})


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available KnowledgeHub tools"""
    return [
        Tool(
            name="activate_project",
            description="Activate a project for code intelligence analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project to activate"
                    }
                },
                "required": ["project_path"]
            },
        ),
        Tool(
            name="get_symbols_overview",
            description="Get overview of symbols in a project or specific file",
            inputSchema={
                "type": "object", 
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional specific file to analyze"
                    }
                },
                "required": ["project_path"]
            },
        ),
        Tool(
            name="find_symbol",
            description="Find a specific symbol by name or path",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string", 
                        "description": "Path to the project"
                    },
                    "name_path": {
                        "type": "string",
                        "description": "Symbol name or path to find"
                    },
                    "include_body": {
                        "type": "boolean",
                        "description": "Include symbol source code",
                        "default": False
                    },
                    "include_references": {
                        "type": "boolean", 
                        "description": "Include symbol references",
                        "default": False
                    }
                },
                "required": ["project_path", "name_path"]
            },
        ),
        Tool(
            name="search_pattern",
            description="Search for a pattern in project files",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "File pattern to limit search (e.g., '*.py')"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines around matches",
                        "default": 2
                    }
                },
                "required": ["project_path", "pattern"]
            },
        ),
        Tool(
            name="replace_symbol",
            description="Replace a symbol's body with new code",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project"
                    },
                    "symbol_path": {
                        "type": "string", 
                        "description": "Path to the symbol to replace"
                    },
                    "new_body": {
                        "type": "string",
                        "description": "New code body for the symbol"
                    }
                },
                "required": ["project_path", "symbol_path", "new_body"]
            },
        ),
        Tool(
            name="save_memory",
            description="Save a project-specific memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project"
                    },
                    "name": {
                        "type": "string",
                        "description": "Memory name"
                    },
                    "content": {
                        "type": "string", 
                        "description": "Memory content"
                    }
                },
                "required": ["project_path", "name", "content"]
            },
        ),
        Tool(
            name="load_memory",
            description="Load a project-specific memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project"
                    },
                    "name": {
                        "type": "string",
                        "description": "Memory name to load"
                    }
                },
                "required": ["project_path", "name"]
            },
        ),
        Tool(
            name="list_memories",
            description="List all memories for a project",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project"
                    }
                },
                "required": ["project_path"]
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls to KnowledgeHub API"""
    
    try:
        if name == "activate_project":
            response = await http_client.post(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/activate-project",
                json=arguments
            )
            response.raise_for_status()
            result = response.json()
            
        elif name == "get_symbols_overview":
            params = {"project_path": arguments["project_path"]}
            if "file_path" in arguments:
                params["file_path"] = arguments["file_path"]
            
            response = await http_client.get(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/symbols/overview",
                params=params
            )
            response.raise_for_status()
            result = response.json()
            
        elif name == "find_symbol":
            response = await http_client.post(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/symbols/find",
                json=arguments
            )
            response.raise_for_status()
            result = response.json()
            
        elif name == "search_pattern":
            response = await http_client.post(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/search/pattern",
                json=arguments
            )
            response.raise_for_status()
            result = response.json()
            
        elif name == "replace_symbol":
            response = await http_client.post(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/symbols/replace",
                json=arguments
            )
            response.raise_for_status()
            result = response.json()
            
        elif name == "save_memory":
            response = await http_client.post(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/memory/save",
                json=arguments
            )
            response.raise_for_status()
            result = response.json()
            
        elif name == "load_memory":
            response = await http_client.get(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/memory/load",
                params=arguments
            )
            response.raise_for_status()
            result = response.json()
            
        elif name == "list_memories":
            response = await http_client.get(
                f"{KNOWLEDGEHUB_API_URL}/api/code-intelligence/memory/list",
                params=arguments
            )
            response.raise_for_status()
            result = response.json()
            
        else:
            raise ValueError(f"Unknown tool: {name}")
            
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        error_result = {"error": str(e), "tool": name, "arguments": arguments}
        return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def main():
    """Main entry point for the MCP server"""
    
    # Arguments
    parser = argparse.ArgumentParser(description="KnowledgeHub MCP Server")
    parser.add_argument("--api-url", default=KNOWLEDGEHUB_API_URL, 
                       help="KnowledgeHub API URL")
    parser.add_argument("--project-path", default=KNOWLEDGEHUB_PROJECT_PATH,
                       help="Default project path")
    args = parser.parse_args()
    
    # Update global configuration
    global KNOWLEDGEHUB_API_URL, KNOWLEDGEHUB_PROJECT_PATH
    KNOWLEDGEHUB_API_URL = args.api_url
    KNOWLEDGEHUB_PROJECT_PATH = args.project_path
    
    logger.info(f"Starting KnowledgeHub MCP Server")
    logger.info(f"API URL: {KNOWLEDGEHUB_API_URL}")
    logger.info(f"Project Path: {KNOWLEDGEHUB_PROJECT_PATH}")
    
    # Run the server using stdin/stdout streams
    async with server.run_stdio() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="knowledgehub",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())