#!/usr/bin/env python3
"""
Test script for Code Intelligence MCP Server
"""

import asyncio
import json
from pathlib import Path

# Add the API path to import the MCP server
import sys
sys.path.append('/opt/projects/knowledgehub/api/services')

from code_intelligence_mcp import app

async def test_mcp_server():
    """Test the MCP server tools"""
    print("Testing Code Intelligence MCP Server...")
    
    # Test tool listing
    tools = await app.list_tools()
    print(f"\nAvailable tools: {len(tools())}")
    for tool in tools():
        print(f"  - {tool.name}: {tool.description}")
    
    # Test project activation
    print("\n=== Testing Project Activation ===")
    result = await app.call_tool("activate_project", {
        "project_path": "/opt/projects/knowledgehub"
    })
    for item in result:
        print(item.text)
    
    # Test symbols overview
    print("\n=== Testing Symbols Overview ===")
    result = await app.call_tool("get_symbols_overview", {
        "relative_path": "api/main.py"
    })
    for item in result:
        print(item.text[:500] + "..." if len(item.text) > 500 else item.text)
    
    # Test pattern search
    print("\n=== Testing Pattern Search ===")
    result = await app.call_tool("search_pattern", {
        "pattern": "def.*health",
        "file_pattern": "*.py"
    })
    for item in result:
        print(item.text[:800] + "..." if len(item.text) > 800 else item.text)
    
    print("\n=== MCP Server Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())