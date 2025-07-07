#!/usr/bin/env python3
"""Start MCP server with health file"""
import os
import asyncio
from .server import KnowledgeHubMCPServer

async def main():
    # Create health file immediately
    try:
        with open('/tmp/mcp_healthy', 'w') as f:
            f.write('healthy')
        print("Health file created successfully")
    except Exception as e:
        print(f"Failed to create health file: {e}")
    
    # Start server
    server = KnowledgeHubMCPServer()
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())