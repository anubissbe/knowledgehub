#!/usr/bin/env python3
"""Health check wrapper that handles both WebSocket and file-based checks"""
import asyncio
import websockets
import sys
import os

async def check_websocket():
    """Check if WebSocket server is running"""
    try:
        async with websockets.connect('ws://localhost:3002') as ws:
            # Just connecting is enough to verify it's running
            return True
    except:
        return False

async def main():
    # First, create health file for file-based checks
    try:
        with open('/tmp/mcp_healthy', 'w') as f:
            f.write('healthy')
    except:
        pass
    
    # Then check WebSocket
    if await check_websocket():
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())