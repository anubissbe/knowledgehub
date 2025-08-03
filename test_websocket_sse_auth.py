#!/usr/bin/env python3
"""Test WebSocket and SSE authentication"""

import asyncio
import websockets
import aiohttp
import json

BASE_URL = "localhost:3000"


async def test_websocket_anonymous():
    """Test anonymous WebSocket connection"""
    print("\n=== Testing Anonymous WebSocket Connection ===")
    
    try:
        uri = f"ws://{BASE_URL}/ws/notifications"
        async with websockets.connect(uri) as websocket:
            # Wait for welcome message
            welcome = await websocket.recv()
            data = json.loads(welcome)
            print(f"Welcome message: {json.dumps(data, indent=2)}")
            
            assert data["type"] == "connected"
            assert data["authenticated"] == False
            assert data["user_id"] == "anonymous"
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            pong = await websocket.recv()
            print(f"Ping response: {pong}")
            
            print("✅ Anonymous WebSocket connection successful")
            
    except Exception as e:
        print(f"❌ Anonymous WebSocket failed: {e}")


async def test_websocket_authenticated():
    """Test authenticated WebSocket connection"""
    print("\n=== Testing Authenticated WebSocket Connection ===")
    
    try:
        # Simulate auth token
        token = "test-auth-token-12345"
        uri = f"ws://{BASE_URL}/ws/notifications?token={token}"
        
        async with websockets.connect(uri) as websocket:
            # Wait for welcome message
            welcome = await websocket.recv()
            data = json.loads(welcome)
            print(f"Welcome message: {json.dumps(data, indent=2)}")
            
            assert data["type"] == "connected"
            assert data["authenticated"] == True
            assert data["user_id"] == "authenticated_user"
            
            print("✅ Authenticated WebSocket connection successful")
            
    except Exception as e:
        print(f"❌ Authenticated WebSocket failed: {e}")


async def test_sse_anonymous():
    """Test anonymous SSE connection"""
    print("\n=== Testing Anonymous SSE Connection ===")
    
    try:
        url = f"http://{BASE_URL}/api/realtime/stream"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                assert response.status == 200
                assert response.headers["Content-Type"] == "text/event-stream"
                
                # Read first few events
                count = 0
                async for line in response.content:
                    if count > 3:
                        break
                    
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data:"):
                        print(f"SSE Event: {line_str[:50]}...")
                        count += 1
                
                print("✅ Anonymous SSE connection successful")
                
    except Exception as e:
        print(f"❌ Anonymous SSE failed: {e}")


async def test_sse_authenticated():
    """Test authenticated SSE connection"""
    print("\n=== Testing Authenticated SSE Connection ===")
    
    try:
        token = "test-auth-token-12345"
        url = f"http://{BASE_URL}/api/realtime/stream?token={token}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                assert response.status == 200
                assert response.headers["Content-Type"] == "text/event-stream"
                
                # Read first event
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data:"):
                        print(f"SSE Event (authenticated): {line_str[:50]}...")
                        break
                
                print("✅ Authenticated SSE connection successful")
                
    except Exception as e:
        print(f"❌ Authenticated SSE failed: {e}")


async def main():
    """Run all tests"""
    print("=== Testing WebSocket and SSE Authentication ===")
    
    # Wait a bit for API to be ready
    await asyncio.sleep(2)
    
    # Run tests
    await test_websocket_anonymous()
    await test_websocket_authenticated()
    await test_sse_anonymous()
    await test_sse_authenticated()
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())