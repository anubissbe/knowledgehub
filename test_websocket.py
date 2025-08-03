#!/usr/bin/env python3
"""Test WebSocket functionality"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:3000/ws/notifications"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established")
            
            # Send a test message
            test_message = {
                "type": "ping",
                "timestamp": "2025-07-21T10:00:00Z"
            }
            await websocket.send(json.dumps(test_message))
            print(f"📤 Sent: {test_message}")
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"📥 Received: {response}")
            
            print("✅ WebSocket test passed!")
            
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
    except asyncio.TimeoutError:
        print("❌ WebSocket timeout - no response received")
    except ConnectionRefusedError:
        print("❌ Connection refused - API not running on port 3000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())