#!/usr/bin/env python3
"""Test the KnowledgeHub MCP Server"""

import asyncio
import json
import subprocess
import sys
import tempfile

async def test_mcp_server():
    """Test the MCP server functionality"""
    
    print("🧪 Testing KnowledgeHub MCP Server...")
    
    # Test data
    test_requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        }
    ]
    
    # Start MCP server process
    try:
        process = subprocess.Popen(
            [sys.executable, "/opt/projects/knowledgehub/knowledgehub_mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send test requests
        for i, request in enumerate(test_requests):
            request_str = json.dumps(request) + "\n"
            print(f"📤 Sending request {i+1}: {request['method']}")
            
            process.stdin.write(request_str)
            process.stdin.flush()
            
            # Read response (simplified)
            try:
                response_line = process.stdout.readline()
                if response_line:
                    response = json.loads(response_line)
                    print(f"✅ Received response: {response.get('id')} - {len(str(response))} chars")
                else:
                    print(f"❌ No response received")
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON response: {e}")
            except Exception as e:
                print(f"❌ Error reading response: {e}")
        
        # Cleanup
        process.terminate()
        process.wait(timeout=5)
        
        print("✅ MCP Server test completed")
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_mcp_server())