#!/usr/bin/env python3
"""Test available API endpoints."""

import httpx
import asyncio
import json

API_BASE = "http://localhost:3000"

async def test_endpoints():
    """Test various endpoint combinations."""
    
    endpoints_to_test = [
        # Try different path combinations
        ("/api/rag/search", "POST", {"query": "test"}),
        ("/api/rag/enhanced/search", "POST", {"query": "test"}),
        ("/rag/enhanced/search", "POST", {"query": "test"}),
        ("/api/agent/workflow", "POST", {"query": "test"}),
        ("/api/agent/workflows/execute", "POST", {"query": "test"}),
        ("/agent/workflows/execute", "POST", {"query": "test"}),
        ("/api/rag/query", "POST", {"query": "test"}),
        ("/api/memory/session", "POST", {"session_id": "test"}),
        ("/api/memory/sessions", "GET", None),
        ("/api/docs", "GET", None),
        ("/api/openapi.json", "GET", None),
    ]
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        print("Testing API endpoints:\n")
        
        working_endpoints = []
        
        for path, method, data in endpoints_to_test:
            try:
                url = f"{API_BASE}{path}"
                
                if method == "GET":
                    response = await client.get(url)
                else:
                    response = await client.post(url, json=data or {})
                
                if response.status_code == 404:
                    status = "❌ 404 Not Found"
                elif response.status_code == 200:
                    status = "✅ 200 OK"
                    working_endpoints.append(path)
                elif response.status_code == 422:
                    status = "⚠️ 422 Validation Error"
                    working_endpoints.append(path)
                elif response.status_code == 401:
                    status = "🔒 401 Unauthorized"
                    working_endpoints.append(path)
                elif response.status_code == 400:
                    status = "⚠️ 400 Bad Request"
                    working_endpoints.append(path)
                else:
                    status = f"🔸 {response.status_code}"
                    if response.status_code != 404:
                        working_endpoints.append(path)
                
                print(f"{status:20} {method:6} {path}")
                
            except Exception as e:
                print(f"❌ Error           {method:6} {path}: {str(e)[:50]}")
        
        # Try to get OpenAPI spec to see all routes
        print("\n" + "="*60)
        print("Checking OpenAPI spec for available routes...")
        
        try:
            response = await client.get(f"{API_BASE}/api/openapi.json")
            if response.status_code == 200:
                openapi = response.json()
                paths = openapi.get("paths", {})
                
                print(f"\nFound {len(paths)} documented endpoints:")
                
                # Group by tag
                by_tag = {}
                for path, methods in paths.items():
                    for method, details in methods.items():
                        if method in ["get", "post", "put", "delete", "patch"]:
                            tags = details.get("tags", ["untagged"])
                            for tag in tags:
                                if tag not in by_tag:
                                    by_tag[tag] = []
                                by_tag[tag].append(f"{method.upper():6} {path}")
                
                # Print grouped endpoints
                for tag in sorted(by_tag.keys()):
                    print(f"\n📦 {tag}:")
                    for endpoint in sorted(by_tag[tag])[:5]:  # Show first 5 per tag
                        print(f"  {endpoint}")
                    if len(by_tag[tag]) > 5:
                        print(f"  ... and {len(by_tag[tag]) - 5} more")
        except Exception as e:
            print(f"Could not fetch OpenAPI spec: {e}")
        
        print("\n" + "="*60)
        print(f"Summary: {len(working_endpoints)} endpoints responding (not 404)")
        if working_endpoints:
            print("\nWorking endpoints:")
            for ep in working_endpoints[:10]:
                print(f"  ✅ {ep}")

asyncio.run(test_endpoints())