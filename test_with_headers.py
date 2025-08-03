#\!/usr/bin/env python3
"""Test with proper headers"""

import requests
import json

BASE_URL = "http://localhost:3000"

# Headers that match the successful curl commands
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Content-Type": "application/json"
}

# Test sources endpoint
print("Testing Sources endpoint:")
response = requests.get(f"{BASE_URL}/api/v1/sources", headers=headers)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Success\! Found {data['total']} sources")
else:
    print(f"Response: {response.text[:200]}")

print("\nTesting WebSocket status:")
response = requests.get(f"{BASE_URL}/api/v1/ws/websocket/status", headers=headers)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("Success\! WebSocket status endpoint is accessible")
