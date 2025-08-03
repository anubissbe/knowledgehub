#!/usr/bin/env python3
"""Test tasks endpoint to see detailed error"""

import requests
import json

try:
    response = requests.get("http://localhost:3001/tasks")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")