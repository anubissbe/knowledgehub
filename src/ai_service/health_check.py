#!/usr/bin/env python3
"""Health check for AI service"""
import sys
try:
    import requests
    response = requests.get('http://localhost:8000/health', timeout=5)
    if response.status_code == 200:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print(f"Health check failed: {e}")
    sys.exit(1)