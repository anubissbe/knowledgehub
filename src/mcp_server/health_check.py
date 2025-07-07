#!/usr/bin/env python3
"""Simple health check that writes to file"""
import sys
import os

# Write health status to file
try:
    with open('/tmp/mcp_healthy', 'w') as f:
        f.write('healthy')
    print("Health check passed")
    sys.exit(0)
except Exception as e:
    print(f"Health check failed: {e}")
    sys.exit(1)