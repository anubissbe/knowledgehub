#!/bin/bash
# Health check script for MCP server

# Check if the health file exists and was updated recently
if [ -f /tmp/mcp_healthy ]; then
    # Check if file was modified in the last 60 seconds
    if [ $(find /tmp/mcp_healthy -mmin -1 | wc -l) -gt 0 ]; then
        exit 0
    fi
fi

# Fallback: Try to connect to the WebSocket port
if nc -z localhost 3002 2>/dev/null; then
    exit 0
fi

# Health check failed
exit 1