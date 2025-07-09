#!/bin/bash
# Health check script for RAG processor

# Check if the health endpoint responds
if curl -f http://localhost:3013/health >/dev/null 2>&1; then
    exit 0
fi

# Health check failed
exit 1