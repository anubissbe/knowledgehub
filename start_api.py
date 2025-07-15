#!/usr/bin/env python3
"""
KnowledgeHub API Startup Script
"""
import uvicorn
from api.main import app

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        log_level="info"
    )