#!/usr/bin/env python3
"""
Start KnowledgeHub API locally without Docker
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, '/opt/projects/knowledgehub')

# Set environment variables
os.environ['DATABASE_HOST'] = 'localhost'
os.environ['DATABASE_PORT'] = '5433'
os.environ['DATABASE_NAME'] = 'knowledgehub'
os.environ['DATABASE_USER'] = 'knowledgehub'
os.environ['DATABASE_PASSWORD'] = 'knowledgehub'
os.environ['REDIS_HOST'] = 'localhost'
os.environ['REDIS_PORT'] = '6381'
os.environ['WEAVIATE_HOST'] = 'localhost'
os.environ['WEAVIATE_PORT'] = '8090'
os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
os.environ['NEO4J_USER'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'knowledgehub123'
os.environ['TIMESCALE_HOST'] = 'localhost'
os.environ['TIMESCALE_PORT'] = '5434'
os.environ['MINIO_ENDPOINT'] = 'localhost:9010'
os.environ['MINIO_ACCESS_KEY'] = 'minioadmin'
os.environ['MINIO_SECRET_KEY'] = 'minioadmin'
os.environ['AI_SERVICE_URL'] = 'http://localhost:8000'

# Start the API
import uvicorn
from api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=False)