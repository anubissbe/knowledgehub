#!/usr/bin/env python3
"""Simple runner that handles environment variables properly"""

import os
import sys
import uvicorn

# Add the app directory to Python path
sys.path.insert(0, '/app')

# Fix CORS_ORIGINS if it's a plain string
cors_origins = os.environ.get('CORS_ORIGINS', '*')
if cors_origins == '*':
    os.environ['CORS_ORIGINS'] = '["*"]'
elif not cors_origins.startswith('['):
    # Convert comma-separated to JSON array
    origins = [f'"{o.strip()}"' for o in cors_origins.split(',')]
    os.environ['CORS_ORIGINS'] = f'[{",".join(origins)}]'

# Import the app
from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get('API_HOST', '0.0.0.0'),
        port=int(os.environ.get('API_PORT', '3000')),
        reload=False
    )