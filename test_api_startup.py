#!/usr/bin/env python3
"""
Minimal test to check if API can start.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing API import...")
    from api.main import app
    print("✅ API module imported successfully!")
    
    # Check if app is a FastAPI instance
    from fastapi import FastAPI
    if isinstance(app, FastAPI):
        print("✅ FastAPI app instance created successfully!")
        
        # List routes
        print("\nAPI Routes:")
        for route in app.routes[:10]:  # Show first 10 routes
            if hasattr(route, 'path'):
                print(f"  - {route.path}")
    
    print("\n✅ API is ready to start!")
    print("You can now run: python3 -m uvicorn api.main:app --host 0.0.0.0 --port 3000")
    
except Exception as e:
    print(f"❌ Failed to import API: {e}")
    import traceback
    traceback.print_exc()