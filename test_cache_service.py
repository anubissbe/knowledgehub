#!/usr/bin/env python3
"""Test cache service import"""

import sys
sys.path.insert(0, '/opt/projects/knowledgehub')

import asyncio

async def test_imports():
    """Test that all imports work correctly"""
    
    try:
        # Test cache service
        from api.services.cache import get_cache_service
        cache = await get_cache_service()
        print("✓ Cache service import successful")
        
        # Test pattern engine
        from api.services.pattern_recognition_engine import get_pattern_engine
        engine = await get_pattern_engine()
        print("✓ Pattern engine import successful")
        
        # Test learning pipeline
        from api.services.realtime_learning_pipeline import get_learning_pipeline
        pipeline = await get_learning_pipeline()
        print("✓ Learning pipeline import successful")
        
        # Test background jobs
        from api.services.background_jobs import BackgroundJobManager
        job_manager = BackgroundJobManager()
        await job_manager.initialize()
        print("✓ Background job manager initialized successfully")
        
        # Test pattern workers
        from api.services.pattern_workers import PatternWorkerManager
        pattern_manager = PatternWorkerManager()
        await pattern_manager.initialize()
        print("✓ Pattern worker manager initialized successfully")
        
        print("\nAll imports and initializations successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_imports())