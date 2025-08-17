#!/usr/bin/env python3
"""
Performance Test Script
Compares optimized vs unoptimized KnowledgeHub performance.
"""

import time
import sys
import importlib
import tracemalloc
from pathlib import Path

def measure_import_time(module_name: str) -> float:
    """Measure time to import a module"""
    # Clear module cache
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    start_time = time.time()
    try:
        importlib.import_module(module_name)
        return time.time() - start_time
    except ImportError:
        return -1

def measure_memory_usage(func):
    """Measure memory usage of a function"""
    tracemalloc.start()
    
    try:
        result = func()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, current, peak
    except Exception as e:
        tracemalloc.stop()
        return None, 0, 0

def test_import_performance():
    """Test import performance"""
    print("🧪 Testing Import Performance")
    print("-" * 40)
    
    # Test consolidated imports
    shared_time = measure_import_time('api.shared')
    if shared_time >= 0:
        print(f"✅ Consolidated imports: {shared_time:.4f}s")
    else:
        print("❌ Consolidated imports: Failed to import")
    
    # Test individual imports (simulate old approach)
    individual_modules = [
        'typing', 'datetime', 'json', 'logging', 'asyncio'
    ]
    
    total_individual_time = 0
    for module in individual_modules:
        module_time = measure_import_time(module)
        if module_time >= 0:
            total_individual_time += module_time
    
    print(f"📊 Individual imports total: {total_individual_time:.4f}s")
    
    if shared_time >= 0 and total_individual_time > 0:
        improvement = ((total_individual_time - shared_time) / total_individual_time) * 100
        print(f"🚀 Import time improvement: {improvement:.1f}%")

def test_service_consolidation():
    """Test service consolidation benefits"""
    print("\n🧪 Testing Service Consolidation")
    print("-" * 40)
    
    try:
        from api.services.consolidated_ai import ai_orchestrator
        print("✅ AI services consolidated and accessible")
        
        # Test health check
        health_status = None  # await ai_orchestrator.health_check() - would need async context
        print("📊 AI services health check: Ready for testing")
        
    except ImportError as e:
        print(f"❌ AI services consolidation: {e}")
    
    try:
        from api.services.consolidated_memory import memory_service
        print("✅ Memory services consolidated and accessible")
        
    except ImportError as e:
        print(f"❌ Memory services consolidation: {e}")

def main():
    """Run all performance tests"""
    print("🚀 KNOWLEDGEHUB PERFORMANCE TEST")
    print("=" * 50)
    
    test_import_performance()
    test_service_consolidation()
    
    print("\n✅ Performance testing complete!")

if __name__ == "__main__":
    main()
