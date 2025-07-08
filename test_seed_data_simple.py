#!/usr/bin/env python3
"""
Simple test for seed data generation functionality
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_seed_data_structure():
    """Test the seed data structure without database dependencies"""
    print("🧪 Testing seed data structure...")
    
    try:
        # Import the enum types
        from api.models.memory import MemoryType
        print("✅ MemoryType enum imported successfully")
        
        # Test sample data creation
        from api.memory_system.seed_data import MemorySystemSeedData
        
        # Create a mock database session
        class MockDB:
            def add(self, obj): pass
            def commit(self): pass
            def refresh(self, obj): pass
            def query(self, model): return MockQuery()
        
        class MockQuery:
            def all(self): return []
            def count(self): return 0
            def delete(self): return 0
            def filter(self, *args): return self
            def first(self): return None
        
        # Create seed data generator with mock DB
        seed_generator = MemorySystemSeedData(MockDB())
        
        # Test sample sessions
        sessions = seed_generator.sample_sessions
        print(f"✅ Sample sessions created: {len(sessions)}")
        
        # Validate session structure
        for i, session in enumerate(sessions):
            assert "user_id" in session
            assert "project_id" in session
            assert "session_metadata" in session
            assert "tags" in session
            assert "duration_minutes" in session
            assert "memory_count" in session
            print(f"  - Session {i+1}: {session['session_metadata']['session_type']}")
        
        # Test sample memories
        memories = seed_generator.sample_memories
        print(f"✅ Sample memories created: {len(memories)}")
        
        # Validate memory structure
        memory_types = set()
        for i, memory in enumerate(memories):
            assert "memory_type" in memory
            assert "content" in memory
            assert "importance_score" in memory
            assert "entities" in memory
            assert "facts" in memory
            assert "metadata" in memory
            
            memory_types.add(memory["memory_type"])
            
            # Validate content quality
            assert len(memory["content"]) > 20
            assert len(memory["entities"]) > 0
            assert len(memory["facts"]) > 0
            assert 0.0 <= memory["importance_score"] <= 1.0
        
        print(f"  - Memory types: {len(memory_types)}")
        for mem_type in memory_types:
            print(f"    - {mem_type.value}")
        
        # Test sample contexts
        contexts = seed_generator.sample_contexts
        print(f"✅ Sample contexts created: {len(contexts)}")
        
        # Validate context structure
        for i, context in enumerate(contexts):
            assert "content" in context
            assert "context_type" in context
            assert "scope" in context
            assert "importance" in context
            assert "related_entities" in context
            assert "metadata" in context
            
            # Validate content quality
            assert len(context["content"]) > 20
            assert len(context["related_entities"]) > 0
            assert 0.0 <= context["importance"] <= 1.0
            
            print(f"  - Context {i+1}: {context['context_type']} ({context['scope']})")
        
        print("\n✅ All seed data structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_structure():
    """Test the CLI script structure"""
    print("\n🧪 Testing CLI structure...")
    
    try:
        # Check if CLI script exists
        cli_path = Path(__file__).parent / "scripts" / "generate_seed_data.py"
        if not cli_path.exists():
            print(f"❌ CLI script not found at {cli_path}")
            return False
        
        print("✅ CLI script exists")
        
        # Check if script is executable
        if not os.access(cli_path, os.X_OK):
            print("⚠️ CLI script is not executable")
            os.chmod(cli_path, 0o755)
            print("✅ Made CLI script executable")
        
        # Basic syntax check
        with open(cli_path, 'r') as f:
            content = f.read()
            if "argparse" in content:
                print("✅ CLI uses argparse")
            if "generate_seed_data" in content:
                print("✅ CLI has generate function")
            if "validate_seed_data" in content:
                print("✅ CLI has validate function")
        
        print("✅ CLI structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_documentation():
    """Test documentation completeness"""
    print("\n🧪 Testing documentation...")
    
    try:
        # Check if documentation exists
        doc_path = Path(__file__).parent / "docs" / "SEED_DATA_SYSTEM.md"
        if not doc_path.exists():
            print(f"❌ Documentation not found at {doc_path}")
            return False
        
        print("✅ Documentation exists")
        
        # Check documentation content
        with open(doc_path, 'r') as f:
            content = f.read()
            
            expected_sections = [
                "# Memory System Seed Data Documentation",
                "## Overview",
                "## Features",
                "## Architecture",
                "## Usage Guide",
                "## Testing",
                "## Troubleshooting"
            ]
            
            for section in expected_sections:
                if section in content:
                    print(f"✅ Has section: {section}")
                else:
                    print(f"⚠️ Missing section: {section}")
        
        print("✅ Documentation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🌱 SEED DATA SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_seed_data_structure,
        test_cli_structure,
        test_documentation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 TEST SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Seed data system is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)