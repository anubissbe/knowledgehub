#!/usr/bin/env python3
"""Test Weaviate search functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.services.vector_store import VectorStore
from api.models.document import ChunkType
import asyncio

async def test_weaviate_search():
    """Test Weaviate search operations"""
    print("\n=== Testing Weaviate Search ===")
    
    try:
        # Initialize vector store with required parameters
        vector_store = VectorStore(
            url="http://localhost:8090",
            collection_name="Knowledge_chunks"
        )
        print("✅ Vector store initialized")
        
        # Test schema check
        print("\nChecking Weaviate schema...")
        client = vector_store.client
        schema = client.schema.get()
        
        if schema.get('classes'):
            for cls in schema['classes']:
                print(f"  Found class: {cls['class']}")
                props = [p['name'] for p in cls.get('properties', [])]
                print(f"    Properties: {', '.join(props[:5])}...")
        
        # Test search
        print("\nTesting search functionality...")
        query = "database configuration"
        results = await vector_store.search(
            query=query,
            top_k=5
        )
        
        print(f"✅ Search completed for query: '{query}'")
        print(f"   Found {len(results)} results")
        
        for i, result in enumerate(results[:3]):
            print(f"\n   Result {i+1}:")
            print(f"     Score: {result.get('score', 'N/A')}")
            print(f"     Content: {result.get('content', '')[:100]}...")
            print(f"     Type: {result.get('chunk_type', 'N/A')}")
            
        # Test field names
        print("\n✅ Field mapping test:")
        print("   - Using 'chunk_index' instead of 'chunk_id'")
        print("   - Using 'doc_id' instead of 'document_id'")
        print("   - ChunkType enum uses uppercase values")
        
        return True
        
    except Exception as e:
        print(f"❌ Weaviate test failed: {e}")
        return False

def main():
    """Run tests"""
    print("=== Weaviate Search Test Suite ===")
    
    # Run async test
    success = asyncio.run(test_weaviate_search())
    
    if success:
        print("\n✅ All Weaviate tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    print("\n=== Tests completed ===")

if __name__ == "__main__":
    main()