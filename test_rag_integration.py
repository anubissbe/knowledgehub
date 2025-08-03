#!/usr/bin/env python3
"""
Test RAG Integration with KnowledgeHub
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.services.rag.simple_rag_service import SimpleRAGService
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

async def test_rag_integration():
    """Test the RAG system integration"""
    print("🧪 Testing RAG Integration...")
    
    # Test 1: Qdrant Connection
    print("\n1️⃣ Testing Qdrant connection...")
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        info = qdrant_client.get_collections()
        print(f"✅ Qdrant connected! Collections: {info}")
        
        # Create a test collection
        collection_name = "test_rag_collection"
        try:
            qdrant_client.delete_collection(collection_name)
        except:
            pass
            
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"✅ Created test collection: {collection_name}")
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False
        
    # Test 2: RAG Service
    print("\n2️⃣ Testing RAG service...")
    try:
        rag_service = SimpleRAGService()
        print("✅ RAG service initialized")
        
        # Test document ingestion
        test_doc = """
        # FastAPI Documentation
        
        FastAPI is a modern, fast web framework for building APIs with Python 3.7+.
        
        ## Installation
        ```bash
        pip install fastapi
        pip install uvicorn[standard]
        ```
        
        ## Quick Start
        Create a file main.py:
        ```python
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"Hello": "World"}
        ```
        """
        
        result = await rag_service.ingest_document(
            content=test_doc,
            metadata={
                "title": "FastAPI Quick Start",
                "source": "test",
                "type": "documentation"
            }
        )
        print(f"✅ Document ingested: {result}")
        
        # Test query
        query_result = await rag_service.query(
            query_text="How do I install FastAPI?",
            user_id="test-user",
            top_k=3
        )
        print(f"✅ Query executed: {query_result['response'][:200]}...")
        
    except Exception as e:
        print(f"❌ RAG service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # Test 3: Contextual Enrichment
    print("\n3️⃣ Testing contextual enrichment...")
    try:
        from api.services.rag.contextual_enrichment import get_enrichment_service
        enrichment_service = get_enrichment_service()
        
        enriched_text, metadata = await enrichment_service.enrich_chunk(
            chunk_text="pip install fastapi",
            chunk_metadata={"type": "code"},
            document_context={
                "title": "FastAPI Installation",
                "doc_type": "documentation"
            }
        )
        print(f"✅ Enrichment working: {enriched_text[:100]}...")
        
    except Exception as e:
        print(f"⚠️  Enrichment service not available (expected): {e}")
        
    print("\n✅ RAG Integration Test Completed Successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_rag_integration())
    sys.exit(0 if success else 1)