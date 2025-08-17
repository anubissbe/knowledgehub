#!/usr/bin/env python3
"""
Simple RAG Systems Test Script
Tests the available RAG implementations directly without relying on the API
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

async def test_weaviate_connection():
    """Test direct connection to Weaviate"""
    print("=== Testing Weaviate Connection ===")
    try:
        import weaviate
        client = weaviate.Client("http://localhost:8090")
        
        # Test connection
        meta = client.get_meta()
        print(f"✅ Weaviate connected successfully")
        print(f"   Version: {meta.get('version', 'Unknown')}")
        
        # List available classes
        try:
            schema = client.schema.get()
            classes = [cls["class"] for cls in schema.get("classes", [])]
            print(f"   Available classes: {classes}")
        except Exception as e:
            print(f"   Warning: Could not get schema: {e}")
            
        return True, {"version": meta.get("version"), "classes": classes if "classes" in locals() else []}
    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        return False, str(e)

async def test_simple_embedding():
    """Test basic embedding functionality"""
    print("\n=== Testing Embedding Service ===")
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try to load a simple model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_text = "This is a test document for embedding generation."
        
        embedding = model.encode(test_text)
        print(f"✅ Embedding generation successful")
        print(f"   Text: {test_text[:50]}...")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        return True, {
            "model": "all-MiniLM-L6-v2",
            "dimension": len(embedding),
            "test_text": test_text
        }
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False, str(e)

async def test_basic_chunking():
    """Test simple text chunking"""
    print("\n=== Testing Text Chunking ===")
    try:
        test_content = """
        # FastAPI Documentation

        FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

        ## Key Features

        - Fast: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic).
        - Fast to code: Increase the speed to develop features by about 200% to 300%.
        - Fewer bugs: Reduce about 40% of human (developer) induced errors.
        - Intuitive: Great editor support. Completion everywhere. Less time debugging.
        - Easy: Designed to be easy to use and learn. Less time reading docs.
        - Short: Minimize code duplication. Multiple features from each parameter declaration.
        - Robust: Get production-ready code. With automatic interactive documentation.
        - Standards-based: Based on (and fully compatible with) the open standards for APIs: OpenAPI and JSON Schema.

        ## Installation

        ```bash
        pip install fastapi
        pip install "uvicorn[standard]"
        ```

        ## Example

        Create a file main.py with:

        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/")
        def read_root():
            return {"Hello": "World"}

        @app.get("/items/{item_id}")
        def read_item(item_id: int, q: str = None):
            return {"item_id": item_id, "q": q}
        ```

        Then run the server with:

        ```bash
        uvicorn main:app --reload
        ```
        """
        
        # Simple chunking logic
        chunk_size = 500
        chunk_overlap = 50
        chunks = []
        
        for i in range(0, len(test_content), chunk_size - chunk_overlap):
            chunk = test_content[i:i + chunk_size]
            if chunk.strip():
                chunks.append({
                    "content": chunk.strip(),
                    "start": i,
                    "end": min(i + chunk_size, len(test_content)),
                    "index": len(chunks)
                })
        
        print(f"✅ Text chunking successful")
        print(f"   Original length: {len(test_content)} characters")
        print(f"   Created chunks: {len(chunks)}")
        print(f"   Average chunk size: {sum(len(c['content']) for c in chunks) / len(chunks):.0f} chars")
        
        return True, {
            "original_length": len(test_content),
            "chunk_count": len(chunks),
            "chunks": chunks
        }
    except Exception as e:
        print(f"❌ Text chunking failed: {e}")
        return False, str(e)

async def test_weaviate_ingestion_retrieval():
    """Test document ingestion and retrieval with Weaviate"""
    print("\n=== Testing Weaviate Ingestion & Retrieval ===")
    try:
        import weaviate
        from sentence_transformers import SentenceTransformer
        import uuid
        
        client = weaviate.Client("http://localhost:8090")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define test class name
        class_name = "TestDocument"
        
        # Create schema if it doesn't exist
        try:
            existing_classes = [cls["class"] for cls in client.schema.get().get("classes", [])]
            if class_name not in existing_classes:
                schema = {
                    "class": class_name,
                    "description": "Test documents for RAG testing",
                    "vectorizer": "none",  # We'll provide vectors manually
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The document content"
                        },
                        {
                            "name": "title",
                            "dataType": ["string"],
                            "description": "Document title"
                        },
                        {
                            "name": "source",
                            "dataType": ["string"],
                            "description": "Document source"
                        }
                    ]
                }
                client.schema.create_class(schema)
                print(f"   Created schema for class: {class_name}")
        except Exception as e:
            print(f"   Schema creation note: {e}")
        
        # Test document
        test_doc = {
            "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It provides automatic API documentation, type checking, and high performance.",
            "title": "FastAPI Introduction",
            "source": "test"
        }
        
        # Generate embedding
        embedding = model.encode(test_doc["content"])
        
        # Ingest document
        doc_id = str(uuid.uuid4())
        client.data_object.create(
            class_name=class_name,
            uuid=doc_id,
            data_object=test_doc,
            vector=embedding
        )
        print(f"✅ Document ingested successfully")
        print(f"   Document ID: {doc_id}")
        
        # Test retrieval by similarity
        query = "What is FastAPI and how is it used for building APIs?"
        query_embedding = model.encode(query)
        
        results = client.query.get(
            class_name, 
            ["content", "title", "source"]
        ).with_near_vector({
            "vector": query_embedding,
            "certainty": 0.7
        }).with_limit(5).do()
        
        retrieved_docs = results.get("data", {}).get("Get", {}).get(class_name, [])
        
        print(f"✅ Document retrieval successful")
        print(f"   Query: {query}")
        print(f"   Retrieved documents: {len(retrieved_docs)}")
        
        if retrieved_docs:
            print(f"   Best match: {retrieved_docs[0]['title']}")
            print(f"   Content preview: {retrieved_docs[0]['content'][:100]}...")
        
        # Clean up
        try:
            client.data_object.delete(doc_id, class_name)
            print(f"   Cleaned up test document")
        except:
            pass
        
        return True, {
            "document_id": doc_id,
            "query": query,
            "results_count": len(retrieved_docs),
            "first_result": retrieved_docs[0] if retrieved_docs else None
        }
        
    except Exception as e:
        print(f"❌ Weaviate ingestion/retrieval failed: {e}")
        return False, str(e)

async def test_llamaindex_availability():
    """Test if LlamaIndex components are available"""
    print("\n=== Testing LlamaIndex Availability ===")
    try:
        # Try basic imports
        from llama_index.core import VectorStoreIndex, Document
        from llama_index.core.node_parser import SentenceSplitter
        
        print("✅ LlamaIndex core components available")
        
        # Try advanced imports
        try:
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            print("✅ LlamaIndex vector store and embedding components available")
            advanced_available = True
        except ImportError as e:
            print(f"⚠️  Advanced LlamaIndex components not available: {e}")
            advanced_available = False
        
        return True, {
            "core_available": True,
            "advanced_available": advanced_available
        }
        
    except ImportError as e:
        print(f"❌ LlamaIndex not available: {e}")
        return False, str(e)

async def test_simple_rag_pipeline():
    """Test a complete simple RAG pipeline"""
    print("\n=== Testing Simple RAG Pipeline ===")
    try:
        from sentence_transformers import SentenceTransformer
        import weaviate
        import uuid
        
        # Initialize components
        client = weaviate.Client("http://localhost:8090")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        class_name = "SimpleRAGTest"
        
        # Setup schema
        try:
            existing_classes = [cls["class"] for cls in client.schema.get().get("classes", [])]
            if class_name not in existing_classes:
                schema = {
                    "class": class_name,
                    "vectorizer": "none",
                    "properties": [
                        {"name": "content", "dataType": ["text"]},
                        {"name": "title", "dataType": ["string"]},
                        {"name": "chunk_index", "dataType": ["int"]}
                    ]
                }
                client.schema.create_class(schema)
        except Exception as e:
            print(f"   Schema setup note: {e}")
        
        # Test documents
        documents = [
            {
                "title": "FastAPI Installation",
                "content": "To install FastAPI, run: pip install fastapi. You also need an ASGI server like uvicorn: pip install uvicorn[standard]"
            },
            {
                "title": "FastAPI Hello World", 
                "content": "Create a simple FastAPI app by importing FastAPI, creating an app instance, and adding route handlers with decorators like @app.get('/')"
            },
            {
                "title": "FastAPI Benefits",
                "content": "FastAPI provides automatic API documentation, type validation, high performance, and is based on standard Python type hints"
            }
        ]
        
        # Ingest documents
        doc_ids = []
        print(f"   Ingesting {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            embedding = model.encode(doc["content"])
            doc_id = str(uuid.uuid4())
            
            client.data_object.create(
                class_name=class_name,
                uuid=doc_id,
                data_object={
                    "content": doc["content"],
                    "title": doc["title"],
                    "chunk_index": i
                },
                vector=embedding
            )
            doc_ids.append(doc_id)
        
        print(f"✅ Ingested {len(doc_ids)} documents")
        
        # Test queries
        queries = [
            "How do I install FastAPI?",
            "How to create a simple FastAPI application?",
            "What are the benefits of using FastAPI?"
        ]
        
        results_summary = []
        for query in queries:
            query_embedding = model.encode(query)
            
            results = client.query.get(
                class_name, 
                ["content", "title", "chunk_index"]
            ).with_near_vector({
                "vector": query_embedding,
                "certainty": 0.6
            }).with_limit(2).do()
            
            retrieved = results.get("data", {}).get("Get", {}).get(class_name, [])
            
            results_summary.append({
                "query": query,
                "results_count": len(retrieved),
                "top_result": retrieved[0]["title"] if retrieved else "No results"
            })
            
            print(f"   Query: {query}")
            print(f"   Top result: {retrieved[0]['title'] if retrieved else 'No results'}")
        
        # Cleanup
        for doc_id in doc_ids:
            try:
                client.data_object.delete(doc_id, class_name)
            except:
                pass
        
        print(f"✅ Simple RAG pipeline test completed successfully")
        
        return True, {
            "documents_ingested": len(doc_ids),
            "queries_tested": len(queries),
            "results": results_summary
        }
        
    except Exception as e:
        print(f"❌ Simple RAG pipeline failed: {e}")
        return False, str(e)

async def main():
    """Run all RAG system tests"""
    print("🔍 KnowledgeHub RAG Systems Test Suite")
    print("=====================================")
    
    test_results = {}
    
    # Test 1: Weaviate Connection
    success, result = await test_weaviate_connection()
    test_results["weaviate_connection"] = {"success": success, "result": result}
    
    # Test 2: Embedding Service
    success, result = await test_simple_embedding()
    test_results["embedding_service"] = {"success": success, "result": result}
    
    # Test 3: Text Chunking
    success, result = await test_basic_chunking()
    test_results["text_chunking"] = {"success": success, "result": result}
    
    # Test 4: Weaviate Ingestion/Retrieval
    success, result = await test_weaviate_ingestion_retrieval()
    test_results["weaviate_operations"] = {"success": success, "result": result}
    
    # Test 5: LlamaIndex Availability
    success, result = await test_llamaindex_availability()
    test_results["llamaindex_availability"] = {"success": success, "result": result}
    
    # Test 6: Complete RAG Pipeline
    success, result = await test_simple_rag_pipeline()
    test_results["complete_rag_pipeline"] = {"success": success, "result": result}
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for t in test_results.values() if t["success"])
    
    for test_name, test_result in test_results.items():
        status = "✅ PASSED" if test_result["success"] else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All RAG systems are functional!")
    else:
        print(f"⚠️  {total_tests - passed_tests} systems need attention")
    
    # Save detailed results
    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests
        },
        "tests": test_results
    }
    
    with open("rag_test_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: rag_test_results.json")
    
    return test_results

if __name__ == "__main__":
    asyncio.run(main())