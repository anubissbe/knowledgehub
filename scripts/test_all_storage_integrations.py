#!/usr/bin/env python3
"""
Comprehensive test of all storage system integrations.

Tests:
- PostgreSQL (main database)
- Redis (caching)
- Weaviate (vector storage)
- Neo4j (knowledge graph)
- TimescaleDB (time-series analytics)
- MinIO (object storage)
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import json

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_postgresql():
    """Test PostgreSQL main database connection."""
    try:
        logger.info("🐘 Testing PostgreSQL connection...")
        
        from api.models import get_db
        from sqlalchemy import text
        
        db = next(get_db())
        result = db.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        logger.info(f"✅ PostgreSQL connected: {version[:50]}...")
        
        # Test basic table operations
        db.execute(text("CREATE TABLE IF NOT EXISTS _test_storage (id SERIAL PRIMARY KEY, data TEXT);"))
        db.execute(text("INSERT INTO _test_storage (data) VALUES ('test_data');"))
        result = db.execute(text("SELECT data FROM _test_storage LIMIT 1;"))
        data = result.fetchone()[0]
        assert data == 'test_data'
        db.execute(text("DROP TABLE _test_storage;"))
        db.commit()
        
        logger.info("✅ PostgreSQL operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ PostgreSQL test failed: {e}")
        return False


async def test_redis():
    """Test Redis cache connection."""
    try:
        logger.info("🔴 Testing Redis connection...")
        
        from api.services.cache import redis_client
        
        # Initialize Redis connection
        await redis_client.initialize()
        
        # Test basic operations
        await redis_client.set("_test_key", "test_value", expiry=10)
        value = await redis_client.get("_test_key")
        assert value == "test_value"
        
        await redis_client.delete("_test_key")
        
        logger.info("✅ Redis operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis test failed: {e}")
        return False


async def test_weaviate():
    """Test Weaviate vector database connection."""
    try:
        logger.info("🔍 Testing Weaviate connection...")
        
        from api.services.vector_store import vector_store
        
        # Test health check
        health = await vector_store.health_check()
        assert health is True
        
        # Test basic vector operations
        test_doc = {
            "content": "This is a test document for storage integration",
            "metadata": {"test": True, "timestamp": datetime.utcnow().isoformat()}
        }
        
        # Store a test document
        doc_id = await vector_store.store_document(
            content=test_doc["content"],
            metadata=test_doc["metadata"],
            source_id="test_storage"
        )
        
        # Search for the document
        results = await vector_store.search(
            query="test document storage",
            limit=1
        )
        
        assert len(results) > 0
        logger.info(f"✅ Weaviate operations test passed (found {len(results)} results)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Weaviate test failed: {e}")
        return False


async def test_neo4j():
    """Test Neo4j knowledge graph connection."""
    try:
        logger.info("🕸️ Testing Neo4j connection...")
        
        from api.services.knowledge_graph import KnowledgeGraphService
        
        graph_service = KnowledgeGraphService()
        await graph_service.initialize()
        
        # Test basic graph operations
        test_entity = {
            "id": "test_entity_storage",
            "type": "TestEntity",
            "properties": {
                "name": "Storage Test Entity",
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        # Create test entity
        await graph_service.create_entity(
            entity_id=test_entity["id"],
            entity_type=test_entity["type"],
            properties=test_entity["properties"]
        )
        
        # Query for the entity
        results = await graph_service.search_entities(
            query="Storage Test Entity",
            limit=1
        )
        
        assert len(results) > 0
        
        # Clean up
        await graph_service.delete_entity(test_entity["id"])
        await graph_service.cleanup()
        
        logger.info("✅ Neo4j operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Neo4j test failed: {e}")
        return False


async def test_timescaledb():
    """Test TimescaleDB time-series analytics."""
    try:
        logger.info("⏰ Testing TimescaleDB connection...")
        
        from api.services.time_series_analytics import TimeSeriesAnalyticsService, MetricType
        
        analytics = TimeSeriesAnalyticsService()
        await analytics.initialize()
        
        # Test metric recording
        success = await analytics.record_metric(
            metric_type=MetricType.PERFORMANCE,
            value=0.123,
            tags={"test": "storage_integration"},
            metadata={"component": "storage_test"}
        )
        
        assert success is True
        
        # Test knowledge evolution recording
        success = await analytics.record_knowledge_evolution(
            entity_type="storage_test",
            entity_id="test_entity",
            change_type="creation",
            new_value={"status": "testing"}
        )
        
        assert success is True
        
        await analytics.cleanup()
        
        logger.info("✅ TimescaleDB operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ TimescaleDB test failed: {e}")
        return False


async def test_minio():
    """Test MinIO object storage."""
    try:
        logger.info("📦 Testing MinIO connection...")
        
        from api.services.object_storage import ObjectStorageService
        
        storage = ObjectStorageService()
        await storage.initialize()
        
        # Test file upload and download
        test_content = b"This is test content for storage integration testing"
        file_key = "test/storage_integration_test.txt"
        
        # Upload test file
        success = await storage.upload_file(
            file_key=file_key,
            file_content=test_content,
            content_type="text/plain",
            metadata={"test": "storage_integration"}
        )
        
        assert success is True
        
        # Download test file
        downloaded_content = await storage.download_file(file_key)
        assert downloaded_content == test_content
        
        # List files
        files = await storage.list_files(prefix="test/")
        assert any(f["key"] == file_key for f in files)
        
        # Delete test file
        success = await storage.delete_file(file_key)
        assert success is True
        
        await storage.cleanup()
        
        logger.info("✅ MinIO operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ MinIO test failed: {e}")
        return False


async def test_integration_workflow():
    """Test a complete workflow using all storage systems."""
    try:
        logger.info("🔄 Testing integrated workflow...")
        
        # 1. Store a document in PostgreSQL and Weaviate
        from api.services.vector_store import vector_store
        from api.models import get_db
        from sqlalchemy import text
        
        db = next(get_db())
        
        # Create document record in PostgreSQL
        doc_content = "Integrated storage test workflow document"
        doc_metadata = {"workflow": "integration_test", "timestamp": datetime.utcnow().isoformat()}
        
        result = db.execute(text("""
            INSERT INTO knowledge_sources (name, url, type, status)
            VALUES ('Integration Test', 'http://test.local', 'document', 'active')
            RETURNING id;
        """))
        source_id = result.fetchone()[0]
        db.commit()
        
        # Store document vector in Weaviate
        doc_id = await vector_store.store_document(
            content=doc_content,
            metadata=doc_metadata,
            source_id=f"source_{source_id}"
        )
        
        # 2. Record metrics in TimescaleDB
        from api.services.time_series_analytics import TimeSeriesAnalyticsService, MetricType
        
        analytics = TimeSeriesAnalyticsService()
        await analytics.initialize()
        
        await analytics.record_metric(
            metric_type=MetricType.KNOWLEDGE_CREATION,
            value=1.0,
            tags={"workflow": "integration"},
            metadata={"document_id": doc_id, "source_id": source_id}
        )
        
        # 3. Create knowledge graph entities in Neo4j
        from api.services.knowledge_graph import KnowledgeGraphService
        
        graph = KnowledgeGraphService()
        await graph.initialize()
        
        await graph.create_entity(
            entity_id=f"document_{doc_id}",
            entity_type="Document",
            properties={
                "title": "Integration Test Document",
                "source_id": source_id,
                "content_preview": doc_content[:50]
            }
        )
        
        # 4. Cache results in Redis
        from api.services.cache import redis_client
        
        cache_key = f"integration_test_{doc_id}"
        cache_data = {
            "doc_id": doc_id,
            "source_id": source_id,
            "created_at": datetime.utcnow().isoformat()
        }
        await redis_client.set(cache_key, cache_data, expiry=300)
        
        # 5. Store metadata file in MinIO
        from api.services.object_storage import ObjectStorageService
        
        storage = ObjectStorageService()
        await storage.initialize()
        
        metadata_content = json.dumps(doc_metadata, indent=2).encode()
        await storage.upload_file(
            file_key=f"integration_test/{doc_id}/metadata.json",
            file_content=metadata_content,
            content_type="application/json"
        )
        
        # 6. Verify all data is accessible
        # Check PostgreSQL
        result = db.execute(text("SELECT name FROM knowledge_sources WHERE id = :id"), {"id": source_id})
        assert result.fetchone()[0] == "Integration Test"
        
        # Check Weaviate
        search_results = await vector_store.search("integrated storage test", limit=1)
        assert len(search_results) > 0
        
        # Check Redis
        cached_data = await redis_client.get(cache_key)
        assert cached_data is not None
        
        # Check Neo4j
        entities = await graph.search_entities("Integration Test Document", limit=1)
        assert len(entities) > 0
        
        # Check MinIO
        downloaded_metadata = await storage.download_file(f"integration_test/{doc_id}/metadata.json")
        assert json.loads(downloaded_metadata.decode()) == doc_metadata
        
        # Clean up
        db.execute(text("DELETE FROM knowledge_sources WHERE id = :id"), {"id": source_id})
        db.commit()
        await redis_client.delete(cache_key)
        await graph.delete_entity(f"document_{doc_id}")
        await storage.delete_file(f"integration_test/{doc_id}/metadata.json")
        
        await analytics.cleanup()
        await graph.cleanup()
        await storage.cleanup()
        
        logger.info("✅ Integrated workflow test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all storage integration tests."""
    logger.info("🚀 Starting comprehensive storage integration tests...")
    
    test_results = {}
    
    # Individual storage system tests
    tests = [
        ("PostgreSQL", test_postgresql),
        ("Redis", test_redis),
        ("Weaviate", test_weaviate),
        ("Neo4j", test_neo4j),
        ("TimescaleDB", test_timescaledb),
        ("MinIO", test_minio),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {test_name}...")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            test_results[test_name] = False
    
    # Integrated workflow test
    logger.info(f"\n{'='*50}")
    logger.info("Testing Integrated Workflow...")
    logger.info(f"{'='*50}")
    
    try:
        result = await test_integration_workflow()
        test_results["Integrated Workflow"] = result
    except Exception as e:
        logger.error(f"❌ Integrated workflow test failed with exception: {e}")
        test_results["Integrated Workflow"] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 STORAGE INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All storage integrations are working correctly!")
        return True
    else:
        logger.error(f"❌ {total - passed} storage integration(s) failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)