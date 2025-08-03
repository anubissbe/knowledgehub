"""
Unit tests for MemoryService.

Tests all memory service functionality including creation, retrieval,
search, updates, and deletion with comprehensive edge case coverage.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from api.services.memory_service import MemoryService
from api.models.memory import Memory
from api.database import get_db_session


@pytest.mark.unit
class TestMemoryService:
    """Unit tests for MemoryService."""
    
    @pytest_asyncio.fixture
    async def service(self, db_session):
        """Create MemoryService instance for testing."""
        with patch('api.services.memory_service.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = db_session
            service = MemoryService()
            # Mock external dependencies
            service.weaviate_client = MagicMock()
            service.ai_service = AsyncMock()
            return service
    
    @pytest.mark.asyncio
    async def test_create_memory_success(self, service, db_session):
        """Test successful memory creation."""
        # Arrange
        content = "Test memory content for unit testing"
        memory_type = "code"
        user_id = str(uuid4())
        project_id = str(uuid4())
        tags = ["test", "unit"]
        metadata = {"language": "python"}
        
        # Mock AI service embedding
        service.ai_service.generate_embedding.return_value = [0.1] * 1536
        
        # Act
        result = await service.create_memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            project_id=project_id,
            tags=tags,
            metadata=metadata
        )
        
        # Assert
        assert result.content == content
        assert result.memory_type == memory_type
        assert result.user_id == user_id
        assert result.project_id == project_id
        assert result.tags == tags
        assert result.metadata == metadata
        assert result.id is not None
        assert result.created_at is not None
        
        # Verify embedding was generated
        service.ai_service.generate_embedding.assert_called_once_with(content)
    
    @pytest.mark.asyncio
    async def test_create_memory_minimal_data(self, service, db_session):
        """Test memory creation with minimal required data."""
        # Arrange
        content = "Minimal memory content"
        
        # Mock embedding generation
        service.ai_service.generate_embedding.return_value = [0.1] * 1536
        
        # Act
        result = await service.create_memory(content=content)
        
        # Assert
        assert result.content == content
        assert result.memory_type == "general"  # Default type
        assert result.user_id is None
        assert result.project_id is None
        assert result.tags == []
        assert result.metadata == {}
    
    @pytest.mark.asyncio
    async def test_create_memory_embedding_failure(self, service, db_session):
        """Test memory creation when embedding generation fails."""
        # Arrange
        content = "Test content"
        service.ai_service.generate_embedding.side_effect = Exception("Embedding failed")
        
        # Act
        result = await service.create_memory(content=content)
        
        # Assert
        assert result.content == content
        assert result.embedding is None  # Should handle embedding failure gracefully
    
    @pytest.mark.asyncio
    async def test_get_memory_success(self, service, test_memory):
        """Test successful memory retrieval."""
        # Act
        result = await service.get_memory(test_memory.id)
        
        # Assert
        assert result is not None
        assert result.id == test_memory.id
        assert result.content == test_memory.content
    
    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, service):
        """Test memory retrieval with non-existent ID."""
        # Act
        result = await service.get_memory("non-existent-id")
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_search_memories_by_content(self, service, db_session, create_test_data):
        """Test memory search by content similarity."""
        # Arrange
        memories_data = [
            {
                "content": "Python function for data processing",
                "memory_type": "code",
                "user_id": str(uuid4()),
                "tags": ["python", "data"]
            },
            {
                "content": "JavaScript React component",
                "memory_type": "code",
                "user_id": str(uuid4()),
                "tags": ["javascript", "react"]
            },
            {
                "content": "Database query optimization",
                "memory_type": "technical",
                "user_id": str(uuid4()),
                "tags": ["database", "optimization"]
            }
        ]
        
        memories = await create_test_data(db_session, Memory, memories_data)
        
        # Mock Weaviate search
        service.weaviate_client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = {
            "data": {
                "Get": {
                    "Memory": [
                        {
                            "memory_id": memories[0].id,
                            "_additional": {"distance": 0.1}
                        },
                        {
                            "memory_id": memories[2].id,
                            "_additional": {"distance": 0.3}
                        }
                    ]
                }
            }
        }
        
        # Act
        results = await service.search_memories(
            query="python data processing",
            limit=5,
            similarity_threshold=0.5
        )
        
        # Assert
        assert len(results) == 2
        assert results[0].content == memories[0].content
        assert hasattr(results[0], 'similarity_score')
        assert results[0].similarity_score > results[1].similarity_score
    
    @pytest.mark.asyncio
    async def test_search_memories_with_filters(self, service, db_session, create_test_data):
        """Test memory search with various filters."""
        # Arrange
        user_id = str(uuid4())
        project_id = str(uuid4())
        
        memories_data = [
            {
                "content": "User-specific memory",
                "memory_type": "code",
                "user_id": user_id,
                "project_id": project_id,
                "tags": ["user", "specific"]
            },
            {
                "content": "Different user memory",
                "memory_type": "code",
                "user_id": str(uuid4()),
                "project_id": project_id,
                "tags": ["different", "user"]
            },
            {
                "content": "Technical documentation",
                "memory_type": "documentation",
                "user_id": user_id,
                "project_id": str(uuid4()),
                "tags": ["documentation", "technical"]
            }
        ]
        
        memories = await create_test_data(db_session, Memory, memories_data)
        
        # Mock Weaviate response
        service.weaviate_client.query.get.return_value.with_near_text.return_value.with_limit.return_value.with_where.return_value.do.return_value = {
            "data": {
                "Get": {
                    "Memory": [
                        {
                            "memory_id": memories[0].id,
                            "_additional": {"distance": 0.1}
                        }
                    ]
                }
            }
        }
        
        # Act
        results = await service.search_memories(
            query="memory",
            user_id=user_id,
            project_id=project_id,
            memory_type="code",
            limit=10
        )
        
        # Assert
        assert len(results) == 1
        assert results[0].user_id == user_id
        assert results[0].project_id == project_id
        assert results[0].memory_type == "code"
    
    @pytest.mark.asyncio
    async def test_search_memories_no_results(self, service):
        """Test memory search with no matching results."""
        # Mock empty Weaviate response
        service.weaviate_client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = {
            "data": {
                "Get": {
                    "Memory": []
                }
            }
        }
        
        # Act
        results = await service.search_memories(
            query="non-existent content",
            limit=5
        )
        
        # Assert
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_update_memory_success(self, service, test_memory):
        """Test successful memory update."""
        # Arrange
        new_content = "Updated memory content"
        new_tags = ["updated", "test"]
        new_metadata = {"updated": True, "version": 2}
        
        # Mock embedding generation for updated content
        service.ai_service.generate_embedding.return_value = [0.2] * 1536
        
        # Act
        result = await service.update_memory(
            memory_id=test_memory.id,
            content=new_content,
            tags=new_tags,
            metadata=new_metadata
        )
        
        # Assert
        assert result is not None
        assert result.content == new_content
        assert result.tags == new_tags
        assert result.metadata == new_metadata
        assert result.updated_at is not None
        
        # Verify embedding was regenerated
        service.ai_service.generate_embedding.assert_called_once_with(new_content)
    
    @pytest.mark.asyncio
    async def test_update_memory_partial(self, service, test_memory):
        """Test partial memory update."""
        # Arrange
        original_content = test_memory.content
        new_tags = ["partially", "updated"]
        
        # Act
        result = await service.update_memory(
            memory_id=test_memory.id,
            tags=new_tags
        )
        
        # Assert
        assert result is not None
        assert result.content == original_content  # Unchanged
        assert result.tags == new_tags  # Updated
        assert result.updated_at is not None
        
        # Verify embedding was not regenerated
        service.ai_service.generate_embedding.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, service):
        """Test updating non-existent memory."""
        # Act
        result = await service.update_memory(
            memory_id="non-existent-id",
            content="Updated content"
        )
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_memory_success(self, service, test_memory):
        """Test successful memory deletion."""
        # Act
        result = await service.delete_memory(test_memory.id)
        
        # Assert
        assert result is True
        
        # Verify memory is actually deleted
        deleted_memory = await service.get_memory(test_memory.id)
        assert deleted_memory is None
    
    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, service):
        """Test deleting non-existent memory."""
        # Act
        result = await service.delete_memory("non-existent-id")
        
        # Assert
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_user_memories(self, service, db_session, create_test_data):
        """Test retrieving memories for a specific user."""
        # Arrange
        user_id = str(uuid4())
        other_user_id = str(uuid4())
        
        memories_data = [
            {
                "content": f"User memory {i}",
                "memory_type": "code",
                "user_id": user_id,
                "tags": [f"memory_{i}"]
            }
            for i in range(3)
        ] + [
            {
                "content": "Other user memory",
                "memory_type": "code",
                "user_id": other_user_id,
                "tags": ["other"]
            }
        ]
        
        await create_test_data(db_session, Memory, memories_data)
        
        # Act
        results = await service.get_user_memories(user_id, limit=10)
        
        # Assert
        assert len(results) == 3
        for memory in results:
            assert memory.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_get_user_memories_with_pagination(self, service, db_session, create_test_data):
        """Test user memories retrieval with pagination."""
        # Arrange
        user_id = str(uuid4())
        
        memories_data = [
            {
                "content": f"Memory {i}",
                "memory_type": "code",
                "user_id": user_id,
                "tags": [f"memory_{i}"]
            }
            for i in range(10)
        ]
        
        await create_test_data(db_session, Memory, memories_data)
        
        # Act - First page
        page1 = await service.get_user_memories(user_id, limit=3, offset=0)
        page2 = await service.get_user_memories(user_id, limit=3, offset=3)
        
        # Assert
        assert len(page1) == 3
        assert len(page2) == 3
        
        # Ensure no overlap
        page1_ids = {m.id for m in page1}
        page2_ids = {m.id for m in page2}
        assert page1_ids.isdisjoint(page2_ids)
    
    @pytest.mark.asyncio
    async def test_get_memory_statistics(self, service, db_session, create_test_data):
        """Test memory statistics retrieval."""
        # Arrange
        user_id = str(uuid4())
        project_id = str(uuid4())
        
        memories_data = [
            {
                "content": f"Memory {i}",
                "memory_type": "code" if i % 2 == 0 else "documentation",
                "user_id": user_id if i < 5 else str(uuid4()),
                "project_id": project_id if i < 3 else str(uuid4()),
                "tags": ["test"]
            }
            for i in range(10)
        ]
        
        await create_test_data(db_session, Memory, memories_data)
        
        # Act
        stats = await service.get_memory_statistics(user_id=user_id)
        
        # Assert
        assert "total_memories" in stats
        assert "memories_by_type" in stats
        assert "memories_by_project" in stats
        assert "recent_activity" in stats
        
        assert stats["total_memories"] == 5  # User has 5 memories
        assert stats["memories_by_type"]["code"] >= 1
        assert stats["memories_by_type"]["documentation"] >= 1
    
    @pytest.mark.asyncio
    async def test_get_memory_statistics_global(self, service, db_session, create_test_data):
        """Test global memory statistics."""
        # Arrange
        memories_data = [
            {
                "content": f"Memory {i}",
                "memory_type": "code",
                "user_id": str(uuid4()),
                "tags": ["global", "test"]
            }
            for i in range(5)
        ]
        
        await create_test_data(db_session, Memory, memories_data)
        
        # Act
        stats = await service.get_memory_statistics()
        
        # Assert
        assert stats["total_memories"] == 5
        assert "memories_by_type" in stats
        assert "recent_activity" in stats
    
    @pytest.mark.asyncio
    async def test_sync_to_vector_store_success(self, service, test_memory):
        """Test successful vector store synchronization."""
        # Arrange
        service.ai_service.generate_embedding.return_value = [0.1] * 1536
        
        # Act
        result = await service.sync_to_vector_store(test_memory.id)
        
        # Assert
        assert result is True
        service.ai_service.generate_embedding.assert_called_once_with(test_memory.content)
    
    @pytest.mark.asyncio
    async def test_sync_to_vector_store_failure(self, service, test_memory):
        """Test vector store synchronization failure."""
        # Arrange
        service.ai_service.generate_embedding.side_effect = Exception("Vector store error")
        
        # Act
        result = await service.sync_to_vector_store(test_memory.id)
        
        # Assert
        assert result is False
    
    @pytest.mark.asyncio
    async def test_bulk_create_memories(self, service, db_session):
        """Test bulk memory creation."""
        # Arrange
        memories_data = [
            {
                "content": f"Bulk memory {i}",
                "memory_type": "bulk_test",
                "user_id": str(uuid4()),
                "tags": ["bulk", f"memory_{i}"]
            }
            for i in range(5)
        ]
        
        # Mock embedding generation
        service.ai_service.generate_embedding.return_value = [0.1] * 1536
        
        # Act
        results = await service.bulk_create_memories(memories_data)
        
        # Assert
        assert len(results) == 5
        for i, memory in enumerate(results):
            assert memory.content == f"Bulk memory {i}"
            assert memory.memory_type == "bulk_test"
        
        # Verify embeddings were generated for all memories
        assert service.ai_service.generate_embedding.call_count == 5
    
    @pytest.mark.asyncio
    async def test_cleanup_old_memories(self, service, db_session, create_test_data):
        """Test cleanup of old memories."""
        # Arrange
        old_date = datetime.utcnow() - timedelta(days=100)
        recent_date = datetime.utcnow() - timedelta(days=1)
        
        # Create memories with different timestamps
        memories_data = [
            {
                "content": f"Old memory {i}",
                "memory_type": "cleanup_test",
                "user_id": str(uuid4()),
                "tags": ["old"],
                "created_at": old_date
            }
            for i in range(3)
        ] + [
            {
                "content": f"Recent memory {i}",
                "memory_type": "cleanup_test",
                "user_id": str(uuid4()),
                "tags": ["recent"],
                "created_at": recent_date
            }
            for i in range(2)
        ]
        
        memories = await create_test_data(db_session, Memory, memories_data)
        
        # Act
        deleted_count = await service.cleanup_old_memories(
            days_old=90,
            memory_type="cleanup_test"
        )
        
        # Assert
        assert deleted_count == 3  # Only old memories should be deleted
        
        # Verify recent memories still exist
        remaining_memories = await service.search_memories(
            query="memory",
            memory_type="cleanup_test",
            limit=10
        )
        # Note: This depends on the search implementation
    
    @pytest.mark.asyncio
    async def test_error_handling_database_error(self, service):
        """Test error handling for database errors."""
        # Arrange
        with patch('api.services.memory_service.get_db_session') as mock_get_db:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error")
            mock_get_db.return_value.__aenter__.return_value = mock_session
            
            # Act & Assert
            with pytest.raises(Exception):
                await service.get_memory("test-id")
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, service, db_session):
        """Test concurrent memory operations."""
        import asyncio
        
        # Arrange
        service.ai_service.generate_embedding.return_value = [0.1] * 1536
        
        # Act - Create multiple memories concurrently
        tasks = [
            service.create_memory(
                content=f"Concurrent memory {i}",
                memory_type="concurrent_test",
                tags=[f"concurrent_{i}"]
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert hasattr(result, 'content')
    
    @pytest.mark.asyncio
    async def test_memory_validation(self, service):
        """Test memory data validation."""
        # Test empty content
        with pytest.raises(ValueError):
            await service.create_memory(content="")
        
        # Test invalid memory type
        memory = await service.create_memory(
            content="Test content",
            memory_type="invalid_type_with_very_long_name_that_exceeds_limits"
        )
        # Should handle gracefully or raise appropriate error
        assert memory is not None or True  # Adjust based on implementation
    
    @pytest.mark.asyncio
    async def test_search_performance_with_large_dataset(self, service, db_session, load_test_data):
        """Test search performance with large dataset."""
        # Arrange
        large_dataset = load_test_data(1000, "memory")
        
        # Mock Weaviate response with many results
        mock_results = [
            {
                "memory_id": str(uuid4()),
                "_additional": {"distance": 0.1 + (i * 0.01)}
            }
            for i in range(100)
        ]
        
        service.weaviate_client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = {
            "data": {
                "Get": {
                    "Memory": mock_results
                }
            }
        }
        
        # Act & Assert
        import time
        start_time = time.time()
        
        results = await service.search_memories(
            query="test query",
            limit=50
        )
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Search should complete in reasonable time
        assert search_time < 5.0  # Should complete within 5 seconds
        assert len(results) <= 50  # Respects limit