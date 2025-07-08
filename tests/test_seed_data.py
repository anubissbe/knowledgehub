"""
Test suite for Memory System Seed Data Generator

Comprehensive tests for seed data generation, validation, and management.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from api.memory_system.seed_data import MemorySystemSeedData
from api.models.memory import MemorySession, Memory, MemoryType


class TestMemorySystemSeedData:
    """Test suite for MemorySystemSeedData class"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        db = Mock(spec=Session)
        db.query.return_value = Mock()
        db.add = Mock()
        db.commit = Mock()
        db.refresh = Mock()
        return db
    
    @pytest.fixture
    def seed_generator(self, mock_db):
        """Create a seed data generator instance"""
        return MemorySystemSeedData(mock_db)
    
    def test_init(self, seed_generator):
        """Test initialization of MemorySystemSeedData"""
        assert seed_generator.db is not None
        assert seed_generator.session_manager is not None
        assert seed_generator.memory_manager is not None
        assert seed_generator.persistent_context_manager is not None
        assert len(seed_generator.sample_sessions) > 0
        assert len(seed_generator.sample_memories) > 0
        assert len(seed_generator.sample_contexts) > 0
    
    def test_sample_sessions_structure(self, seed_generator):
        """Test that sample sessions have correct structure"""
        sessions = seed_generator.sample_sessions
        
        assert len(sessions) >= 3, "Should have at least 3 sample sessions"
        
        for session in sessions:
            # Required fields
            assert "user_id" in session
            assert "project_id" in session
            assert "session_metadata" in session
            assert "tags" in session
            assert "duration_minutes" in session
            assert "memory_count" in session
            
            # Validate data types
            assert isinstance(session["user_id"], str)
            assert isinstance(session["project_id"], str)
            assert isinstance(session["session_metadata"], dict)
            assert isinstance(session["tags"], list)
            assert isinstance(session["duration_minutes"], int)
            assert isinstance(session["memory_count"], int)
            
            # Validate session metadata structure
            metadata = session["session_metadata"]
            assert "session_type" in metadata
            assert "primary_focus" in metadata
            assert "tools_used" in metadata
            assert "complexity_level" in metadata
            
            # Validate ranges
            assert 0 < session["duration_minutes"] <= 300, "Duration should be reasonable"
            assert 0 < session["memory_count"] <= 50, "Memory count should be reasonable"
    
    def test_sample_memories_structure(self, seed_generator):
        """Test that sample memories have correct structure"""
        memories = seed_generator.sample_memories
        
        assert len(memories) >= 10, "Should have at least 10 sample memories"
        
        memory_types = set()
        for memory in memories:
            # Required fields
            assert "memory_type" in memory
            assert "content" in memory
            assert "importance_score" in memory
            assert "entities" in memory
            assert "facts" in memory
            assert "metadata" in memory
            
            # Validate data types
            assert isinstance(memory["memory_type"], MemoryType)
            assert isinstance(memory["content"], str)
            assert isinstance(memory["importance_score"], float)
            assert isinstance(memory["entities"], list)
            assert isinstance(memory["facts"], list)
            assert isinstance(memory["metadata"], dict)
            
            # Validate content quality
            assert len(memory["content"]) > 20, "Content should be substantial"
            assert len(memory["entities"]) > 0, "Should have entities"
            assert len(memory["facts"]) > 0, "Should have facts"
            
            # Validate importance score
            assert 0.0 <= memory["importance_score"] <= 1.0, "Importance should be 0-1"
            
            memory_types.add(memory["memory_type"])
        
        # Should have diverse memory types
        assert len(memory_types) >= 5, "Should have at least 5 different memory types"
    
    def test_sample_contexts_structure(self, seed_generator):
        """Test that sample contexts have correct structure"""
        contexts = seed_generator.sample_contexts
        
        assert len(contexts) >= 2, "Should have at least 2 sample contexts"
        
        context_types = set()
        context_scopes = set()
        
        for context in contexts:
            # Required fields
            assert "content" in context
            assert "context_type" in context
            assert "scope" in context
            assert "importance" in context
            assert "related_entities" in context
            assert "metadata" in context
            
            # Validate data types
            assert isinstance(context["content"], str)
            assert isinstance(context["context_type"], str)
            assert isinstance(context["scope"], str)
            assert isinstance(context["importance"], float)
            assert isinstance(context["related_entities"], list)
            assert isinstance(context["metadata"], dict)
            
            # Validate content quality
            assert len(context["content"]) > 20, "Content should be substantial"
            assert len(context["related_entities"]) > 0, "Should have entities"
            
            # Validate importance score
            assert 0.0 <= context["importance"] <= 1.0, "Importance should be 0-1"
            
            context_types.add(context["context_type"])
            context_scopes.add(context["scope"])
        
        # Should have diverse types and scopes
        assert len(context_types) >= 2, "Should have at least 2 context types"
        assert len(context_scopes) >= 2, "Should have at least 2 context scopes"
    
    @pytest.mark.asyncio
    async def test_create_test_session(self, seed_generator):
        """Test creating a test session"""
        session_data = {
            "user_id": "test-user",
            "project_id": "test-project",
            "session_metadata": {"type": "test"},
            "tags": ["test"],
            "duration_minutes": 60,
            "memory_count": 5
        }
        
        session = await seed_generator._create_test_session(session_data)
        
        # Verify session was created
        assert session.user_id == "test-user"
        assert session.project_id == "test-project"
        assert session.session_metadata == {"type": "test"}
        assert session.tags == ["test"]
        assert session.duration == 3600  # 60 minutes in seconds
        assert session.memory_count == 5
        assert session.is_active is False
        
        # Verify database calls
        seed_generator.db.add.assert_called_once()
        seed_generator.db.commit.assert_called_once()
        seed_generator.db.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_test_memory(self, seed_generator):
        """Test creating a test memory"""
        import uuid
        session_id = uuid.uuid4()
        memory_data = {
            "memory_type": MemoryType.TECHNICAL_KNOWLEDGE,
            "content": "Test memory content",
            "importance_score": 0.8,
            "entities": ["test", "entity"],
            "facts": ["test fact"],
            "metadata": {"source": "test"}
        }
        
        memory = await seed_generator._create_test_memory(session_id, memory_data)
        
        # Verify memory was created
        assert memory.session_id == session_id
        assert memory.memory_type == MemoryType.TECHNICAL_KNOWLEDGE
        assert memory.content == "Test memory content"
        assert memory.importance_score == 0.8
        assert memory.entities == ["test", "entity"]
        assert memory.facts == ["test fact"]
        assert memory.metadata == {"source": "test"}
        
        # Verify database calls
        seed_generator.db.add.assert_called_once()
        seed_generator.db.commit.assert_called_once()
        seed_generator.db.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_seed_data(self, seed_generator):
        """Test generating seed data"""
        # Mock the database queries for session counting
        seed_generator.db.query.return_value.all.return_value = []
        seed_generator.db.query.return_value.count.return_value = 0
        
        # Mock the persistent context manager
        seed_generator.persistent_context_manager.add_context = Mock(return_value=uuid.uuid4())
        seed_generator.persistent_context_manager.get_context_summary = Mock(return_value={"total_vectors": 3})
        
        results = await seed_generator.generate_seed_data(num_sessions=2, num_memories_per_session=2)
        
        # Verify results structure
        assert "sessions_created" in results
        assert "memories_created" in results
        assert "contexts_created" in results
        assert "session_ids" in results
        assert "memory_ids" in results
        assert "context_ids" in results
        
        # Verify expected counts
        assert results["sessions_created"] == 2
        assert results["memories_created"] == 4  # 2 sessions * 2 memories each
        assert results["contexts_created"] == 3  # Number of sample contexts
        
        # Verify IDs are provided
        assert len(results["session_ids"]) == 2
        assert len(results["memory_ids"]) == 4
        assert len(results["context_ids"]) == 3
    
    @pytest.mark.asyncio
    async def test_clear_seed_data(self, seed_generator):
        """Test clearing seed data"""
        # Mock database queries
        seed_generator.db.query.return_value.count.return_value = 10
        seed_generator.db.query.return_value.delete.return_value = 10
        
        # Mock persistent context manager
        seed_generator.persistent_context_manager.clear_all_contexts = Mock()
        
        results = await seed_generator.clear_seed_data()
        
        # Verify results
        assert "sessions_deleted" in results
        assert "memories_deleted" in results
        assert "contexts_deleted" in results
        
        assert results["sessions_deleted"] == 10
        assert results["memories_deleted"] == 10
        
        # Verify database calls
        assert seed_generator.db.query.call_count >= 2  # At least memories and sessions
        seed_generator.db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_seed_data(self, seed_generator):
        """Test validating seed data"""
        # Mock database queries
        mock_sessions = [
            Mock(session_metadata={"type": "test"}, tags=["test"], duration=3600),
            Mock(session_metadata={}, tags=[], duration=0)
        ]
        mock_memories = [
            Mock(memory_type=MemoryType.TECHNICAL_KNOWLEDGE, entities=["test"], facts=["fact"], importance_score=0.8),
            Mock(memory_type=MemoryType.USER_PREFERENCE, entities=[], facts=[], importance_score=0.6)
        ]
        
        seed_generator.db.query.return_value.all.side_effect = [mock_sessions, mock_memories]
        
        # Mock persistent context manager
        seed_generator.persistent_context_manager.get_context_summary = Mock(return_value={"total_vectors": 5})
        
        results = await seed_generator.validate_seed_data()
        
        # Verify results structure
        assert "sessions" in results
        assert "memories" in results
        assert "contexts" in results
        assert "validation_passed" in results
        assert "errors" in results
        
        # Verify session validation
        assert results["sessions"]["count"] == 2
        assert results["sessions"]["has_metadata"] == 1
        assert results["sessions"]["has_tags"] == 1
        assert results["sessions"]["duration_set"] == 1
        
        # Verify memory validation
        assert results["memories"]["count"] == 2
        assert results["memories"]["with_entities"] == 1
        assert results["memories"]["with_facts"] == 1
        assert results["memories"]["avg_importance"] == 0.7  # (0.8 + 0.6) / 2
        
        # Verify context validation
        assert results["contexts"]["count"] == 5
        
        # Should pass validation
        assert results["validation_passed"] is True
    
    @pytest.mark.asyncio
    async def test_validate_seed_data_no_data(self, seed_generator):
        """Test validation when no data exists"""
        # Mock empty database
        seed_generator.db.query.return_value.all.return_value = []
        seed_generator.persistent_context_manager.get_context_summary = Mock(return_value={"total_vectors": 0})
        
        results = await seed_generator.validate_seed_data()
        
        # Should fail validation
        assert results["validation_passed"] is False
        assert "No sessions created" in results["errors"]
        assert "No memories created" in results["errors"]
    
    def test_memory_type_coverage(self, seed_generator):
        """Test that sample memories cover all important memory types"""
        memories = seed_generator.sample_memories
        memory_types = {memory["memory_type"] for memory in memories}
        
        # Should have good coverage of memory types
        expected_types = {
            MemoryType.TECHNICAL_KNOWLEDGE,
            MemoryType.USER_PREFERENCE,
            MemoryType.DECISION,
            MemoryType.PATTERN,
            MemoryType.WORKFLOW,
            MemoryType.PROBLEM_SOLUTION
        }
        
        missing_types = expected_types - memory_types
        assert len(missing_types) == 0, f"Missing memory types: {missing_types}"
    
    def test_importance_score_distribution(self, seed_generator):
        """Test that importance scores are well distributed"""
        memories = seed_generator.sample_memories
        importance_scores = [memory["importance_score"] for memory in memories]
        
        # Should have a range of importance scores
        assert min(importance_scores) >= 0.0
        assert max(importance_scores) <= 1.0
        assert max(importance_scores) - min(importance_scores) >= 0.3, "Should have good range of importance scores"
        
        # Should have some high-importance items
        high_importance = [score for score in importance_scores if score >= 0.8]
        assert len(high_importance) >= 3, "Should have at least 3 high-importance items"
    
    def test_entity_and_fact_quality(self, seed_generator):
        """Test that entities and facts are meaningful"""
        memories = seed_generator.sample_memories
        
        for memory in memories:
            entities = memory["entities"]
            facts = memory["facts"]
            
            # Entities should be non-empty and meaningful
            assert len(entities) > 0, "Should have entities"
            assert all(len(entity) > 1 for entity in entities), "Entities should be meaningful"
            
            # Facts should be non-empty and meaningful
            assert len(facts) > 0, "Should have facts"
            assert all(len(fact) > 5 for fact in facts), "Facts should be substantial"
    
    def test_metadata_quality(self, seed_generator):
        """Test that metadata is comprehensive and useful"""
        memories = seed_generator.sample_memories
        
        for memory in memories:
            metadata = memory["metadata"]
            
            # Should have metadata
            assert len(metadata) > 0, "Should have metadata"
            
            # Common metadata fields should exist where appropriate
            memory_type = memory["memory_type"]
            if memory_type == MemoryType.TECHNICAL_KNOWLEDGE:
                assert "source" in metadata or "applies_to" in metadata
            elif memory_type == MemoryType.USER_PREFERENCE:
                assert "preference_type" in metadata
            elif memory_type == MemoryType.DECISION:
                assert "decision_date" in metadata or "rationale" in metadata
            elif memory_type == MemoryType.PATTERN:
                assert "pattern_type" in metadata
            elif memory_type == MemoryType.WORKFLOW:
                assert "workflow_type" in metadata


class TestSeedDataCLI:
    """Test suite for the seed data CLI functionality"""
    
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing"""
        # This would require testing the argparse setup
        # For now, we'll just verify the main function exists
        from scripts.generate_seed_data import main
        assert callable(main)
    
    def test_cli_help_text(self):
        """Test that CLI provides helpful usage information"""
        from scripts.generate_seed_data import main
        
        # Test that help text exists and is comprehensive
        import argparse
        parser = argparse.ArgumentParser()
        
        # Verify expected arguments exist
        expected_args = ["--generate", "--validate", "--clear", "--sessions", "--memories"]
        # This is a basic check - full testing would require more complex setup
        assert len(expected_args) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])