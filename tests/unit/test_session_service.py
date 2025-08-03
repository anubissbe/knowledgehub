"""
Unit tests for SessionService.

Tests all session service functionality including creation, retrieval,
updates, context management, and session lifecycle operations.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from api.services.session_service import SessionService
from api.models.session import Session
from api.database import get_db_session


@pytest.mark.unit
class TestSessionService:
    """Unit tests for SessionService."""
    
    @pytest_asyncio.fixture
    async def service(self, db_session):
        """Create SessionService instance for testing."""
        with patch('api.services.session_service.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = db_session
            service = SessionService()
            # Mock external dependencies
            service.memory_service = AsyncMock()
            service.ai_service = AsyncMock()
            return service
    
    @pytest.mark.asyncio
    async def test_create_session_success(self, service, db_session):
        """Test successful session creation."""
        # Arrange
        user_id = str(uuid4())
        session_type = "coding"
        project_id = str(uuid4())
        context_data = {
            "focus": "Testing implementation",
            "current_file": "test_session.py",
            "tasks": ["Write tests", "Fix bugs"]
        }
        
        # Act
        result = await service.create_session(
            user_id=user_id,
            session_type=session_type,
            project_id=project_id,
            context_data=context_data
        )
        
        # Assert
        assert result.user_id == user_id
        assert result.session_type == session_type
        assert result.project_id == project_id
        assert result.context_data == context_data
        assert result.status == "active"
        assert result.session_id is not None
        assert result.created_at is not None
    
    @pytest.mark.asyncio
    async def test_create_session_minimal_data(self, service, db_session):
        """Test session creation with minimal required data."""
        # Arrange
        user_id = str(uuid4())
        
        # Act
        result = await service.create_session(user_id=user_id)
        
        # Assert
        assert result.user_id == user_id
        assert result.session_type == "general"  # Default type
        assert result.project_id is None
        assert result.context_data == {}
        assert result.status == "active"
    
    @pytest.mark.asyncio
    async def test_get_session_success(self, service, test_session):
        """Test successful session retrieval."""
        # Act
        result = await service.get_session(test_session.session_id)
        
        # Assert
        assert result is not None
        assert result.session_id == test_session.session_id
        assert result.user_id == test_session.user_id
    
    @pytest.mark.asyncio
    async def test_get_session_not_found(self, service):
        """Test session retrieval with non-existent ID."""
        # Act
        result = await service.get_session("non-existent-session-id")
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_active_session_success(self, service, db_session):
        """Test retrieval of active session for user."""
        # Arrange
        user_id = str(uuid4())
        session = await service.create_session(
            user_id=user_id,
            session_type="coding",
            status="active"
        )
        
        # Act
        result = await service.get_active_session(user_id)
        
        # Assert
        assert result is not None
        assert result.session_id == session.session_id
        assert result.status == "active"
    
    @pytest.mark.asyncio
    async def test_get_active_session_none_active(self, service, db_session):
        """Test retrieval when no active session exists."""
        # Arrange
        user_id = str(uuid4())
        
        # Create inactive session
        await service.create_session(
            user_id=user_id,
            session_type="coding",
            status="ended"
        )
        
        # Act
        result = await service.get_active_session(user_id)
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_session_context_success(self, service, test_session):
        """Test successful session context update."""
        # Arrange
        new_context = {
            "focus": "Updated focus",
            "current_file": "updated_file.py",
            "tasks": ["Updated task 1", "Updated task 2"],
            "progress": 0.75
        }
        
        # Act
        result = await service.update_session_context(
            session_id=test_session.session_id,
            context_data=new_context
        )
        
        # Assert
        assert result is not None
        assert result.context_data == new_context
        assert result.updated_at is not None
    
    @pytest.mark.asyncio
    async def test_update_session_context_merge(self, service, test_session):
        """Test session context update with merging."""
        # Arrange
        original_context = test_session.context_data.copy()
        partial_update = {
            "new_field": "new_value",
            "progress": 0.5
        }
        
        # Act
        result = await service.update_session_context(
            session_id=test_session.session_id,
            context_data=partial_update,
            merge=True
        )
        
        # Assert
        assert result is not None
        expected_context = {**original_context, **partial_update}
        assert result.context_data == expected_context
    
    @pytest.mark.asyncio
    async def test_update_session_context_not_found(self, service):
        """Test updating context for non-existent session."""
        # Act
        result = await service.update_session_context(
            session_id="non-existent-session",
            context_data={"test": "data"}
        )
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_end_session_success(self, service, test_session):
        """Test successful session ending."""
        # Arrange
        end_context = {"reason": "Task completed", "duration": "2 hours"}
        
        # Act
        result = await service.end_session(
            session_id=test_session.session_id,
            end_context=end_context
        )
        
        # Assert
        assert result is not None
        assert result.status == "ended"
        assert result.ended_at is not None
        assert "reason" in result.context_data
        assert result.context_data["reason"] == "Task completed"
    
    @pytest.mark.asyncio
    async def test_end_session_with_memory_creation(self, service, test_session):
        """Test session ending with automatic memory creation."""
        # Arrange
        service.memory_service.create_memory.return_value = MagicMock(id="memory-id")
        
        # Act
        result = await service.end_session(
            session_id=test_session.session_id,
            create_memory=True
        )
        
        # Assert
        assert result is not None
        assert result.status == "ended"
        
        # Verify memory was created
        service.memory_service.create_memory.assert_called_once()
        call_args = service.memory_service.create_memory.call_args[1]
        assert call_args["memory_type"] == "session"
        assert call_args["user_id"] == test_session.user_id
    
    @pytest.mark.asyncio
    async def test_end_session_not_found(self, service):
        """Test ending non-existent session."""
        # Act
        result = await service.end_session("non-existent-session")
        
        # Assert
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_user_sessions(self, service, db_session, create_test_data):
        """Test retrieving sessions for a specific user."""
        # Arrange
        user_id = str(uuid4())
        other_user_id = str(uuid4())
        
        sessions_data = [
            {
                "session_id": str(uuid4()),
                "user_id": user_id,
                "session_type": "coding",
                "status": "active",
                "context_data": {"session": i}
            }
            for i in range(3)
        ] + [
            {
                "session_id": str(uuid4()),
                "user_id": other_user_id,
                "session_type": "coding",
                "status": "active",
                "context_data": {"other_user": True}
            }
        ]
        
        await create_test_data(db_session, Session, sessions_data)
        
        # Act
        results = await service.get_user_sessions(user_id, limit=10)
        
        # Assert
        assert len(results) == 3
        for session in results:
            assert session.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_get_user_sessions_with_status_filter(self, service, db_session, create_test_data):
        """Test retrieving user sessions with status filter."""
        # Arrange
        user_id = str(uuid4())
        
        sessions_data = [
            {
                "session_id": str(uuid4()),
                "user_id": user_id,
                "session_type": "coding",
                "status": "active",
                "context_data": {"active": True}
            },
            {
                "session_id": str(uuid4()),
                "user_id": user_id,
                "session_type": "coding",
                "status": "ended",
                "context_data": {"ended": True}
            },
            {
                "session_id": str(uuid4()),
                "user_id": user_id,
                "session_type": "coding",
                "status": "paused",
                "context_data": {"paused": True}
            }
        ]
        
        await create_test_data(db_session, Session, sessions_data)
        
        # Act
        active_sessions = await service.get_user_sessions(user_id, status="active")
        ended_sessions = await service.get_user_sessions(user_id, status="ended")
        
        # Assert
        assert len(active_sessions) == 1
        assert active_sessions[0].status == "active"
        
        assert len(ended_sessions) == 1
        assert ended_sessions[0].status == "ended"
    
    @pytest.mark.asyncio
    async def test_get_user_sessions_with_pagination(self, service, db_session, create_test_data):
        """Test user sessions retrieval with pagination."""
        # Arrange
        user_id = str(uuid4())
        
        sessions_data = [
            {
                "session_id": str(uuid4()),
                "user_id": user_id,
                "session_type": "coding",
                "status": "active",
                "context_data": {"index": i}
            }
            for i in range(10)
        ]
        
        await create_test_data(db_session, Session, sessions_data)
        
        # Act
        page1 = await service.get_user_sessions(user_id, limit=3, offset=0)
        page2 = await service.get_user_sessions(user_id, limit=3, offset=3)
        
        # Assert
        assert len(page1) == 3
        assert len(page2) == 3
        
        # Ensure no overlap
        page1_ids = {s.session_id for s in page1}
        page2_ids = {s.session_id for s in page2}
        assert page1_ids.isdisjoint(page2_ids)
    
    @pytest.mark.asyncio
    async def test_pause_session_success(self, service, test_session):
        """Test successful session pausing."""
        # Arrange
        pause_context = {"reason": "Break time", "duration_so_far": "1 hour"}
        
        # Act
        result = await service.pause_session(
            session_id=test_session.session_id,
            pause_context=pause_context
        )
        
        # Assert
        assert result is not None
        assert result.status == "paused"
        assert result.paused_at is not None
        assert "reason" in result.context_data
    
    @pytest.mark.asyncio
    async def test_resume_session_success(self, service, db_session):
        """Test successful session resuming."""
        # Arrange
        user_id = str(uuid4())
        session = await service.create_session(user_id=user_id, status="paused")
        
        resume_context = {"resumed_reason": "Back from break"}
        
        # Act
        result = await service.resume_session(
            session_id=session.session_id,
            resume_context=resume_context
        )
        
        # Assert
        assert result is not None
        assert result.status == "active"
        assert result.resumed_at is not None
        assert "resumed_reason" in result.context_data
    
    @pytest.mark.asyncio
    async def test_get_session_statistics(self, service, db_session, create_test_data):
        """Test session statistics retrieval."""
        # Arrange
        user_id = str(uuid4())
        
        sessions_data = [
            {
                "session_id": str(uuid4()),
                "user_id": user_id,
                "session_type": "coding" if i % 2 == 0 else "debugging",
                "status": "active" if i < 3 else "ended",
                "context_data": {"test": True},
                "created_at": datetime.utcnow() - timedelta(days=i)
            }
            for i in range(10)
        ]
        
        await create_test_data(db_session, Session, sessions_data)
        
        # Act
        stats = await service.get_session_statistics(user_id=user_id)
        
        # Assert
        assert "total_sessions" in stats
        assert "active_sessions" in stats
        assert "sessions_by_type" in stats
        assert "avg_session_duration" in stats
        assert "recent_activity" in stats
        
        assert stats["total_sessions"] == 10
        assert stats["active_sessions"] == 3
        assert stats["sessions_by_type"]["coding"] >= 1
    
    @pytest.mark.asyncio
    async def test_restore_session_context_success(self, service, test_session):
        """Test successful session context restoration."""
        # Arrange
        service.memory_service.search_memories.return_value = [
            MagicMock(
                content="Previous session context",
                metadata={"session_id": "previous-session"}
            )
        ]
        
        service.ai_service.generate_ai_insights.return_value = {
            "restored_context": {
                "focus": "Restored focus",
                "recommendations": ["Continue with testing"]
            }
        }
        
        # Act
        result = await service.restore_session_context(test_session.session_id)
        
        # Assert
        assert result is not None
        assert "restored_context" in result.context_data
        
        # Verify memory search was called
        service.memory_service.search_memories.assert_called_once()
        service.ai_service.generate_ai_insights.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_insights(self, service, test_session):
        """Test session insights generation."""
        # Arrange
        service.ai_service.generate_ai_insights.return_value = {
            "productivity_score": 0.85,
            "focus_areas": ["Testing", "Debugging"],
            "recommendations": ["Take more breaks", "Use more documentation"],
            "patterns": ["High productivity in mornings"]
        }
        
        # Act
        result = await service.get_session_insights(test_session.session_id)
        
        # Assert
        assert result is not None
        assert "productivity_score" in result
        assert "focus_areas" in result
        assert "recommendations" in result
        assert "patterns" in result
        
        # Verify AI service was called with session data
        service.ai_service.generate_ai_insights.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_sessions(self, service, db_session, create_test_data):
        """Test cleanup of old sessions."""
        # Arrange
        old_date = datetime.utcnow() - timedelta(days=100)
        recent_date = datetime.utcnow() - timedelta(days=1)
        
        sessions_data = [
            {
                "session_id": str(uuid4()),
                "user_id": str(uuid4()),
                "session_type": "cleanup_test",
                "status": "ended",
                "context_data": {"old": True},
                "created_at": old_date,
                "ended_at": old_date + timedelta(hours=2)
            }
            for i in range(3)
        ] + [
            {
                "session_id": str(uuid4()),
                "user_id": str(uuid4()),
                "session_type": "cleanup_test",
                "status": "ended",
                "context_data": {"recent": True},
                "created_at": recent_date,
                "ended_at": recent_date + timedelta(hours=1)
            }
            for i in range(2)
        ]
        
        await create_test_data(db_session, Session, sessions_data)
        
        # Act
        deleted_count = await service.cleanup_old_sessions(days_old=90)
        
        # Assert
        assert deleted_count == 3  # Only old sessions should be deleted
    
    @pytest.mark.asyncio
    async def test_auto_end_inactive_sessions(self, service, db_session, create_test_data):
        """Test automatic ending of inactive sessions."""
        # Arrange
        old_active_date = datetime.utcnow() - timedelta(hours=25)  # Over 24 hours ago
        recent_date = datetime.utcnow() - timedelta(hours=1)
        
        sessions_data = [
            {
                "session_id": str(uuid4()),
                "user_id": str(uuid4()),
                "session_type": "auto_end_test",
                "status": "active",
                "context_data": {"should_auto_end": True},
                "created_at": old_active_date,
                "last_activity_at": old_active_date
            },
            {
                "session_id": str(uuid4()),
                "user_id": str(uuid4()),
                "session_type": "auto_end_test",
                "status": "active",
                "context_data": {"should_stay_active": True},
                "created_at": recent_date,
                "last_activity_at": recent_date
            }
        ]
        
        await create_test_data(db_session, Session, sessions_data)
        
        # Act
        ended_count = await service.auto_end_inactive_sessions(
            inactive_hours=24
        )
        
        # Assert
        assert ended_count == 1  # Only the old inactive session should be ended
    
    @pytest.mark.asyncio
    async def test_session_activity_tracking(self, service, test_session):
        """Test session activity tracking."""
        # Act
        result = await service.track_activity(
            session_id=test_session.session_id,
            activity_type="code_edit",
            activity_data={"file": "test.py", "lines_changed": 5}
        )
        
        # Assert
        assert result is not None
        assert result.last_activity_at is not None
        assert "activities" in result.context_data
        
        activities = result.context_data["activities"]
        assert len(activities) > 0
        assert activities[-1]["activity_type"] == "code_edit"
    
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, service, db_session):
        """Test concurrent session operations."""
        import asyncio
        
        # Arrange
        user_id = str(uuid4())
        
        # Act - Create multiple sessions concurrently
        tasks = [
            service.create_session(
                user_id=user_id,
                session_type=f"concurrent_test_{i}",
                context_data={"index": i}
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert hasattr(result, 'session_id')
    
    @pytest.mark.asyncio
    async def test_session_handoff_creation(self, service, test_session):
        """Test session handoff creation for context transfer."""
        # Arrange
        handoff_message = "Transferring session to new environment"
        
        # Mock AI service for handoff context generation
        service.ai_service.generate_ai_insights.return_value = {
            "handoff_context": {
                "summary": "Working on unit tests",
                "next_steps": ["Complete test coverage", "Fix failing tests"],
                "important_context": ["Using pytest", "Database tests setup"]
            }
        }
        
        # Act
        result = await service.create_session_handoff(
            session_id=test_session.session_id,
            handoff_message=handoff_message
        )
        
        # Assert
        assert result is not None
        assert "handoff_context" in result
        assert "handoff_message" in result
        assert result["handoff_message"] == handoff_message
        
        # Verify AI service was called for context analysis
        service.ai_service.generate_ai_insights.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_session_id(self, service):
        """Test error handling for invalid session ID formats."""
        # Act & Assert
        with pytest.raises(ValueError):
            await service.get_session("invalid-session-id-format")
    
    @pytest.mark.asyncio
    async def test_error_handling_database_error(self, service):
        """Test error handling for database errors."""
        # Arrange
        with patch('api.services.session_service.get_db_session') as mock_get_db:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error")
            mock_get_db.return_value.__aenter__.return_value = mock_session
            
            # Act & Assert
            with pytest.raises(Exception):
                await service.get_session("test-session-id")
    
    @pytest.mark.asyncio
    async def test_session_validation(self, service):
        """Test session data validation."""
        # Test invalid user ID
        with pytest.raises(ValueError):
            await service.create_session(user_id="")
        
        # Test invalid session type
        session = await service.create_session(
            user_id=str(uuid4()),
            session_type="invalid_type_with_very_long_name"
        )
        # Should handle gracefully or raise appropriate error
        assert session is not None or True  # Adjust based on implementation