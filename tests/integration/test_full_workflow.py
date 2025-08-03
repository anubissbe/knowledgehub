"""
Integration tests for KnowledgeHub full workflows.

Tests complete end-to-end workflows including memory creation, session management,
AI enhancements, and cross-service integrations.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from uuid import uuid4
import asyncio

from api.services.memory_service import MemoryService
from api.services.session_service import SessionService
from api.services.ai_service import AIService
from api.services.pattern_service import PatternService
from api.services.copilot_service import CopilotEnhancementService
from api.services.ai_feedback_loop import AIFeedbackLoop, FeedbackType


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete KnowledgeHub workflows."""
    
    @pytest_asyncio.fixture
    async def services(self, db_session):
        """Create all service instances for integration testing."""
        return {
            "memory": MemoryService(),
            "session": SessionService(),
            "ai": AIService(),
            "pattern": PatternService(),
            "copilot": CopilotEnhancementService(),
            "feedback": AIFeedbackLoop()
        }
    
    @pytest.mark.asyncio
    async def test_complete_coding_session_workflow(self, services, integration_environment):
        """Test a complete coding session workflow from start to finish."""
        # Arrange
        users = integration_environment["users"]
        user = users[0]
        project_id = str(uuid4())
        
        memory_service = services["memory"]
        session_service = services["session"]
        
        # Act & Assert
        
        # 1. Start a new coding session
        session = await session_service.create_session(
            user_id=user.user_id,
            session_type="coding",
            project_id=project_id,
            context_data={
                "focus": "Building REST API",
                "current_file": "api/routes.py",
                "tasks": ["Create user endpoints", "Add authentication"]
            }
        )
        
        assert session is not None
        assert session.status == "active"
        assert session.context_data["focus"] == "Building REST API"
        
        # 2. Create memories during the session
        code_memory = await memory_service.create_memory(
            content="def create_user(user_data):\n    return User.objects.create(**user_data)",
            memory_type="code",
            user_id=user.user_id,
            project_id=project_id,
            tags=["user_management", "crud"],
            metadata={
                "file_path": "api/routes.py",
                "language": "python",
                "session_id": session.session_id
            }
        )
        
        decision_memory = await memory_service.create_memory(
            content="Decided to use JWT tokens for authentication",
            memory_type="decision",
            user_id=user.user_id,
            project_id=project_id,
            tags=["authentication", "security"],
            metadata={
                "alternatives": ["Session-based auth", "OAuth"],
                "reasoning": "Better for API scalability",
                "session_id": session.session_id
            }
        )
        
        assert code_memory.id is not None
        assert decision_memory.id is not None
        
        # 3. Update session context with progress
        updated_session = await session_service.update_session_context(
            session_id=session.session_id,
            context_data={
                "progress": 0.5,
                "completed_tasks": ["Create user endpoints"],
                "current_task": "Add authentication"
            },
            merge=True
        )
        
        assert updated_session.context_data["progress"] == 0.5
        assert "Create user endpoints" in updated_session.context_data["completed_tasks"]
        
        # 4. Search for relevant memories
        search_results = await memory_service.search_memories(
            query="user authentication",
            user_id=user.user_id,
            project_id=project_id,
            limit=5
        )
        
        # Should find both memories we created
        memory_contents = [m.content for m in search_results]
        assert any("JWT tokens" in content for content in memory_contents)
        
        # 5. Track activity in session
        activity_session = await session_service.track_activity(
            session_id=session.session_id,
            activity_type="code_edit",
            activity_data={
                "file": "api/routes.py",
                "lines_added": 15,
                "functions_created": ["create_user", "authenticate_user"]
            }
        )
        
        assert activity_session.last_activity_at is not None
        assert "activities" in activity_session.context_data
        
        # 6. Get session insights
        insights = await session_service.get_session_insights(session.session_id)
        
        assert insights is not None
        assert "productivity_score" in insights or "focus_areas" in insights
        
        # 7. End session with memory creation
        ended_session = await session_service.end_session(
            session_id=session.session_id,
            create_memory=True,
            end_context={
                "reason": "Task completed",
                "duration_hours": 2.5,
                "achievements": ["User API completed", "Authentication implemented"]
            }
        )
        
        assert ended_session.status == "ended"
        assert ended_session.ended_at is not None
        
        # 8. Verify session memory was created
        session_memories = await memory_service.search_memories(
            query="REST API session",
            user_id=user.user_id,
            memory_type="session",
            limit=5
        )
        
        assert len(session_memories) > 0
        session_memory = next(
            (m for m in session_memories if session.session_id in m.content),
            None
        )
        assert session_memory is not None
    
    @pytest.mark.asyncio
    async def test_ai_enhancement_workflow(self, services, integration_environment):
        """Test AI enhancement workflow with multiple services."""
        # Arrange
        user = integration_environment["users"][0]
        project_id = str(uuid4())
        
        memory_service = services["memory"]
        pattern_service = services["pattern"]
        copilot_service = services["copilot"]
        
        # Create baseline memories and patterns
        await memory_service.create_memory(
            content="Use dataclasses for DTOs in Python projects",
            memory_type="best_practice",
            user_id=user.user_id,
            project_id=project_id,
            tags=["python", "dto", "dataclass"]
        )
        
        await memory_service.create_memory(
            content="Always validate input data before processing",
            memory_type="guideline",
            user_id=user.user_id,
            project_id=project_id,
            tags=["validation", "security"]
        )
        
        # Act
        
        # 1. Simulate Copilot suggestion enhancement
        original_suggestion = "def process_user_data(data):\n    pass"
        context = {
            "file_path": "user_service.py",
            "language": "python",
            "code": "from dataclasses import dataclass\n@dataclass\nclass User:\n    name: str",
            "cursor_position": {"line": 5, "column": 0}
        }
        
        enhanced_suggestion = await copilot_service.enhance_suggestion(
            original_suggestion,
            context,
            user.user_id,
            project_id
        )
        
        # Assert enhancement worked
        assert enhanced_suggestion.original_suggestion == original_suggestion
        assert enhanced_suggestion.confidence > 0.5
        assert len(enhanced_suggestion.context_sources) > 0
        
        # 2. Analyze patterns in the enhanced code
        pattern_analysis = await pattern_service.analyze_code_patterns(
            user_id=user.user_id,
            project_id=project_id,
            force_refresh=True
        )
        
        assert "patterns" in pattern_analysis
        
        # 3. Test context injection
        copilot_request = {
            "prompt": "Create a function to validate user input",
            "context": {"language": "python", "file": "validators.py"}
        }
        
        enhanced_request = await copilot_service.inject_context(
            copilot_request,
            user.user_id,
            project_id
        )
        
        assert "knowledgehub_context" in enhanced_request
        assert enhanced_request["prompt"] != copilot_request["prompt"]  # Should be enhanced
        
        # 4. Create feedback loop
        feedback_result = await copilot_service.create_feedback_loop(
            enhanced_suggestion.id,
            "accepted",
            {
                "user_satisfaction": "high",
                "used_as_is": True,
                "helpful_context": enhanced_suggestion.context_sources
            }
        )
        
        assert feedback_result["status"] == "processed"
        assert feedback_result["learning_applied"] is True
    
    @pytest.mark.asyncio
    async def test_cross_service_memory_propagation(self, services, integration_environment):
        """Test memory propagation across different services."""
        # Arrange
        user = integration_environment["users"][0]
        project_id = str(uuid4())
        
        memory_service = services["memory"]
        session_service = services["session"]
        feedback_service = services["feedback"]
        
        # Act
        
        # 1. Create a session with important context
        session = await session_service.create_session(
            user_id=user.user_id,
            session_type="debugging",
            project_id=project_id,
            context_data={
                "focus": "Fix memory leak in user service",
                "error_patterns": ["OutOfMemoryError", "High CPU usage"],
                "investigation_notes": "Memory usage grows after each request"
            }
        )
        
        # 2. Create error memories
        error_memory = await memory_service.create_memory(
            content="Memory leak caused by unclosed database connections",
            memory_type="error",
            user_id=user.user_id,
            project_id=project_id,
            tags=["memory_leak", "database", "connection_pool"],
            metadata={
                "error_type": "OutOfMemoryError",
                "solution": "Use connection context managers",
                "session_id": session.session_id
            }
        )
        
        solution_memory = await memory_service.create_memory(
            content="with get_db_connection() as conn:\n    # Database operations",
            memory_type="solution",
            user_id=user.user_id,
            project_id=project_id,
            tags=["database", "context_manager", "best_practice"],
            metadata={
                "related_error": error_memory.id,
                "session_id": session.session_id
            }
        )
        
        # 3. Record feedback about the solution effectiveness
        await feedback_service.record_feedback(
            feedback_type=FeedbackType.POSITIVE,
            source="debugging_session",
            context={
                "session_id": session.session_id,
                "error_resolved": True,
                "solution_memory_id": solution_memory.id
            },
            details={
                "effectiveness": "high",
                "time_to_resolution": "30_minutes",
                "user_confidence": 0.9
            },
            user_id=user.user_id,
            project_id=project_id
        )
        
        # 4. End session and verify memory creation
        ended_session = await session_service.end_session(
            session_id=session.session_id,
            create_memory=True,
            end_context={
                "reason": "Issue resolved",
                "resolution": "Fixed memory leak using connection context managers"
            }
        )
        
        # Assert
        
        # 5. Search for related memories
        debugging_memories = await memory_service.search_memories(
            query="memory leak database connection",
            user_id=user.user_id,
            project_id=project_id,
            limit=10
        )
        
        # Should find error, solution, and session memories
        memory_types = {m.memory_type for m in debugging_memories}
        expected_types = {"error", "solution", "session"}
        assert expected_types.issubset(memory_types)
        
        # 6. Verify cross-references in metadata
        error_mem = next(m for m in debugging_memories if m.id == error_memory.id)
        solution_mem = next(m for m in debugging_memories if m.id == solution_memory.id)
        
        assert error_mem.metadata["session_id"] == session.session_id
        assert solution_mem.metadata["related_error"] == error_memory.id
        
        # 7. Analyze feedback patterns
        feedback_analysis = await feedback_service.analyze_feedback_patterns(
            time_window_hours=1,
            user_id=user.user_id,
            project_id=project_id
        )
        
        if feedback_analysis["status"] == "analysis_complete":
            assert "patterns" in feedback_analysis
            assert "insights" in feedback_analysis
    
    @pytest.mark.asyncio
    async def test_multi_user_collaboration_workflow(self, services, integration_environment):
        """Test multi-user collaboration workflow."""
        # Arrange
        users = integration_environment["users"]
        user1, user2 = users[0], users[1]
        shared_project_id = str(uuid4())
        
        memory_service = services["memory"]
        session_service = services["session"]
        
        # Act
        
        # 1. User 1 creates initial work
        session1 = await session_service.create_session(
            user_id=user1.user_id,
            session_type="feature_development",
            project_id=shared_project_id,
            context_data={
                "focus": "User authentication system",
                "role": "backend_developer"
            }
        )
        
        auth_memory = await memory_service.create_memory(
            content="Implemented JWT-based authentication with refresh tokens",
            memory_type="implementation",
            user_id=user1.user_id,
            project_id=shared_project_id,
            tags=["authentication", "jwt", "backend"],
            metadata={
                "api_endpoints": ["/login", "/refresh", "/logout"],
                "security_features": ["token_rotation", "blacklisting"]
            }
        )
        
        # 2. User 2 searches for User 1's work
        search_results = await memory_service.search_memories(
            query="authentication jwt",
            project_id=shared_project_id,  # No user_id = search across all users
            limit=5
        )
        
        # Should find User 1's authentication memory
        found_auth_memory = next(
            (m for m in search_results if m.id == auth_memory.id),
            None
        )
        assert found_auth_memory is not None
        
        # 3. User 2 starts complementary work
        session2 = await session_service.create_session(
            user_id=user2.user_id,
            session_type="frontend_development",
            project_id=shared_project_id,
            context_data={
                "focus": "Authentication UI",
                "role": "frontend_developer",
                "backend_apis": found_auth_memory.metadata["api_endpoints"]
            }
        )
        
        ui_memory = await memory_service.create_memory(
            content="Created login form with JWT token handling",
            memory_type="implementation",
            user_id=user2.user_id,
            project_id=shared_project_id,
            tags=["authentication", "frontend", "ui"],
            metadata={
                "components": ["LoginForm", "AuthProvider"],
                "related_backend": auth_memory.id
            }
        )
        
        # 4. Cross-reference memories
        await memory_service.update_memory(
            memory_id=auth_memory.id,
            metadata={
                **auth_memory.metadata,
                "related_frontend": ui_memory.id
            }
        )
        
        # Assert
        
        # 5. Verify cross-project collaboration
        project_memories = await memory_service.search_memories(
            query="authentication",
            project_id=shared_project_id,
            limit=10
        )
        
        # Should have memories from both users
        user_ids = {m.user_id for m in project_memories}
        assert user1.user_id in user_ids
        assert user2.user_id in user_ids
        
        # 6. Verify cross-references
        updated_auth_memory = await memory_service.get_memory(auth_memory.id)
        assert updated_auth_memory.metadata["related_frontend"] == ui_memory.id
        assert ui_memory.metadata["related_backend"] == auth_memory.id
        
        # 7. Get project statistics
        project_stats = await memory_service.get_memory_statistics(
            project_id=shared_project_id
        )
        
        assert project_stats["total_memories"] >= 2
        assert len(project_stats["contributors"]) == 2
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, services, integration_environment):
        """Test error recovery and resilience workflow."""
        # Arrange
        user = integration_environment["users"][0]
        project_id = str(uuid4())
        
        memory_service = services["memory"]
        session_service = services["session"]
        
        # Act & Assert
        
        # 1. Simulate session with error
        session = await session_service.create_session(
            user_id=user.user_id,
            session_type="debugging",
            project_id=project_id,
            context_data={"focus": "Fix critical bug"}
        )
        
        # 2. Create error memory
        error_memory = await memory_service.create_memory(
            content="NullPointerException in user validation",
            memory_type="error",
            user_id=user.user_id,
            project_id=project_id,
            tags=["error", "null_pointer", "validation"],
            metadata={
                "stack_trace": "line 42 in validate_user()",
                "severity": "critical",
                "occurrence_count": 5
            }
        )
        
        # 3. Simulate service interruption and recovery
        original_session_id = session.session_id
        
        # End session abruptly (simulating crash)
        await session_service.end_session(
            session_id=session.session_id,
            end_context={"reason": "Service interruption"}
        )
        
        # 4. Restore context in new session
        new_session = await session_service.create_session(
            user_id=user.user_id,
            session_type="debugging",
            project_id=project_id,
            context_data={
                "focus": "Continue debugging after service restart",
                "previous_session": original_session_id
            }
        )
        
        # Restore context from memories
        restored_session = await session_service.restore_session_context(
            new_session.session_id
        )
        
        # 5. Find solution and create resolution memory
        solution_memory = await memory_service.create_memory(
            content="Added null check before user validation",
            memory_type="solution",
            user_id=user.user_id,
            project_id=project_id,
            tags=["solution", "null_check", "validation"],
            metadata={
                "related_error": error_memory.id,
                "code_change": "if user is not None: validate_user(user)",
                "testing_status": "verified"
            }
        )
        
        # Assert
        
        # 6. Verify error-solution relationship
        error_solution_search = await memory_service.search_memories(
            query="null pointer validation solution",
            user_id=user.user_id,
            project_id=project_id,
            limit=5
        )
        
        memory_ids = {m.id for m in error_solution_search}
        assert error_memory.id in memory_ids
        assert solution_memory.id in memory_ids
        
        # 7. Verify session continuity
        assert restored_session is not None
        assert "restored_context" in restored_session.context_data or "handoff_context" in restored_session.context_data
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, services, integration_environment, load_test_data):
        """Test system performance under concurrent load."""
        # Arrange
        users = integration_environment["users"]
        memory_service = services["memory"]
        session_service = services["session"]
        
        num_operations = 50
        concurrent_users = len(users)
        
        # Act
        async def user_workflow(user, operation_count):
            """Simulate a user workflow with multiple operations."""
            project_id = str(uuid4())
            results = []
            
            # Create session
            session = await session_service.create_session(
                user_id=user.user_id,
                session_type="load_test",
                project_id=project_id
            )
            results.append(("session_created", session.session_id))
            
            # Create multiple memories
            for i in range(operation_count):
                memory = await memory_service.create_memory(
                    content=f"Load test memory {i} for user {user.username}",
                    memory_type="load_test",
                    user_id=user.user_id,
                    project_id=project_id,
                    tags=["load_test", f"batch_{i//10}"]
                )
                results.append(("memory_created", memory.id))
                
                # Perform search every 10 operations
                if i % 10 == 0:
                    search_results = await memory_service.search_memories(
                        query="load test",
                        user_id=user.user_id,
                        project_id=project_id,
                        limit=5
                    )
                    results.append(("search_completed", len(search_results)))
            
            # End session
            ended_session = await session_service.end_session(session.session_id)
            results.append(("session_ended", ended_session.status))
            
            return results
        
        # Execute concurrent workflows
        start_time = datetime.utcnow()
        
        tasks = [
            user_workflow(users[i % concurrent_users], num_operations // concurrent_users)
            for i in range(concurrent_users)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()
        
        # Assert
        
        # 1. Verify no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        
        # 2. Verify all operations completed
        total_operations = sum(len(result) for result in results)
        expected_operations = concurrent_users * (num_operations // concurrent_users * 2 + 2)  # memories + searches + sessions
        assert total_operations >= expected_operations * 0.9  # Allow 10% variance
        
        # 3. Verify reasonable performance
        operations_per_second = total_operations / total_duration
        assert operations_per_second > 10, f"Performance too slow: {operations_per_second} ops/sec"
        
        # 4. Verify data consistency
        total_memories_created = sum(
            len([op for op in result if op[0] == "memory_created"])
            for result in results
        )
        
        # Search for all load test memories
        all_memories = await memory_service.search_memories(
            query="load test",
            memory_type="load_test",
            limit=total_memories_created + 10
        )
        
        # Allow for some variance due to async operations
        assert len(all_memories) >= total_memories_created * 0.9
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_services(self, services, integration_environment):
        """Test data consistency across different services."""
        # Arrange
        user = integration_environment["users"][0]
        project_id = str(uuid4())
        
        memory_service = services["memory"]
        session_service = services["session"]
        pattern_service = services["pattern"]
        
        # Act
        
        # 1. Create interconnected data
        session = await session_service.create_session(
            user_id=user.user_id,
            session_type="consistency_test",
            project_id=project_id,
            context_data={"test": "data_consistency"}
        )
        
        memories = []
        for i in range(5):
            memory = await memory_service.create_memory(
                content=f"Consistency test memory {i}",
                memory_type="test",
                user_id=user.user_id,
                project_id=project_id,
                tags=["consistency", f"item_{i}"],
                metadata={"session_id": session.session_id, "index": i}
            )
            memories.append(memory)
        
        # 2. Update session with memory references
        memory_ids = [m.id for m in memories]
        await session_service.update_session_context(
            session_id=session.session_id,
            context_data={"created_memories": memory_ids},
            merge=True
        )
        
        # 3. Perform pattern analysis
        pattern_results = await pattern_service.analyze_code_patterns(
            user_id=user.user_id,
            project_id=project_id
        )
        
        # Assert
        
        # 4. Verify session-memory consistency
        updated_session = await session_service.get_session(session.session_id)
        assert "created_memories" in updated_session.context_data
        
        session_memory_ids = set(updated_session.context_data["created_memories"])
        actual_memory_ids = set(m.id for m in memories)
        assert session_memory_ids == actual_memory_ids
        
        # 5. Verify memory-session consistency
        for memory in memories:
            fresh_memory = await memory_service.get_memory(memory.id)
            assert fresh_memory.metadata["session_id"] == session.session_id
            assert fresh_memory.user_id == user.user_id
            assert fresh_memory.project_id == project_id
        
        # 6. Verify cross-service data integrity
        search_results = await memory_service.search_memories(
            query="consistency test",
            user_id=user.user_id,
            project_id=project_id,
            limit=10
        )
        
        found_memory_ids = {m.id for m in search_results}
        assert actual_memory_ids.issubset(found_memory_ids)
        
        # 7. Clean up and verify cascade operations
        deleted = await session_service.end_session(session.session_id)
        assert deleted.status == "ended"
        
        # Session should still exist but be marked as ended
        final_session = await session_service.get_session(session.session_id)
        assert final_session.status == "ended"
        
        # Memories should still exist (no cascade delete)
        for memory_id in memory_ids:
            memory_exists = await memory_service.get_memory(memory_id)
            assert memory_exists is not None