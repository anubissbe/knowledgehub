"""
Unit tests for CopilotEnhancementService.

Tests all Copilot enhancement functionality including webhook processing,
suggestion enhancement, context injection, and feedback loop integration.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from api.services.copilot_service import CopilotEnhancementService, CopilotSuggestion


@pytest.mark.unit
class TestCopilotEnhancementService:
    """Unit tests for CopilotEnhancementService."""
    
    @pytest_asyncio.fixture
    async def service(self):
        """Create CopilotEnhancementService instance with mocked dependencies."""
        service = CopilotEnhancementService()
        service.memory_service = AsyncMock()
        service.session_service = AsyncMock()
        service.ai_service = AsyncMock()
        service.pattern_service = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_receive_webhook_suggestion_request(self, service):
        """Test receiving suggestion request webhook."""
        # Arrange
        webhook_type = "suggestion_request"
        payload = {
            "suggestion": "def process_data():\n    pass",
            "context": {
                "file_path": "data_processor.py",
                "language": "python",
                "code": "import pandas as pd\ndf = pd.read_csv('data.csv')"
            }
        }
        user_id = str(uuid4())
        
        # Mock enhancement
        service.enhance_suggestion = AsyncMock(return_value=MagicMock(
            id="suggestion-id",
            enhanced_suggestion="def process_data(df: pd.DataFrame) -> pd.DataFrame:\n    return df.dropna()",
            confidence=0.8,
            context_sources=["memory", "patterns"]
        ))
        
        # Act
        result = await service.receive_webhook(webhook_type, payload, user_id)
        
        # Assert
        assert result["status"] == "enhanced"
        assert "suggestion_id" in result
        assert "enhanced_suggestion" in result
        assert result["confidence"] == 0.8
        
        service.enhance_suggestion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_receive_webhook_suggestion_accepted(self, service):
        """Test receiving suggestion accepted webhook."""
        # Arrange
        webhook_type = "suggestion_accepted"
        suggestion_id = str(uuid4())
        payload = {
            "suggestion_id": suggestion_id,
            "accepted_suggestion": "enhanced code"
        }
        
        service.create_feedback_loop = AsyncMock(return_value={
            "status": "processed",
            "learning_applied": True
        })
        
        # Act
        result = await service.receive_webhook(webhook_type, payload)
        
        # Assert
        assert result["status"] == "processed"
        assert result["learning_applied"] is True
        
        service.create_feedback_loop.assert_called_once_with(
            suggestion_id, "accepted", payload
        )
    
    @pytest.mark.asyncio
    async def test_receive_webhook_unknown_type(self, service):
        """Test receiving unknown webhook type."""
        # Arrange
        webhook_type = "unknown_type"
        payload = {"data": "test"}
        
        # Act
        result = await service.receive_webhook(webhook_type, payload)
        
        # Assert
        assert result["status"] == "ignored"
        assert "Unknown type" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_enhance_suggestion_success(self, service):
        """Test successful suggestion enhancement."""
        # Arrange
        original_suggestion = "def calculate():\n    pass"
        context = {
            "code": "x = 10\ny = 20",
            "file_path": "calculator.py",
            "language": "python",
            "cursor_position": {"line": 3, "column": 0}
        }
        user_id = str(uuid4())
        project_id = str(uuid4())
        
        # Mock dependencies
        service._get_relevant_memories = AsyncMock(return_value=[
            {"content": "Math operations", "relevance": 0.8}
        ])
        service._analyze_context_patterns = AsyncMock(return_value=[
            {"type": "function_pattern", "confidence": 0.7}
        ])
        service._get_project_conventions = AsyncMock(return_value={
            "language": "python", "style": "pep8"
        })
        service._ai_enhance_suggestion = AsyncMock(return_value="def calculate(x: int, y: int) -> int:\n    return x + y")
        service._calculate_enhancement_confidence = AsyncMock(return_value=0.85)
        
        # Act
        result = await service.enhance_suggestion(original_suggestion, context, user_id, project_id)
        
        # Assert
        assert isinstance(result, CopilotSuggestion)
        assert result.original_suggestion == original_suggestion
        assert result.enhanced_suggestion == "def calculate(x: int, y: int) -> int:\n    return x + y"
        assert result.confidence == 0.85
        assert len(result.context_sources) > 0
        
        # Verify all enhancement steps were called
        service._get_relevant_memories.assert_called_once()
        service._analyze_context_patterns.assert_called_once()
        service._ai_enhance_suggestion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enhance_suggestion_fallback_on_error(self, service):
        """Test suggestion enhancement fallback when AI enhancement fails."""
        # Arrange
        original_suggestion = "def test(): pass"
        context = {"code": "test code"}
        
        # Mock dependencies to fail
        service._get_relevant_memories = AsyncMock(side_effect=Exception("Memory error"))
        service._analyze_context_patterns = AsyncMock(side_effect=Exception("Pattern error"))
        service._ai_enhance_suggestion = AsyncMock(side_effect=Exception("AI error"))
        
        # Act
        result = await service.enhance_suggestion(original_suggestion, context)
        
        # Assert
        assert isinstance(result, CopilotSuggestion)
        assert result.original_suggestion == original_suggestion
        assert result.enhanced_suggestion == original_suggestion  # Fallback to original
        assert result.confidence == 0.5  # Default confidence
        assert "fallback" in result.context_sources
    
    @pytest.mark.asyncio
    async def test_inject_context_success(self, service):
        """Test successful context injection."""
        # Arrange
        request = {
            "prompt": "Generate a function to process data",
            "context": {"language": "python"}
        }
        user_id = str(uuid4())
        project_id = str(uuid4())
        
        # Mock context gathering
        service._get_session_context = AsyncMock(return_value={
            "session_id": "session-123",
            "focus": "Data processing"
        })
        service._get_project_context = AsyncMock(return_value={
            "project_id": project_id,
            "type": "data_analysis"
        })
        service._get_relevant_decisions = AsyncMock(return_value=[
            {"decision": "Use pandas for data processing", "reasoning": "Performance"}
        ])
        
        # Act
        result = await service.inject_context(request, user_id, project_id)
        
        # Assert
        assert "knowledgehub_context" in result
        assert "session" in result["knowledgehub_context"]
        assert "project" in result["knowledgehub_context"]
        assert "decisions" in result["knowledgehub_context"]
        
        # Verify prompt was enhanced
        assert "KnowledgeHub Context" in result["prompt"]
        assert "Data processing" in result["prompt"]
    
    @pytest.mark.asyncio
    async def test_inject_context_minimal(self, service):
        """Test context injection with minimal context available."""
        # Arrange
        request = {"prompt": "Write a function"}
        
        # Mock empty context
        service._get_session_context = AsyncMock(return_value={})
        service._get_project_context = AsyncMock(return_value={})
        service._get_relevant_decisions = AsyncMock(return_value=[])
        
        # Act
        result = await service.inject_context(request)
        
        # Assert
        assert "knowledgehub_context" in result
        assert result["prompt"] == "Write a function"  # Unchanged since no context
    
    @pytest.mark.asyncio
    async def test_create_feedback_loop_success(self, service):
        """Test successful feedback loop creation."""
        # Arrange
        suggestion_id = str(uuid4())
        suggestion = CopilotSuggestion(
            original_suggestion="original",
            enhanced_suggestion="enhanced",
            confidence=0.8,
            context_sources=["memory", "patterns"]
        )
        service.active_suggestions[suggestion_id] = suggestion
        
        feedback_type = "accepted"
        feedback_data = {"user_satisfaction": "high"}
        
        # Mock feedback processing
        service._process_feedback = AsyncMock(return_value={"processed": True})
        service._update_models_from_feedback = AsyncMock()
        service._store_feedback_data = AsyncMock()
        
        # Act
        result = await service.create_feedback_loop(suggestion_id, feedback_type, feedback_data)
        
        # Assert
        assert result["status"] == "processed"
        assert result["suggestion_id"] == suggestion_id
        assert result["feedback_type"] == feedback_type
        assert result["learning_applied"] is True
        
        # Verify processing steps
        service._process_feedback.assert_called_once()
        service._update_models_from_feedback.assert_called_once()
        service._store_feedback_data.assert_called_once()
        
        # Verify suggestion was removed from active suggestions
        assert suggestion_id not in service.active_suggestions
    
    @pytest.mark.asyncio
    async def test_create_feedback_loop_suggestion_not_found(self, service):
        """Test feedback loop creation when suggestion is not found."""
        # Arrange
        suggestion_id = "non-existent-id"
        feedback_type = "accepted"
        feedback_data = {}
        
        # Act
        result = await service.create_feedback_loop(suggestion_id, feedback_type, feedback_data)
        
        # Assert
        assert result["status"] == "ignored"
        assert "not found" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories(self, service):
        """Test getting relevant memories for enhancement."""
        # Arrange
        code_context = "def process_data(): pass"
        file_path = "processor.py"
        user_id = str(uuid4())
        project_id = str(uuid4())
        
        # Mock memory service
        mock_memories = [
            MagicMock(
                content="Data processing example",
                memory_type="code",
                similarity_score=0.8
            ),
            MagicMock(
                content="Function documentation",
                memory_type="documentation",
                similarity_score=0.6
            )
        ]
        service.memory_service.search_memories.return_value = mock_memories
        
        # Act
        result = await service._get_relevant_memories(code_context, file_path, user_id, project_id)
        
        # Assert
        assert len(result) == 2
        assert result[0]["content"] == "Data processing example"
        assert result[0]["relevance"] == 0.8
        
        service.memory_service.search_memories.assert_called_once_with(
            query=code_context[:500],
            user_id=user_id,
            project_id=project_id,
            memory_type="code",
            limit=5
        )
    
    @pytest.mark.asyncio
    async def test_analyze_context_patterns(self, service):
        """Test analyzing context patterns for enhancement."""
        # Arrange
        code_context = "class DataProcessor:\n    def __init__(self):\n        pass"
        file_path = "processor.py"
        language = "python"
        user_id = str(uuid4())
        project_id = str(uuid4())
        
        # Mock pattern service
        service.pattern_service.analyze_code_patterns.return_value = {
            "patterns": [
                {
                    "pattern_type": "class_definition",
                    "description": "Class with constructor",
                    "confidence": 0.9
                },
                {
                    "pattern_type": "naming_convention",
                    "description": "CamelCase class name",
                    "confidence": 0.8
                }
            ]
        }
        
        # Act
        result = await service._analyze_context_patterns(
            code_context, file_path, language, user_id, project_id
        )
        
        # Assert
        assert len(result) == 2
        assert result[0]["type"] == "class_definition"
        assert result[0]["confidence"] == 0.9
        
        service.pattern_service.analyze_code_patterns.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ai_enhance_suggestion(self, service):
        """Test AI-powered suggestion enhancement."""
        # Arrange
        original = "def process(): pass"
        context = "data = [1, 2, 3]"
        memories = [{"content": "Use pandas for data processing"}]
        patterns = [{"type": "function_pattern"}]
        conventions = {"language": "python"}
        
        # Mock AI service
        service.ai_service.generate_ai_insights.return_value = {
            "enhanced_suggestion": "def process(data: List[int]) -> List[int]:\n    return [x * 2 for x in data]"
        }
        
        # Act
        result = await service._ai_enhance_suggestion(original, context, memories, patterns, conventions)
        
        # Assert
        assert "def process(data: List[int])" in result
        
        service.ai_service.generate_ai_insights.assert_called_once_with(
            context="copilot_enhancement",
            data={
                "original_suggestion": original,
                "code_context": context,
                "relevant_memories": memories[:3],
                "detected_patterns": patterns[:3],
                "conventions": conventions
            }
        )
    
    @pytest.mark.asyncio
    async def test_calculate_enhancement_confidence(self, service):
        """Test enhancement confidence calculation."""
        # Arrange
        original = "def test(): pass"
        enhanced = "def test() -> None:\n    \"\"\"Test function.\"\"\"\n    pass"
        memories = [
            {"relevance": 0.8},
            {"relevance": 0.6}
        ]
        patterns = [
            {"confidence": 0.9},
            {"confidence": 0.7}
        ]
        
        # Act
        confidence = await service._calculate_enhancement_confidence(original, enhanced, memories, patterns)
        
        # Assert
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high due to good context
    
    @pytest.mark.asyncio
    async def test_prepare_context_sources(self, service):
        """Test context sources preparation."""
        # Arrange
        memories = [{"content": "mem1"}, {"content": "mem2"}]
        patterns = [{"type": "pattern1"}]
        conventions = {"language": "python"}
        
        # Act
        sources = service._prepare_context_sources(memories, patterns, conventions)
        
        # Assert
        assert "2 relevant memories" in sources
        assert "1 code patterns" in sources
        assert "project conventions" in sources
    
    @pytest.mark.asyncio
    async def test_get_session_context(self, service):
        """Test getting session context."""
        # Arrange
        user_id = str(uuid4())
        
        mock_session = MagicMock(
            session_id="session-123",
            context_data={
                "focus": "Testing",
                "tasks": ["Write tests", "Fix bugs"]
            }
        )
        service.session_service.get_active_session.return_value = mock_session
        
        # Act
        result = await service._get_session_context(user_id)
        
        # Assert
        assert result["session_id"] == "session-123"
        assert result["focus"] == "Testing"
        assert result["tasks"] == ["Write tests", "Fix bugs"]
    
    @pytest.mark.asyncio
    async def test_get_session_context_no_active_session(self, service):
        """Test getting session context when no active session exists."""
        # Arrange
        user_id = str(uuid4())
        service.session_service.get_active_session.return_value = None
        
        # Act
        result = await service._get_session_context(user_id)
        
        # Assert
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_relevant_decisions(self, service):
        """Test getting relevant technical decisions."""
        # Arrange
        prompt = "How to handle user authentication"
        user_id = str(uuid4())
        project_id = str(uuid4())
        
        mock_decisions = [
            MagicMock(
                content="Use JWT tokens for authentication",
                metadata={"reasoning": "Stateless and secure"},
                created_at=datetime.utcnow()
            )
        ]
        service.memory_service.search_memories.return_value = mock_decisions
        
        # Act
        result = await service._get_relevant_decisions(prompt, user_id, project_id)
        
        # Assert
        assert len(result) == 1
        assert result[0]["decision"] == "Use JWT tokens for authentication"
        assert result[0]["reasoning"] == "Stateless and secure"
        
        service.memory_service.search_memories.assert_called_once_with(
            query=prompt[:200],
            user_id=user_id,
            project_id=project_id,
            memory_type="decision",
            limit=3
        )
    
    @pytest.mark.asyncio
    async def test_build_context_injection(self, service):
        """Test building context injection string."""
        # Arrange
        session_context = {"focus": "Testing"}
        project_context = {"type": "web_application"}
        decisions = [
            {"decision": "Use React for frontend"},
            {"decision": "Use PostgreSQL for database"}
        ]
        
        # Act
        result = service._build_context_injection(session_context, project_context, decisions)
        
        # Assert
        assert "KnowledgeHub Context" in result
        assert "Current session focus: Testing" in result
        assert "Project type: web_application" in result
        assert "Recent technical decisions:" in result
        assert "Use React for frontend" in result
    
    @pytest.mark.asyncio
    async def test_build_context_injection_empty(self, service):
        """Test building context injection with empty context."""
        # Act
        result = service._build_context_injection({}, {}, [])
        
        # Assert
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_process_feedback(self, service):
        """Test feedback processing."""
        # Arrange
        suggestion = CopilotSuggestion(
            original_suggestion="original",
            enhanced_suggestion="enhanced",
            confidence=0.8,
            context_sources=["memory"]
        )
        feedback_type = "accepted"
        feedback_data = {"satisfaction": "high"}
        
        # Act
        result = await service._process_feedback(suggestion, feedback_type, feedback_data)
        
        # Assert
        assert result["feedback_type"] == feedback_type
        assert result["original_confidence"] == 0.8
        assert result["context_sources"] == ["memory"]
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_update_models_from_feedback(self, service):
        """Test updating models from feedback."""
        # Arrange
        suggestion = CopilotSuggestion(
            original_suggestion="original",
            enhanced_suggestion="enhanced",
            confidence=0.8,
            context_sources=["memory"]
        )
        feedback_type = "accepted"
        learning_update = {"processed": True}
        
        # Act
        await service._update_models_from_feedback(suggestion, feedback_type, learning_update)
        
        # Assert
        service.memory_service.create_memory.assert_called_once()
        call_args = service.memory_service.create_memory.call_args[1]
        assert call_args["memory_type"] == "learning"
        assert "Copilot suggestion feedback" in call_args["content"]
    
    @pytest.mark.asyncio
    async def test_concurrent_enhancement_requests(self, service):
        """Test handling multiple concurrent enhancement requests."""
        import asyncio
        
        # Arrange
        suggestions = [f"def func_{i}(): pass" for i in range(5)]
        contexts = [{"code": f"context_{i}"} for i in range(5)]
        
        # Mock dependencies
        service._get_relevant_memories = AsyncMock(return_value=[])
        service._analyze_context_patterns = AsyncMock(return_value=[])
        service._ai_enhance_suggestion = AsyncMock(side_effect=lambda orig, *args: f"enhanced_{orig}")
        service._calculate_enhancement_confidence = AsyncMock(return_value=0.7)
        
        # Act
        tasks = [
            service.enhance_suggestion(suggestion, context)
            for suggestion, context in zip(suggestions, contexts)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        assert len(results) == 5
        for i, result in enumerate(results):
            assert not isinstance(result, Exception)
            assert isinstance(result, CopilotSuggestion)
            assert f"enhanced_def func_{i}()" in result.enhanced_suggestion
    
    @pytest.mark.asyncio
    async def test_webhook_error_handling(self, service):
        """Test webhook error handling."""
        # Arrange
        webhook_type = "suggestion_request"
        payload = {"invalid": "payload"}  # Missing required fields
        
        # Mock enhancement to fail
        service.enhance_suggestion = AsyncMock(side_effect=Exception("Enhancement failed"))
        
        # Act & Assert
        with pytest.raises(Exception):
            await service.receive_webhook(webhook_type, payload)
    
    @pytest.mark.asyncio
    async def test_suggestion_timeout_handling(self, service):
        """Test handling of suggestion enhancement timeouts."""
        # Arrange
        original_suggestion = "def slow_function(): pass"
        context = {"code": "test"}
        
        # Mock slow AI service
        service._ai_enhance_suggestion = AsyncMock(side_effect=asyncio.TimeoutError("Timeout"))
        
        # Act
        with patch('asyncio.wait_for'):  # Mock timeout mechanism
            result = await service.enhance_suggestion(original_suggestion, context)
        
        # Assert
        assert isinstance(result, CopilotSuggestion)
        # Should fallback to original on timeout
        assert result.enhanced_suggestion == original_suggestion
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_feedback(self, service):
        """Test cleanup of processed suggestions after feedback."""
        # Arrange
        suggestion_ids = [str(uuid4()) for _ in range(3)]
        
        for suggestion_id in suggestion_ids:
            suggestion = CopilotSuggestion(
                original_suggestion="test",
                enhanced_suggestion="enhanced test",
                confidence=0.8,
                context_sources=["test"]
            )
            service.active_suggestions[suggestion_id] = suggestion
        
        # Mock feedback processing
        service._process_feedback = AsyncMock(return_value={})
        service._update_models_from_feedback = AsyncMock()
        service._store_feedback_data = AsyncMock()
        
        # Act
        for suggestion_id in suggestion_ids:
            await service.create_feedback_loop(suggestion_id, "accepted", {})
        
        # Assert
        assert len(service.active_suggestions) == 0  # All should be cleaned up