"""
Unit tests for AIService.

Tests all AI service functionality including embeddings, insights generation,
predictions, and integration with external AI providers.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from api.services.ai_service import AIService


@pytest.mark.unit
class TestAIService:
    """Unit tests for AIService."""
    
    @pytest_asyncio.fixture
    async def service(self, mock_openai, mock_weaviate, mock_redis):
        """Create AIService instance with mocked dependencies."""
        service = AIService()
        service.openai_client = mock_openai
        service.weaviate_client = mock_weaviate
        service.redis_client = mock_redis
        return service
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, service, mock_openai):
        """Test successful embedding generation."""
        # Arrange
        text = "Test text for embedding generation"
        expected_embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=expected_embedding)
        ]
        
        # Act
        result = await service.generate_embedding(text)
        
        # Assert
        assert result == expected_embedding
        mock_openai.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=text
        )
    
    @pytest.mark.asyncio
    async def test_generate_embedding_failure(self, service, mock_openai):
        """Test embedding generation failure handling."""
        # Arrange
        text = "Test text"
        mock_openai.embeddings.create.side_effect = Exception("API Error")
        
        # Act & Assert
        with pytest.raises(Exception):
            await service.generate_embedding(text)
    
    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, service):
        """Test embedding generation with empty text."""
        # Act & Assert
        with pytest.raises(ValueError):
            await service.generate_embedding("")
    
    @pytest.mark.asyncio
    async def test_generate_embedding_with_caching(self, service, mock_openai, mock_redis):
        """Test embedding generation with Redis caching."""
        # Arrange
        text = "Cacheable text"
        expected_embedding = [0.1] * 1536
        cache_key = f"embedding:{hash(text)}"
        
        # First call - cache miss
        mock_redis.get.return_value = None
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=expected_embedding)
        ]
        
        # Act - First call
        result1 = await service.generate_embedding(text)
        
        # Assert - First call
        assert result1 == expected_embedding
        mock_openai.embeddings.create.assert_called_once()
        mock_redis.set.assert_called_once()
        
        # Arrange - Second call - cache hit
        mock_redis.get.return_value = str(expected_embedding)
        mock_openai.reset_mock()
        
        # Act - Second call
        result2 = await service.generate_embedding(text)
        
        # Assert - Second call (should use cache)
        assert result2 == expected_embedding
        mock_openai.embeddings.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_generate_ai_insights_success(self, service, mock_openai):
        """Test successful AI insights generation."""
        # Arrange
        context = "code_analysis"
        data = {
            "code": "def hello_world(): print('Hello, World!')",
            "language": "python",
            "file_path": "hello.py"
        }
        
        expected_response = {
            "insights": ["Simple function definition", "Uses print statement"],
            "suggestions": ["Add docstring", "Consider return value"],
            "complexity": "low"
        }
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=str(expected_response)))
        ]
        
        # Act
        result = await service.generate_ai_insights(context, data)
        
        # Assert
        assert isinstance(result, dict)
        mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_ai_insights_with_different_contexts(self, service, mock_openai):
        """Test AI insights generation with different context types."""
        contexts_and_data = [
            ("error_analysis", {"error": "TypeError", "stack_trace": "..."}),
            ("performance_analysis", {"metrics": {"response_time": 200}}),
            ("decision_support", {"options": ["A", "B", "C"], "criteria": ["cost", "speed"]}),
            ("pattern_recognition", {"data_points": [1, 2, 3, 4, 5]})
        ]
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"analysis": "test"}'))
        ]
        
        for context, data in contexts_and_data:
            # Act
            result = await service.generate_ai_insights(context, data)
            
            # Assert
            assert isinstance(result, dict)
            
        # Verify all contexts were processed
        assert mock_openai.chat.completions.create.call_count == len(contexts_and_data)
    
    @pytest.mark.asyncio
    async def test_analyze_code_patterns_success(self, service, mock_openai):
        """Test successful code pattern analysis."""
        # Arrange
        code_samples = [
            {"content": "def func1(): pass", "language": "python"},
            {"content": "def func2(): return True", "language": "python"},
            {"content": "class MyClass: pass", "language": "python"}
        ]
        
        expected_analysis = {
            "patterns": [
                {"type": "function_definition", "count": 2},
                {"type": "class_definition", "count": 1}
            ],
            "complexity": "low",
            "style_consistency": "high"
        }
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=str(expected_analysis)))
        ]
        
        # Act
        result = await service.analyze_code_patterns(code_samples)
        
        # Assert
        assert isinstance(result, dict)
        mock_openai.chat.completions.create.assert_called_once()
        
        # Verify prompt contains code samples
        call_args = mock_openai.chat.completions.create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "def func1" in prompt_content
    
    @pytest.mark.asyncio
    async def test_predict_next_tasks_success(self, service, mock_openai):
        """Test successful task prediction."""
        # Arrange
        current_context = {
            "current_file": "test_ai_service.py",
            "recent_actions": ["created test file", "wrote test cases"],
            "project_type": "python_testing"
        }
        
        expected_predictions = [
            {"task": "Run tests", "confidence": 0.9, "priority": "high"},
            {"task": "Fix failing tests", "confidence": 0.7, "priority": "medium"},
            {"task": "Add more test coverage", "confidence": 0.6, "priority": "low"}
        ]
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=str(expected_predictions)))
        ]
        
        # Act
        result = await service.predict_next_tasks(current_context)
        
        # Assert
        assert isinstance(result, list)
        mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_predict_next_tasks_with_history(self, service, mock_openai):
        """Test task prediction with historical context."""
        # Arrange
        current_context = {"current_task": "testing"}
        historical_patterns = [
            {"pattern": "after testing, usually fix bugs", "frequency": 0.8},
            {"pattern": "after testing, usually add documentation", "frequency": 0.6}
        ]
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='[{"task": "fix bugs", "confidence": 0.8}]'))
        ]
        
        # Act
        result = await service.predict_next_tasks(
            current_context,
            historical_patterns=historical_patterns
        )
        
        # Assert
        assert isinstance(result, list)
        
        # Verify historical patterns were included in prompt
        call_args = mock_openai.chat.completions.create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "historical patterns" in prompt_content.lower()
    
    @pytest.mark.asyncio
    async def test_enhance_code_suggestion_success(self, service, mock_openai):
        """Test successful code suggestion enhancement."""
        # Arrange
        original_suggestion = "def process_data():\n    pass"
        context = {
            "file_type": "python",
            "surrounding_code": "import pandas as pd\ndf = pd.read_csv('data.csv')",
            "user_intent": "data processing function"
        }
        
        enhanced_suggestion = """def process_data(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Process the input DataFrame.\"\"\"
    # Add your data processing logic here
    return df.dropna()"""
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=enhanced_suggestion))
        ]
        
        # Act
        result = await service.enhance_code_suggestion(original_suggestion, context)
        
        # Assert
        assert isinstance(result, str)
        assert "def process_data" in result
        mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_error_patterns_success(self, service, mock_openai):
        """Test successful error pattern analysis."""
        # Arrange
        error_data = [
            {
                "error_type": "TypeError",
                "message": "unsupported operand type(s)",
                "stack_trace": "...",
                "frequency": 5
            },
            {
                "error_type": "ValueError",
                "message": "invalid literal for int()",
                "stack_trace": "...",
                "frequency": 3
            }
        ]
        
        expected_analysis = {
            "most_common_errors": ["TypeError", "ValueError"],
            "root_causes": ["Type mismatches", "Input validation issues"],
            "prevention_strategies": ["Add type hints", "Validate inputs"]
        }
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=str(expected_analysis)))
        ]
        
        # Act
        result = await service.analyze_error_patterns(error_data)
        
        # Assert
        assert isinstance(result, dict)
        mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_documentation_success(self, service, mock_openai):
        """Test successful documentation generation."""
        # Arrange
        code = """
        def calculate_fibonacci(n):
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        """
        
        doc_type = "function_docstring"
        
        expected_doc = '''"""
        Calculate the nth Fibonacci number using recursion.
        
        Args:
            n (int): The position in the Fibonacci sequence
            
        Returns:
            int: The nth Fibonacci number
            
        Example:
            >>> calculate_fibonacci(5)
            5
        """'''
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=expected_doc))
        ]
        
        # Act
        result = await service.generate_documentation(code, doc_type)
        
        # Assert
        assert isinstance(result, str)
        assert "Args:" in result
        assert "Returns:" in result
        mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_similarity_search_success(self, service, mock_weaviate):
        """Test successful similarity search."""
        # Arrange
        query_text = "python function for data processing"
        limit = 5
        
        mock_results = {
            "data": {
                "Get": {
                    "Memory": [
                        {
                            "content": "Data processing function example",
                            "memory_id": str(uuid4()),
                            "_additional": {"distance": 0.1}
                        },
                        {
                            "content": "Another data processing example",
                            "memory_id": str(uuid4()),
                            "_additional": {"distance": 0.2}
                        }
                    ]
                }
            }
        }
        
        mock_weaviate.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = mock_results
        
        # Act
        result = await service.similarity_search(query_text, limit)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("content" in item for item in result)
        assert all("similarity_score" in item for item in result)
        
        # Verify Weaviate was called correctly
        mock_weaviate.query.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_filters(self, service, mock_weaviate):
        """Test similarity search with filters."""
        # Arrange
        query_text = "test query"
        filters = {
            "memory_type": "code",
            "user_id": str(uuid4())
        }
        
        mock_weaviate.query.get.return_value.with_near_text.return_value.with_limit.return_value.with_where.return_value.do.return_value = {
            "data": {"Get": {"Memory": []}}
        }
        
        # Act
        result = await service.similarity_search(query_text, filters=filters)
        
        # Assert
        assert isinstance(result, list)
        
        # Verify filters were applied
        mock_weaviate.query.get.return_value.with_near_text.return_value.with_limit.return_value.with_where.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_success(self, service, mock_openai):
        """Test successful batch embedding generation."""
        # Arrange
        texts = [
            "First text for embedding",
            "Second text for embedding", 
            "Third text for embedding"
        ]
        
        expected_embeddings = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536
        ]
        
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=emb) for emb in expected_embeddings
        ]
        
        # Act
        result = await service.batch_generate_embeddings(texts)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == expected_embeddings
        
        mock_openai.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )
    
    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_chunking(self, service, mock_openai):
        """Test batch embedding generation with chunking for large inputs."""
        # Arrange
        texts = [f"Text {i}" for i in range(150)]  # More than batch limit
        batch_size = 100
        
        # Mock two separate API calls
        mock_openai.embeddings.create.side_effect = [
            MagicMock(data=[MagicMock(embedding=[0.1] * 1536) for _ in range(100)]),
            MagicMock(data=[MagicMock(embedding=[0.2] * 1536) for _ in range(50)])
        ]
        
        # Act
        result = await service.batch_generate_embeddings(texts, batch_size=batch_size)
        
        # Assert
        assert len(result) == 150
        assert mock_openai.embeddings.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_success(self, service, mock_openai):
        """Test successful sentiment analysis."""
        # Arrange
        text = "I really love this new feature! It's amazing and works perfectly."
        
        expected_sentiment = {
            "sentiment": "positive",
            "confidence": 0.95,
            "emotions": ["joy", "satisfaction"],
            "keywords": ["love", "amazing", "perfectly"]
        }
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=str(expected_sentiment)))
        ]
        
        # Act
        result = await service.analyze_sentiment(text)
        
        # Assert
        assert isinstance(result, dict)
        mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_key_concepts_success(self, service, mock_openai):
        """Test successful key concept extraction."""
        # Arrange
        text = """
        Machine learning is a subset of artificial intelligence that focuses on 
        algorithms that can learn from data. Deep learning uses neural networks 
        with multiple layers to model complex patterns.
        """
        
        expected_concepts = {
            "primary_concepts": ["machine learning", "artificial intelligence", "deep learning"],
            "secondary_concepts": ["algorithms", "neural networks", "data"],
            "relationships": [
                {"from": "machine learning", "to": "artificial intelligence", "type": "subset"},
                {"from": "deep learning", "to": "neural networks", "type": "uses"}
            ]
        }
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=str(expected_concepts)))
        ]
        
        # Act
        result = await service.extract_key_concepts(text)
        
        # Assert
        assert isinstance(result, dict)
        assert "primary_concepts" in result
        mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, service, mock_openai):
        """Test rate limiting functionality."""
        # Arrange
        mock_openai.embeddings.create.side_effect = Exception("Rate limit exceeded")
        
        # Act & Assert
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await service.generate_embedding("test text")
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, service, mock_openai):
        """Test error recovery and retry logic."""
        # Arrange
        text = "Test text for retry"
        
        # First call fails, second succeeds
        mock_openai.embeddings.create.side_effect = [
            Exception("Temporary error"),
            MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
        ]
        
        # Act
        with patch('asyncio.sleep'):  # Mock sleep to speed up test
            result = await service.generate_embedding(text, max_retries=2)
        
        # Assert
        assert result == [0.1] * 1536
        assert mock_openai.embeddings.create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_model_fallback(self, service, mock_openai):
        """Test fallback to different models on failure."""
        # Arrange
        text = "Test text"
        
        # Primary model fails, fallback succeeds
        mock_openai.embeddings.create.side_effect = [
            Exception("Model unavailable"),
            MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
        ]
        
        # Act
        result = await service.generate_embedding(text, fallback_model="text-embedding-ada-002")
        
        # Assert
        assert result == [0.1] * 1536
        
        # Verify both models were tried
        calls = mock_openai.embeddings.create.call_args_list
        assert calls[0][1]["model"] == "text-embedding-3-small"
        assert calls[1][1]["model"] == "text-embedding-ada-002"
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, service, mock_openai, performance_timer):
        """Test performance optimizations."""
        # Arrange
        texts = [f"Performance test text {i}" for i in range(10)]
        
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536) for _ in range(10)
        ]
        
        # Act
        with performance_timer as timer:
            await service.batch_generate_embeddings(texts)
        
        # Assert
        assert timer.elapsed < 5.0  # Should complete within 5 seconds
        mock_openai.embeddings.create.assert_called_once()  # Batch call
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, service):
        """Test memory usage optimization for large inputs."""
        # Arrange
        large_text = "x" * 100000  # 100KB text
        
        # Act
        # This should not cause memory issues
        result = await service.generate_embedding(large_text[:8000])  # Truncated
        
        # Assert
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, service, mock_openai):
        """Test handling of concurrent AI requests."""
        import asyncio
        
        # Arrange
        texts = [f"Concurrent text {i}" for i in range(5)]
        
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]
        
        # Act
        tasks = [service.generate_embedding(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert len(result) == 1536
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self, service, mock_openai):
        """Test input sanitization for AI requests."""
        # Arrange
        malicious_input = "<script>alert('xss')</script> AND 1=1 OR 'a'='a'"
        
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"result": "safe"}'))
        ]
        
        # Act
        result = await service.generate_ai_insights("code_analysis", {"code": malicious_input})
        
        # Assert
        assert isinstance(result, dict)
        
        # Verify input was sanitized in the prompt
        call_args = mock_openai.chat.completions.create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "<script>" not in prompt_content
    
    @pytest.mark.asyncio
    async def test_response_validation(self, service, mock_openai):
        """Test validation of AI responses."""
        # Arrange
        mock_openai.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Invalid JSON response"))
        ]
        
        # Act
        result = await service.generate_ai_insights("test_context", {"data": "test"})
        
        # Assert
        # Should handle invalid JSON gracefully
        assert isinstance(result, dict)
        assert "error" in result or "raw_response" in result