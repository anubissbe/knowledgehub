"""
Pytest configuration and shared fixtures for KnowledgeHub tests.

Provides common test fixtures, database setup, and test utilities
for comprehensive testing of all KnowledgeHub components.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Import KnowledgeHub components
from api.main import app
from api.database import Base, get_db_session
from api.models.memory import Memory
from api.models.session import Session
from api.models.analytics import Metric
from api.models.user import User
from api.services.memory_service import MemoryService
from api.services.session_service import SessionService
from api.services.ai_service import AIService
from api.services.pattern_service import PatternService
from api.services.copilot_service import CopilotEnhancementService
from api.services.ai_feedback_loop import AIFeedbackLoop

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5433/knowledgehub_test"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    TestSessionLocal = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        # Start a transaction
        await session.begin()
        
        yield session
        
        # Rollback transaction
        await session.rollback()


@pytest.fixture
def override_get_db(db_session):
    """Override the get_db_session dependency for tests."""
    async def _override_get_db():
        yield db_session
    
    return _override_get_db


@pytest.fixture
def test_client(override_get_db):
    """Create test client with database override."""
    app.dependency_overrides[get_db_session] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(override_get_db):
    """Create async test client."""
    app.dependency_overrides[get_db_session] = override_get_db
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


# Test data fixtures
@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "user_id": str(uuid4()),
        "username": "test_user",
        "email": "test@example.com",
        "preferences": {"theme": "dark", "ai_enabled": True}
    }


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "content": "Test memory content for unit testing",
        "memory_type": "code",
        "user_id": str(uuid4()),
        "project_id": str(uuid4()),
        "tags": ["test", "unit_test"],
        "metadata": {
            "file_path": "/test/path.py",
            "language": "python",
            "confidence": 0.9
        }
    }


@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        "session_id": str(uuid4()),
        "user_id": str(uuid4()),
        "session_type": "coding",
        "project_id": str(uuid4()),
        "status": "active",
        "context_data": {
            "focus": "Testing implementation",
            "current_file": "test_conftest.py",
            "tasks": ["Write unit tests", "Add integration tests"]
        }
    }


@pytest.fixture
def sample_metric_data():
    """Sample metric data for testing."""
    return {
        "name": "test_metric",
        "value": 42.0,
        "metric_type": "gauge",
        "tags": {"component": "test", "environment": "testing"},
        "metadata": {"description": "Test metric for unit testing"}
    }


@pytest_asyncio.fixture
async def test_user(db_session, sample_user_data):
    """Create a test user in the database."""
    user = User(**sample_user_data)
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_memory(db_session, sample_memory_data):
    """Create a test memory in the database."""
    memory = Memory(**sample_memory_data)
    db_session.add(memory)
    await db_session.commit()
    await db_session.refresh(memory)
    return memory


@pytest_asyncio.fixture
async def test_session(db_session, sample_session_data):
    """Create a test session in the database."""
    session = Session(**sample_session_data)
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return session


@pytest_asyncio.fixture
async def test_metric(db_session, sample_metric_data):
    """Create a test metric in the database."""
    metric = Metric(**sample_metric_data)
    db_session.add(metric)
    await db_session.commit()
    await db_session.refresh(metric)
    return metric


# Service fixtures with mocked dependencies
@pytest_asyncio.fixture
async def memory_service(db_session):
    """Create memory service with test database."""
    service = MemoryService()
    # Mock external dependencies if needed
    return service


@pytest_asyncio.fixture
async def session_service(db_session):
    """Create session service with test database."""
    service = SessionService()
    return service


@pytest_asyncio.fixture
async def ai_service():
    """Create AI service with mocked external calls."""
    service = AIService()
    # Mock external AI API calls
    service.openai_client = AsyncMock()
    service.weaviate_client = MagicMock()
    return service


@pytest_asyncio.fixture
async def pattern_service(db_session):
    """Create pattern service with test database."""
    service = PatternService()
    return service


@pytest_asyncio.fixture
async def copilot_service():
    """Create Copilot service with mocked dependencies."""
    service = CopilotEnhancementService()
    # Mock external services
    service.memory_service = AsyncMock()
    service.ai_service = AsyncMock()
    service.pattern_service = AsyncMock()
    return service


@pytest_asyncio.fixture
async def feedback_loop():
    """Create feedback loop with mocked dependencies."""
    loop = AIFeedbackLoop()
    # Mock external services
    loop.memory_service = AsyncMock()
    loop.ai_service = AsyncMock()
    return loop


# Test data generators
@pytest.fixture
def memory_factory():
    """Factory for creating test memories."""
    def _create_memory(**kwargs):
        default_data = {
            "content": "Factory-generated test memory",
            "memory_type": "test",
            "user_id": str(uuid4()),
            "project_id": str(uuid4()),
            "tags": ["factory", "test"],
            "metadata": {"generated": True}
        }
        default_data.update(kwargs)
        return Memory(**default_data)
    
    return _create_memory


@pytest.fixture
def session_factory():
    """Factory for creating test sessions."""
    def _create_session(**kwargs):
        default_data = {
            "session_id": str(uuid4()),
            "user_id": str(uuid4()),
            "session_type": "test",
            "project_id": str(uuid4()),
            "status": "active",
            "context_data": {"test": True}
        }
        default_data.update(kwargs)
        return Session(**default_data)
    
    return _create_session


@pytest.fixture
def metric_factory():
    """Factory for creating test metrics."""
    def _create_metric(**kwargs):
        default_data = {
            "name": "test_metric",
            "value": 1.0,
            "metric_type": "counter",
            "tags": {"test": "true"},
            "metadata": {"factory": True}
        }
        default_data.update(kwargs)
        return Metric(**default_data)
    
    return _create_metric


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer utility for performance tests."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.utcnow()
        
        def stop(self):
            self.end_time = datetime.utcnow()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return None
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, *args):
            self.stop()
    
    return Timer


@pytest.fixture
def load_test_data():
    """Generate large datasets for load testing."""
    def _generate_data(count: int, data_type: str = "memory"):
        if data_type == "memory":
            return [
                {
                    "content": f"Load test memory {i}",
                    "memory_type": "load_test",
                    "user_id": str(uuid4()),
                    "project_id": str(uuid4()),
                    "tags": ["load_test", f"batch_{i//100}"],
                    "metadata": {"index": i, "batch": i//100}
                }
                for i in range(count)
            ]
        elif data_type == "session":
            return [
                {
                    "session_id": str(uuid4()),
                    "user_id": str(uuid4()),
                    "session_type": "load_test",
                    "project_id": str(uuid4()),
                    "status": "active",
                    "context_data": {"load_test": True, "index": i}
                }
                for i in range(count)
            ]
        elif data_type == "metric":
            return [
                {
                    "name": "load_test_metric",
                    "value": float(i),
                    "metric_type": "gauge",
                    "tags": {"load_test": "true", "batch": str(i//100)},
                    "metadata": {"index": i}
                }
                for i in range(count)
            ]
    
    return _generate_data


# Mock external services
@pytest.fixture
def mock_weaviate():
    """Mock Weaviate client for testing."""
    mock_client = MagicMock()
    mock_client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = {
        "data": {
            "Get": {
                "Memory": [
                    {
                        "content": "Mocked memory content",
                        "memory_type": "test",
                        "_additional": {"distance": 0.1}
                    }
                ]
            }
        }
    }
    return mock_client


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing."""
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1] * 1536)
    ]
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Mocked AI response"))
    ]
    return mock_client


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_client = AsyncMock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1
    return mock_client


# Test utilities
@pytest.fixture
def assert_response():
    """Utility for asserting API responses."""
    def _assert_response(response, expected_status=200, expected_keys=None):
        assert response.status_code == expected_status
        
        if expected_keys:
            response_data = response.json()
            for key in expected_keys:
                assert key in response_data
        
        return response.json() if response.status_code == 200 else None
    
    return _assert_response


@pytest.fixture
def create_test_data():
    """Utility for creating test data in bulk."""
    async def _create_test_data(db_session, model_class, data_list):
        objects = [model_class(**data) for data in data_list]
        for obj in objects:
            db_session.add(obj)
        await db_session.commit()
        
        for obj in objects:
            await db_session.refresh(obj)
        
        return objects
    
    return _create_test_data


# Integration test fixtures
@pytest_asyncio.fixture
async def integration_environment(test_engine):
    """Set up integration test environment."""
    # Create test database with sample data
    TestSessionLocal = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        # Create sample users
        users = [
            User(
                user_id=str(uuid4()),
                username=f"integration_user_{i}",
                email=f"integration{i}@test.com",
                preferences={"test": True}
            )
            for i in range(3)
        ]
        
        for user in users:
            session.add(user)
        
        await session.commit()
        
        # Create sample memories
        memories = [
            Memory(
                content=f"Integration test memory {i}",
                memory_type="integration",
                user_id=users[i % len(users)].user_id,
                project_id=str(uuid4()),
                tags=["integration", "test"],
                metadata={"test_index": i}
            )
            for i in range(10)
        ]
        
        for memory in memories:
            session.add(memory)
        
        await session.commit()
        
        # Create sample sessions
        sessions = [
            Session(
                session_id=str(uuid4()),
                user_id=users[i % len(users)].user_id,
                session_type="integration_test",
                project_id=str(uuid4()),
                status="active",
                context_data={"integration_test": True, "index": i}
            )
            for i in range(5)
        ]
        
        for session_obj in sessions:
            session.add(session_obj)
        
        await session.commit()
        
        yield {
            "users": users,
            "memories": memories,
            "sessions": sessions
        }


# Security testing fixtures
@pytest.fixture
def security_test_headers():
    """Headers for security testing."""
    return {
        "valid": {
            "Authorization": "Bearer valid_test_token",
            "Content-Type": "application/json"
        },
        "invalid": {
            "Authorization": "Bearer invalid_token",
            "Content-Type": "application/json"
        },
        "malicious": {
            "Authorization": "Bearer <script>alert('xss')</script>",
            "Content-Type": "application/json",
            "X-Forwarded-For": "192.168.1.1",
            "User-Agent": "Mozilla/5.0 (compatible; Malicious Bot)"
        }
    }


@pytest.fixture
def sql_injection_payloads():
    """SQL injection test payloads."""
    return [
        "'; DROP TABLE memories; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM users --",
        "'; INSERT INTO memories (content) VALUES ('hacked'); --",
        "' OR 1=1 --",
        "admin'--",
        "admin' #",
        "admin'/*",
        "' or 1=1#",
        "' or 1=1--",
        "' or 1=1/*"
    ]


# Cleanup utilities
@pytest.fixture(autouse=True)
async def cleanup_test_data(db_session):
    """Automatically clean up test data after each test."""
    yield
    
    # Clean up test data
    try:
        await db_session.execute(text("DELETE FROM metrics WHERE tags->>'test' = 'true'"))
        await db_session.execute(text("DELETE FROM memories WHERE tags @> '[\"test\"]'"))
        await db_session.execute(text("DELETE FROM sessions WHERE context_data->>'test' = 'true'"))
        await db_session.execute(text("DELETE FROM users WHERE username LIKE '%test%'"))
        await db_session.commit()
    except Exception:
        await db_session.rollback()


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )