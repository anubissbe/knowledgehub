
import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

class TestCriticalEndpoints:
    """Test all critical API endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] in ["healthy", "degraded"]
    
    @pytest.mark.asyncio
    async def test_rag_endpoints(self):
        endpoints = [
            "/api/rag/enhanced/health",
            "/api/agents/health",
            "/api/zep/health",
            "/api/graphrag/health"
        ]
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            for endpoint in endpoints:
                response = await client.get(endpoint)
                assert response.status_code == 200

class TestSecurity:
    """Test security features"""
    
    @pytest.mark.asyncio
    async def test_authentication_required(self):
        # Test that protected endpoints require auth
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/memory/create")
            assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self):
        # Test SQL injection protection
        malicious_input = "'; DROP TABLE users; --"
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/search",
                json={"query": malicious_input}
            )
            # Should be safely handled
            assert response.status_code in [200, 400, 422]

class TestPerformance:
    """Test performance requirements"""
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        import time
        async with AsyncClient(app=app, base_url="http://test") as client:
            start = time.time()
            response = await client.get("/health")
            duration = time.time() - start
            assert duration < 0.5  # 500ms max
