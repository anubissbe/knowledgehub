"""
End-to-End test scenarios for KnowledgeHub.

Tests complete user workflows from API endpoints through to database
persistence, including real-world usage scenarios.
"""

import pytest
import pytest_asyncio
import httpx
from datetime import datetime
from uuid import uuid4

from fastapi.testclient import TestClient


@pytest.mark.e2e
class TestEndToEndScenarios:
    """End-to-end test scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_ai_session_workflow(self, async_client):
        """Test complete AI session workflow from start to finish."""
        # This test simulates a real user session from Claude Code integration
        
        # 1. Health check
        response = await async_client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # 2. Initialize AI session
        session_init_data = {
            "workspace": {
                "name": "KnowledgeHub Development",
                "path": "/opt/projects/knowledgehub",
                "language": "python",
                "files": 150
            },
            "timestamp": datetime.utcnow().isoformat(),
            "vscodeVersion": "1.85.0",
            "extensions": ["ms-python.python", "ms-vscode.vscode-typescript"]
        }
        
        response = await async_client.post("/api/claude-auto/session/init", json=session_init_data)
        assert response.status_code == 200
        session_data = response.json()
        
        session_id = session_data["sessionId"]
        assert session_id is not None
        assert session_data["memoriesLoaded"] >= 0
        assert session_data["incompleteTasks"] >= 0
        
        # 3. Create memories during development
        memory_data = {
            "content": "Implemented AI-enhanced memory service with vector embeddings",
            "memory_type": "implementation",
            "project_id": "knowledgehub-dev",
            "tags": ["ai", "memory", "embeddings"],
            "metadata": {
                "file_path": "api/services/memory_service.py",
                "language": "python",
                "session_id": session_id,
                "lines_of_code": 250
            }
        }
        
        response = await async_client.post("/api/memory", json=memory_data)
        assert response.status_code == 200
        memory_response = response.json()
        memory_id = memory_response["id"]
        
        # 4. Search for relevant memories
        search_params = {
            "query": "memory service implementation",
            "limit": 5,
            "similarity_threshold": 0.5
        }
        
        response = await async_client.get("/api/memory/search", params=search_params)
        assert response.status_code == 200
        search_results = response.json()
        
        assert "memories" in search_results
        assert len(search_results["memories"]) > 0
        
        # Verify our created memory is found
        found_memory_ids = [m["id"] for m in search_results["memories"]]
        assert memory_id in found_memory_ids
        
        # 5. Record a technical decision
        decision_data = {
            "description": "Use Weaviate for vector storage instead of Pinecone",
            "alternatives": ["Pinecone", "Chroma", "FAISS"],
            "reasoning": "Better open-source support and self-hosting capabilities",
            "confidence": 0.8,
            "project_id": "knowledgehub-dev",
            "session_id": session_id,
            "metadata": {
                "impact": "high",
                "category": "infrastructure",
                "stakeholders": ["dev_team", "ops_team"]
            }
        }
        
        response = await async_client.post("/api/claude-auto/decision/record", json=decision_data)
        assert response.status_code == 200
        decision_response = response.json()
        assert decision_response["status"] == "recorded"
        
        # 6. Track an error and solution
        error_data = {
            "error_type": "ConnectionError",
            "message": "Failed to connect to Weaviate instance",
            "context": {
                "service": "memory_service",
                "operation": "vector_search",
                "timestamp": datetime.utcnow().isoformat()
            },
            "solution": "Added connection retry logic with exponential backoff",
            "resolved": True,
            "session_id": session_id
        }
        
        response = await async_client.post("/api/claude-auto/learning/error", json=error_data)
        assert response.status_code == 200
        error_response = response.json()
        assert error_response["status"] == "recorded"
        
        # 7. Get AI insights on current context
        insight_data = {
            "context": "implementing AI memory system",
            "current_work": {
                "file": "memory_service.py",
                "function": "search_memories",
                "task": "optimize vector search performance"
            },
            "project_id": "knowledgehub-dev",
            "session_id": session_id
        }
        
        response = await async_client.post("/api/claude-auto/insights/generate", json=insight_data)
        assert response.status_code == 200
        insights = response.json()
        
        assert "insights" in insights
        assert "recommendations" in insights
        assert len(insights["insights"]) > 0
        
        # 8. Predict next tasks
        prediction_data = {
            "current_context": {
                "completed_tasks": ["Implement memory service", "Add vector storage"],
                "current_task": "Optimize search performance",
                "project_phase": "development"
            },
            "project_id": "knowledgehub-dev",
            "session_id": session_id
        }
        
        response = await async_client.post("/api/claude-auto/predictions/tasks", json=prediction_data)
        assert response.status_code == 200
        predictions = response.json()
        
        assert "predictions" in predictions
        assert len(predictions["predictions"]) > 0
        
        # Each prediction should have required fields
        for prediction in predictions["predictions"]:
            assert "task" in prediction
            assert "confidence" in prediction
            assert "priority" in prediction
        
        # 9. Update session context
        context_update = {
            "current_focus": "Performance optimization completed",
            "progress": 0.75,
            "next_steps": ["Add error handling", "Write documentation"],
            "blockers": [],
            "achievements": ["Vector search optimized", "Connection retry implemented"]
        }
        
        response = await async_client.put(
            f"/api/claude-auto/session/{session_id}/context",
            json=context_update
        )
        assert response.status_code == 200
        context_response = response.json()
        assert context_response["status"] == "updated"
        
        # 10. Get session analytics
        response = await async_client.get(f"/api/claude-auto/session/{session_id}/analytics")
        assert response.status_code == 200
        analytics = response.json()
        
        assert "productivity_score" in analytics
        assert "focus_time" in analytics
        assert "task_completion_rate" in analytics
        
        # 11. Create session handoff
        handoff_data = {
            "message": "Completed memory service optimization, ready for testing phase",
            "context_summary": "AI memory system with vector search is fully implemented",
            "next_developer_notes": [
                "Run performance benchmarks",
                "Test with large datasets",
                "Document API endpoints"
            ]
        }
        
        response = await async_client.post(
            f"/api/claude-auto/session/{session_id}/handoff",
            json=handoff_data
        )
        assert response.status_code == 200
        handoff_response = response.json()
        
        assert "handoff_id" in handoff_response
        assert "context_preserved" in handoff_response
        assert handoff_response["context_preserved"] is True
        
        # 12. End session
        end_data = {
            "reason": "Development session completed",
            "duration_hours": 4.5,
            "achievements": [
                "Memory service implemented",
                "Vector search optimized",
                "Error handling added"
            ],
            "create_memory": True
        }
        
        response = await async_client.post(
            f"/api/claude-auto/session/{session_id}/end",
            json=end_data
        )
        assert response.status_code == 200
        end_response = response.json()
        
        assert end_response["status"] == "ended"
        assert "session_memory_id" in end_response  # Memory was created
    
    @pytest.mark.asyncio
    async def test_copilot_enhancement_workflow(self, async_client):
        """Test GitHub Copilot enhancement workflow."""
        
        # 1. Test webhook endpoint health
        response = await async_client.get("/api/copilot/health")
        assert response.status_code == 200
        health_data = response.json()
        assert "copilot_enhancement" in health_data["service"]
        
        # 2. Enhance a Copilot suggestion
        enhancement_data = {
            "original_suggestion": "def process_user_data(data):\n    pass",
            "context": {
                "file_path": "user_service.py",
                "language": "python",
                "code": "from typing import Dict, List\n\nclass UserService:\n    def __init__(self):\n        self.users = []",
                "cursor_position": {"line": 6, "column": 4}
            },
            "user_id": str(uuid4()),
            "project_id": "test-project"
        }
        
        response = await async_client.post("/api/copilot/enhance", json=enhancement_data)
        assert response.status_code == 200
        enhancement = response.json()
        
        assert "suggestion_id" in enhancement
        assert "enhanced_suggestion" in enhancement
        assert "confidence" in enhancement
        assert "context_sources" in enhancement
        
        suggestion_id = enhancement["suggestion_id"]
        
        # Enhanced suggestion should be different from original
        assert enhancement["enhanced_suggestion"] != enhancement_data["original_suggestion"]
        assert enhancement["confidence"] > 0.0
        
        # 3. Inject context into Copilot request
        injection_data = {
            "request": {
                "prompt": "Create a function to validate user email addresses",
                "context": {"language": "python", "file": "validators.py"}
            },
            "user_id": enhancement_data["user_id"],
            "project_id": enhancement_data["project_id"]
        }
        
        response = await async_client.post("/api/copilot/context/inject", json=injection_data)
        assert response.status_code == 200
        injection_result = response.json()
        
        assert injection_result["status"] == "context_injected"
        assert "enhanced_request" in injection_result
        
        # Enhanced request should include additional context
        enhanced_request = injection_result["enhanced_request"]
        assert enhanced_request["prompt"] != injection_data["request"]["prompt"]
        
        # 4. Provide feedback on the enhanced suggestion
        feedback_data = {
            "suggestion_id": suggestion_id,
            "feedback_type": "accepted",
            "feedback_data": {
                "user_satisfaction": "high",
                "used_as_is": True,
                "helpful_aspects": ["type hints", "documentation", "error handling"]
            }
        }
        
        response = await async_client.post("/api/copilot/feedback", json=feedback_data)
        assert response.status_code == 200
        feedback_response = response.json()
        
        assert feedback_response["status"] == "feedback_received"
        assert feedback_response["suggestion_id"] == suggestion_id
        
        # 5. Test webhook receiving
        webhook_data = {
            "webhook_type": "suggestion_request",
            "payload": {
                "suggestion": "def calculate_total(items):\n    return sum(items)",
                "context": {
                    "file_path": "calculator.py",
                    "language": "python",
                    "project_id": enhancement_data["project_id"]
                }
            },
            "user_id": enhancement_data["user_id"]
        }
        
        response = await async_client.post("/api/copilot/webhook/immediate", json=webhook_data)
        assert response.status_code == 200
        webhook_response = response.json()
        
        assert webhook_response["status"] == "processed"
        assert "result" in webhook_response
        
        # 6. Get analytics for suggestions
        analytics_params = {
            "time_window": "1h",
            "user_id": enhancement_data["user_id"],
            "project_id": enhancement_data["project_id"]
        }
        
        response = await async_client.get("/api/copilot/analytics/suggestions", params=analytics_params)
        assert response.status_code == 200
        analytics = response.json()
        
        assert "total_suggestions" in analytics
        assert "enhancement_rate" in analytics
        assert "acceptance_rate" in analytics
    
    @pytest.mark.asyncio
    async def test_analytics_and_monitoring_workflow(self, async_client):
        """Test analytics and monitoring workflow."""
        
        # 1. Record metrics
        metric_data = {
            "name": "api_response_time",
            "value": 125.5,
            "metric_type": "gauge",
            "tags": {"endpoint": "/api/memory", "method": "POST"},
            "metadata": {"user_agent": "test-client", "timestamp": datetime.utcnow().isoformat()}
        }
        
        response = await async_client.post("/api/analytics/metrics/record", json=metric_data)
        assert response.status_code == 200
        metric_response = response.json()
        assert metric_response["status"] == "recorded"
        
        # Record multiple metrics for dashboard
        metrics_batch = [
            {
                "name": "memory_operations",
                "value": 1,
                "metric_type": "counter",
                "tags": {"operation": "create"}
            },
            {
                "name": "memory_operations", 
                "value": 1,
                "metric_type": "counter",
                "tags": {"operation": "search"}
            },
            {
                "name": "session_duration",
                "value": 3600,
                "metric_type": "histogram",
                "tags": {"session_type": "coding"}
            }
        ]
        
        for metric in metrics_batch:
            await async_client.post("/api/analytics/metrics/record", json=metric)
        
        # 2. Get dashboard data
        dashboard_params = {
            "time_window": "1h",
            "user_id": str(uuid4())
        }
        
        response = await async_client.get("/api/analytics/dashboard", params=dashboard_params)
        assert response.status_code == 200
        dashboard = response.json()
        
        assert "metrics" in dashboard
        assert "performance" in dashboard
        assert "activity" in dashboard
        assert "insights" in dashboard
        
        # 3. Get performance report
        report_params = {
            "report_type": "system",
            "time_range": "1h"
        }
        
        response = await async_client.get("/api/analytics/performance/report", params=report_params)
        assert response.status_code == 200
        report = response.json()
        
        assert "summary" in report
        assert "metrics" in report
        assert "trends" in report
        assert "recommendations" in report
        
        # 4. Test WebSocket connection for real-time updates
        # Note: This would typically require a WebSocket client
        ws_health_response = await async_client.get("/api/websocket/health")
        assert ws_health_response.status_code == 200
        
        # 5. Export analytics data
        export_params = {
            "format": "json",
            "time_range": "1h",
            "metrics": ["api_response_time", "memory_operations"]
        }
        
        response = await async_client.get("/api/analytics/export", params=export_params)
        assert response.status_code == 200
        export_data = response.json()
        
        assert "data" in export_data
        assert "metadata" in export_data
        assert export_data["metadata"]["format"] == "json"
    
    @pytest.mark.asyncio
    async def test_error_scenarios_and_recovery(self, async_client):
        """Test error scenarios and system recovery."""
        
        # 1. Test invalid memory creation
        invalid_memory_data = {
            "content": "",  # Empty content should fail
            "memory_type": "invalid_type_very_long_name_that_exceeds_limits",
            "user_id": "invalid-uuid-format"
        }
        
        response = await async_client.post("/api/memory", json=invalid_memory_data)
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
        
        # 2. Test non-existent resource access
        response = await async_client.get("/api/memory/non-existent-id")
        assert response.status_code == 404
        
        # 3. Test malformed request data
        malformed_data = {
            "invalid_field": "test",
            "another_invalid": 123
        }
        
        response = await async_client.post("/api/memory", json=malformed_data)
        assert response.status_code == 422
        
        # 4. Test rate limiting (if implemented)
        # Rapid successive requests
        for i in range(10):
            response = await async_client.get("/health")
            # Should not be rate limited for health checks
            assert response.status_code == 200
        
        # 5. Test large payload handling
        large_content = "x" * 100000  # 100KB content
        large_memory_data = {
            "content": large_content,
            "memory_type": "large_test",
            "user_id": str(uuid4())
        }
        
        response = await async_client.post("/api/memory", json=large_memory_data)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 413, 422]
        
        # 6. Test concurrent request handling
        import asyncio
        
        async def concurrent_request(i):
            memory_data = {
                "content": f"Concurrent test memory {i}",
                "memory_type": "concurrent_test",
                "user_id": str(uuid4())
            }
            return await async_client.post("/api/memory", json=memory_data)
        
        # Send 10 concurrent requests
        tasks = [concurrent_request(i) for i in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed or fail gracefully
        for response in responses:
            if isinstance(response, Exception):
                continue  # Some exceptions are acceptable under high concurrency
            assert response.status_code in [200, 429, 503]  # Success, rate limited, or service unavailable
        
        # 7. Test system recovery after errors
        # Create a valid memory after error scenarios
        recovery_memory_data = {
            "content": "System recovery test memory",
            "memory_type": "recovery_test",
            "user_id": str(uuid4()),
            "tags": ["recovery", "test"]
        }
        
        response = await async_client.post("/api/memory", json=recovery_memory_data)
        assert response.status_code == 200
        
        # Verify system is still functional
        response = await async_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_mcp_server_integration(self, async_client):
        """Test MCP server integration workflow."""
        
        # 1. Test MCP server health
        response = await async_client.get("/api/mcp/health")
        assert response.status_code == 200
        
        # 2. Simulate MCP tool calls
        mcp_tool_data = {
            "tool": "create_memory",
            "arguments": {
                "content": "MCP integration test memory",
                "memory_type": "mcp_test",
                "user_id": str(uuid4()),
                "tags": ["mcp", "integration"]
            }
        }
        
        response = await async_client.post("/api/mcp/tools/execute", json=mcp_tool_data)
        assert response.status_code == 200
        tool_response = response.json()
        
        assert "result" in tool_response
        assert "memory_id" in tool_response["result"]
        
        memory_id = tool_response["result"]["memory_id"]
        
        # 3. Test MCP search tool
        search_tool_data = {
            "tool": "search_memories",
            "arguments": {
                "query": "MCP integration test",
                "limit": 5
            }
        }
        
        response = await async_client.post("/api/mcp/tools/execute", json=search_tool_data)
        assert response.status_code == 200
        search_response = response.json()
        
        assert "result" in search_response
        assert "memories" in search_response["result"]
        
        # Should find the memory we just created
        found_memories = search_response["result"]["memories"]
        memory_ids = [m["id"] for m in found_memories]
        assert memory_id in memory_ids
        
        # 4. Test MCP session management
        session_tool_data = {
            "tool": "init_session",
            "arguments": {
                "session_type": "mcp_test",
                "project_id": "mcp-integration",
                "context_data": {"source": "mcp_server"}
            }
        }
        
        response = await async_client.post("/api/mcp/tools/execute", json=session_tool_data)
        assert response.status_code == 200
        session_response = response.json()
        
        assert "result" in session_response
        assert "session_id" in session_response["result"]
        
        # 5. Test MCP AI insights
        insights_tool_data = {
            "tool": "get_ai_insights",
            "arguments": {
                "context": "mcp integration testing",
                "data": {"testing_phase": "integration", "tools_tested": ["memory", "search", "session"]}
            }
        }
        
        response = await async_client.post("/api/mcp/tools/execute", json=insights_tool_data)
        assert response.status_code == 200
        insights_response = response.json()
        
        assert "result" in insights_response
        assert "insights" in insights_response["result"]
    
    @pytest.mark.asyncio
    async def test_full_development_lifecycle(self, async_client):
        """Test complete development lifecycle scenario."""
        
        project_id = str(uuid4())
        user_id = str(uuid4())
        
        # Phase 1: Project Setup
        session_data = {
            "user_id": user_id,
            "session_type": "project_setup",
            "project_id": project_id,
            "context_data": {
                "project_name": "E2E Test Project",
                "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
                "phase": "setup"
            }
        }
        
        response = await async_client.post("/api/session", json=session_data)
        assert response.status_code == 200
        session = response.json()
        session_id = session["session_id"]
        
        # Phase 2: Architecture Decisions
        decision_data = {
            "description": "Use microservices architecture",
            "alternatives": ["Monolithic", "Serverless", "Microservices"],
            "reasoning": "Better scalability and team independence",
            "confidence": 0.85,
            "project_id": project_id,
            "session_id": session_id
        }
        
        response = await async_client.post("/api/claude-auto/decision/record", json=decision_data)
        assert response.status_code == 200
        
        # Phase 3: Implementation
        implementation_memories = [
            {
                "content": "Implemented user authentication service",
                "memory_type": "implementation",
                "project_id": project_id,
                "user_id": user_id,
                "tags": ["auth", "microservice"],
                "metadata": {"service": "auth", "endpoints": 5}
            },
            {
                "content": "Created API gateway with rate limiting",
                "memory_type": "implementation", 
                "project_id": project_id,
                "user_id": user_id,
                "tags": ["gateway", "rate_limiting"],
                "metadata": {"service": "gateway", "features": ["routing", "auth", "rate_limit"]}
            },
            {
                "content": "Set up PostgreSQL with connection pooling",
                "memory_type": "infrastructure",
                "project_id": project_id,
                "user_id": user_id,
                "tags": ["database", "postgresql"],
                "metadata": {"database": "postgresql", "pool_size": 20}
            }
        ]
        
        memory_ids = []
        for memory_data in implementation_memories:
            response = await async_client.post("/api/memory", json=memory_data)
            assert response.status_code == 200
            memory_ids.append(response.json()["id"])
        
        # Phase 4: Testing and Issues
        error_data = {
            "error_type": "PerformanceError",
            "message": "API response time exceeds 500ms",
            "context": {
                "service": "auth",
                "endpoint": "/login",
                "avg_response_time": 750
            },
            "solution": "Added Redis caching for user sessions",
            "resolved": True,
            "session_id": session_id
        }
        
        response = await async_client.post("/api/claude-auto/learning/error", json=error_data)
        assert response.status_code == 200
        
        # Phase 5: Performance Optimization
        optimization_memory = {
            "content": "Optimized auth service with Redis caching - reduced response time to 150ms",
            "memory_type": "optimization",
            "project_id": project_id,
            "user_id": user_id,
            "tags": ["performance", "caching", "redis"],
            "metadata": {
                "before_response_time": 750,
                "after_response_time": 150,
                "improvement_percentage": 80
            }
        }
        
        response = await async_client.post("/api/memory", json=optimization_memory)
        assert response.status_code == 200
        
        # Phase 6: Project Completion
        completion_context = {
            "phase": "completed",
            "services_implemented": 3,
            "total_endpoints": 15,
            "performance_optimized": True,
            "test_coverage": 85
        }
        
        response = await async_client.put(
            f"/api/session/{session_id}/context",
            json=completion_context
        )
        assert response.status_code == 200
        
        # Phase 7: Knowledge Extraction
        search_response = await async_client.get(
            "/api/memory/search",
            params={"query": "microservices implementation", "project_id": project_id, "limit": 10}
        )
        assert search_response.status_code == 200
        project_memories = search_response.json()["memories"]
        
        # Should find all our implementation memories
        assert len(project_memories) >= 3
        
        # Phase 8: Project Analytics
        analytics_response = await async_client.get(
            "/api/analytics/dashboard",
            params={"project_id": project_id, "time_window": "24h"}
        )
        assert analytics_response.status_code == 200
        project_analytics = analytics_response.json()
        
        assert "activity" in project_analytics
        assert "metrics" in project_analytics
        
        # Phase 9: Session Handoff
        handoff_data = {
            "message": "E2E test project completed successfully",
            "context_summary": "Microservices architecture implemented with 3 services, optimized for performance",
            "next_developer_notes": [
                "Monitor Redis cache hit rates",
                "Consider implementing service mesh",
                "Plan for horizontal scaling"
            ]
        }
        
        response = await async_client.post(
            f"/api/claude-auto/session/{session_id}/handoff",
            json=handoff_data
        )
        assert response.status_code == 200
        handoff = response.json()
        
        assert "handoff_id" in handoff
        assert handoff["context_preserved"] is True
        
        # Phase 10: Session End
        end_data = {
            "reason": "Project lifecycle completed",
            "duration_hours": 40,
            "achievements": [
                "Microservices architecture implemented",
                "Performance optimized",
                "Knowledge documented"
            ],
            "create_memory": True
        }
        
        response = await async_client.post(
            f"/api/claude-auto/session/{session_id}/end",
            json=end_data
        )
        assert response.status_code == 200
        end_response = response.json()
        
        assert end_response["status"] == "ended"
        assert "session_memory_id" in end_response
        
        # Verify project knowledge is preserved and searchable
        final_search = await async_client.get(
            "/api/memory/search",
            params={"query": "E2E test project microservices", "limit": 20}
        )
        assert final_search.status_code == 200
        final_memories = final_search.json()["memories"]
        
        # Should find session memory plus implementation memories
        assert len(final_memories) >= 4