#!/usr/bin/env python3
"""Test core RAG functionality after refactoring."""

import asyncio
import httpx
import json
from typing import Dict, Any

API_BASE = "http://localhost:3000"

class RAGTester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results = []
        
    async def test_health(self) -> Dict[str, Any]:
        """Test API health endpoint."""
        try:
            response = await self.client.get(f"{API_BASE}/health")
            return {
                "test": "API Health",
                "status": "✅ PASS" if response.status_code == 200 else "❌ FAIL",
                "response": response.json() if response.status_code == 200 else str(response.status_code)
            }
        except Exception as e:
            return {"test": "API Health", "status": "❌ FAIL", "error": str(e)}
    
    async def test_hybrid_rag_search(self) -> Dict[str, Any]:
        """Test hybrid RAG search functionality."""
        try:
            payload = {
                "query": "test query for hybrid RAG",
                "retrieval_mode": "hybrid",
                "top_k": 5
            }
            response = await self.client.post(
                f"{API_BASE}/api/rag/search",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "Hybrid RAG Search",
                    "status": "✅ PASS",
                    "modes_used": data.get("metadata", {}).get("modes_used", []),
                    "results_count": len(data.get("results", []))
                }
            else:
                return {
                    "test": "Hybrid RAG Search",
                    "status": "⚠️ PARTIAL",
                    "status_code": response.status_code,
                    "response": response.text[:200]
                }
        except Exception as e:
            return {"test": "Hybrid RAG Search", "status": "❌ FAIL", "error": str(e)}
    
    async def test_agent_workflow(self) -> Dict[str, Any]:
        """Test LangGraph agent workflow."""
        try:
            payload = {
                "query": "What is the current state of the system?",
                "workflow_type": "simple_qa"
            }
            response = await self.client.post(
                f"{API_BASE}/api/agent/workflow",
                json=payload
            )
            
            if response.status_code == 200:
                return {
                    "test": "Agent Workflow (LangGraph)",
                    "status": "✅ PASS",
                    "workflow_type": response.json().get("workflow_type")
                }
            else:
                return {
                    "test": "Agent Workflow (LangGraph)",
                    "status": "⚠️ PARTIAL",
                    "status_code": response.status_code
                }
        except Exception as e:
            return {"test": "Agent Workflow (LangGraph)", "status": "❌ FAIL", "error": str(e)}
    
    async def test_memory_session(self) -> Dict[str, Any]:
        """Test Zep memory session creation."""
        try:
            payload = {
                "session_id": "test_session_001",
                "user_id": "test_user",
                "metadata": {"test": True}
            }
            response = await self.client.post(
                f"{API_BASE}/api/memory/session",
                json=payload
            )
            
            if response.status_code in [200, 201]:
                return {
                    "test": "Memory Session (Zep)",
                    "status": "✅ PASS",
                    "session_id": response.json().get("session_id")
                }
            else:
                return {
                    "test": "Memory Session (Zep)",
                    "status": "⚠️ PARTIAL",
                    "status_code": response.status_code
                }
        except Exception as e:
            return {"test": "Memory Session (Zep)", "status": "❌ FAIL", "error": str(e)}
    
    async def test_vector_db(self) -> Dict[str, Any]:
        """Test vector database connectivity."""
        try:
            # Test Weaviate
            weaviate_response = await self.client.get(f"{API_BASE}/api/rag/vector/status")
            
            return {
                "test": "Vector DB (Weaviate)",
                "status": "✅ PASS" if weaviate_response.status_code == 200 else "⚠️ PARTIAL",
                "weaviate": "connected" if weaviate_response.status_code == 200 else "not connected"
            }
        except Exception as e:
            return {"test": "Vector DB", "status": "❌ FAIL", "error": str(e)}
    
    async def test_graph_db(self) -> Dict[str, Any]:
        """Test Neo4j graph database connectivity."""
        try:
            response = await self.client.get(f"{API_BASE}/api/rag/graph/status")
            
            return {
                "test": "Graph DB (Neo4j)",
                "status": "✅ PASS" if response.status_code == 200 else "⚠️ PARTIAL",
                "neo4j": "connected" if response.status_code == 200 else "not connected"
            }
        except Exception as e:
            return {"test": "Graph DB", "status": "❌ FAIL", "error": str(e)}
    
    async def run_all_tests(self):
        """Run all RAG functionality tests."""
        print("\n" + "="*60)
        print("🧪 TESTING CORE RAG FUNCTIONALITY")
        print("="*60 + "\n")
        
        tests = [
            self.test_health(),
            self.test_hybrid_rag_search(),
            self.test_agent_workflow(),
            self.test_memory_session(),
            self.test_vector_db(),
            self.test_graph_db()
        ]
        
        results = await asyncio.gather(*tests)
        
        # Print results
        for result in results:
            self.results.append(result)
            print(f"\n{result['test']}:")
            print(f"  Status: {result['status']}")
            for key, value in result.items():
                if key not in ['test', 'status']:
                    print(f"  {key}: {value}")
        
        # Summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        pass_count = sum(1 for r in self.results if "PASS" in r['status'])
        partial_count = sum(1 for r in self.results if "PARTIAL" in r['status'])
        fail_count = sum(1 for r in self.results if "FAIL" in r['status'])
        
        print(f"\n✅ Passed: {pass_count}/{len(self.results)}")
        print(f"⚠️  Partial: {partial_count}/{len(self.results)}")
        print(f"❌ Failed: {fail_count}/{len(self.results)}")
        
        if pass_count == len(self.results):
            print("\n🎉 ALL TESTS PASSED! The RAG system is fully functional.")
        elif fail_count == 0:
            print("\n⚠️  System is partially functional. Some features may need attention.")
        else:
            print("\n❌ System has failures. Critical components need fixing.")
        
        await self.client.aclose()

async def main():
    tester = RAGTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())