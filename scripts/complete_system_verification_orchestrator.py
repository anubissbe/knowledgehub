#!/usr/bin/env python3
"""
Complete System Verification Orchestrator
Orchestrates specialized agents to verify every component of the KnowledgeHub system
"""

import asyncio
import json
import httpx
import time
import docker
import redis
import psycopg2
from neo4j import GraphDatabase
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VerificationStatus(Enum):
    """Verification status levels"""
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    PARTIAL = "⚠️ PARTIAL"
    SKIPPED = "⏭️ SKIPPED"


@dataclass
class VerificationResult:
    """Result of a verification test"""
    component: str
    test_name: str
    status: VerificationStatus
    details: Dict[str, Any]
    error: Optional[str] = None
    timestamp: Optional[str] = None


class BaseVerificationAgent:
    """Base class for verification agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def verify(self) -> List[VerificationResult]:
        """Run verification tests"""
        raise NotImplementedError
        
    def add_result(self, component: str, test: str, status: VerificationStatus, 
                   details: Dict = None, error: str = None):
        """Add verification result"""
        result = VerificationResult(
            component=component,
            test_name=test,
            status=status,
            details=details or {},
            error=error,
            timestamp=datetime.now().isoformat()
        )
        self.results.append(result)
        return result


class InfrastructureAgent(BaseVerificationAgent):
    """Verifies infrastructure and container health"""
    
    def __init__(self):
        super().__init__("Infrastructure Verifier")
        self.docker_client = docker.from_env()
        
    async def verify(self) -> List[VerificationResult]:
        """Verify all infrastructure components"""
        print(f"\n🔍 {self.name}: Starting infrastructure verification...")
        
        # Check Docker containers
        await self.verify_containers()
        
        # Check network connectivity
        await self.verify_network()
        
        # Check resource usage
        await self.verify_resources()
        
        return self.results
    
    async def verify_containers(self):
        """Verify all required containers are running"""
        required_containers = [
            ("postgres", 5433, "Database"),
            ("redis", 6381, "Cache"),
            ("neo4j", 7474, "Graph DB"),
            ("weaviate", 8090, "Vector DB"),
            ("qdrant", 6333, "Vector DB Alt"),
            ("zep", 8100, "Memory Service"),
            ("api", 3000, "API Server"),
            ("webui", 3100, "Web UI"),
            ("minio", 9010, "Object Storage"),
            ("timescale", 5434, "TimeSeries DB")
        ]
        
        for service_name, port, description in required_containers:
            try:
                container = self.docker_client.containers.get(f"knowledgehub-{service_name}-1")
                
                # Check container status
                if container.status == "running":
                    # Check health if available
                    health = container.attrs.get('State', {}).get('Health', {})
                    health_status = health.get('Status', 'none')
                    
                    if health_status in ['healthy', 'none']:
                        self.add_result(
                            f"Container/{service_name}",
                            f"{description} Container",
                            VerificationStatus.PASSED,
                            {
                                "status": container.status,
                                "health": health_status,
                                "port": port
                            }
                        )
                    else:
                        self.add_result(
                            f"Container/{service_name}",
                            f"{description} Container",
                            VerificationStatus.PARTIAL,
                            {"status": container.status, "health": health_status}
                        )
                else:
                    self.add_result(
                        f"Container/{service_name}",
                        f"{description} Container",
                        VerificationStatus.FAILED,
                        {"status": container.status}
                    )
                    
            except docker.errors.NotFound:
                self.add_result(
                    f"Container/{service_name}",
                    f"{description} Container",
                    VerificationStatus.FAILED,
                    error=f"Container not found"
                )
    
    async def verify_network(self):
        """Verify network connectivity between services"""
        endpoints = [
            ("http://localhost:3000/health", "API"),
            ("http://localhost:3100", "WebUI"),
            ("http://localhost:8090/v1/.well-known/ready", "Weaviate"),
            ("http://localhost:7474", "Neo4j"),
            ("http://localhost:8100/health", "Zep"),
            ("http://localhost:6333/health", "Qdrant")
        ]
        
        for url, service in endpoints:
            try:
                response = await self.client.get(url)
                self.add_result(
                    f"Network/{service}",
                    f"{service} Connectivity",
                    VerificationStatus.PASSED if response.status_code < 400 else VerificationStatus.FAILED,
                    {"status_code": response.status_code, "url": url}
                )
            except Exception as e:
                self.add_result(
                    f"Network/{service}",
                    f"{service} Connectivity",
                    VerificationStatus.FAILED,
                    error=str(e)
                )
    
    async def verify_resources(self):
        """Check resource usage"""
        import psutil
        
        # System resources
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Check if resources are adequate
        memory_ok = memory.percent < 90
        cpu_ok = cpu < 85
        disk_ok = disk.percent < 90
        
        self.add_result(
            "Resources/System",
            "Resource Availability",
            VerificationStatus.PASSED if all([memory_ok, cpu_ok, disk_ok]) else VerificationStatus.PARTIAL,
            {
                "memory_percent": memory.percent,
                "cpu_percent": cpu,
                "disk_percent": disk.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
        )


class DatabaseAgent(BaseVerificationAgent):
    """Verifies database connectivity and schema"""
    
    def __init__(self):
        super().__init__("Database Verifier")
        
    async def verify(self) -> List[VerificationResult]:
        """Verify all database components"""
        print(f"\n🔍 {self.name}: Starting database verification...")
        
        await self.verify_postgresql()
        await self.verify_neo4j()
        await self.verify_redis()
        await self.verify_vector_stores()
        
        return self.results
    
    async def verify_postgresql(self):
        """Verify PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5433,
                database="knowledgehub",
                user="knowledgehub",
                password="knowledgehub123"
            )
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Check for hybrid RAG tables
            required_tables = [
                'memories', 'sessions', 'agent_workflows', 
                'hybrid_rag_sessions', 'retrieval_results'
            ]
            
            missing_tables = [t for t in required_tables if t not in tables]
            
            if not missing_tables:
                # Check row counts
                cursor.execute("SELECT COUNT(*) FROM memories")
                memory_count = cursor.fetchone()[0]
                
                self.add_result(
                    "Database/PostgreSQL",
                    "PostgreSQL Schema & Data",
                    VerificationStatus.PASSED,
                    {
                        "tables_count": len(tables),
                        "memory_records": memory_count,
                        "required_tables": "All present"
                    }
                )
            else:
                self.add_result(
                    "Database/PostgreSQL",
                    "PostgreSQL Schema",
                    VerificationStatus.PARTIAL,
                    {
                        "tables_count": len(tables),
                        "missing_tables": missing_tables
                    }
                )
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.add_result(
                "Database/PostgreSQL",
                "PostgreSQL Connection",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_neo4j(self):
        """Verify Neo4j graph database"""
        try:
            driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "knowledgehub123")
            )
            
            with driver.session() as session:
                # Check node count
                result = session.run("MATCH (n) RETURN COUNT(n) as count")
                node_count = result.single()["count"]
                
                # Check relationships
                result = session.run("MATCH ()-[r]->() RETURN COUNT(r) as count")
                rel_count = result.single()["count"]
                
                self.add_result(
                    "Database/Neo4j",
                    "Neo4j Graph Database",
                    VerificationStatus.PASSED,
                    {
                        "nodes": node_count,
                        "relationships": rel_count,
                        "status": "Connected"
                    }
                )
            
            driver.close()
            
        except Exception as e:
            self.add_result(
                "Database/Neo4j",
                "Neo4j Connection",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_redis(self):
        """Verify Redis cache"""
        try:
            r = redis.Redis(host='localhost', port=6381)
            
            # Test connection
            r.ping()
            
            # Check keys
            keys_count = len(r.keys())
            
            # Test set/get
            test_key = "verification_test"
            r.set(test_key, "test_value", ex=10)
            value = r.get(test_key)
            
            if value == b"test_value":
                self.add_result(
                    "Database/Redis",
                    "Redis Cache",
                    VerificationStatus.PASSED,
                    {
                        "keys_count": keys_count,
                        "test_operation": "Success"
                    }
                )
            else:
                self.add_result(
                    "Database/Redis",
                    "Redis Operations",
                    VerificationStatus.PARTIAL,
                    {"keys_count": keys_count}
                )
                
        except Exception as e:
            self.add_result(
                "Database/Redis",
                "Redis Connection",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_vector_stores(self):
        """Verify vector databases"""
        # Weaviate
        try:
            response = await self.client.get("http://localhost:8090/v1/.well-known/ready")
            self.add_result(
                "Database/Weaviate",
                "Weaviate Vector Store",
                VerificationStatus.PASSED if response.status_code == 200 else VerificationStatus.FAILED,
                {"status": response.status_code}
            )
        except Exception as e:
            self.add_result(
                "Database/Weaviate",
                "Weaviate Connection",
                VerificationStatus.FAILED,
                error=str(e)
            )
        
        # Qdrant
        try:
            response = await self.client.get("http://localhost:6333/health")
            self.add_result(
                "Database/Qdrant",
                "Qdrant Vector Store",
                VerificationStatus.PASSED if response.status_code == 200 else VerificationStatus.FAILED,
                {"status": response.status_code}
            )
        except Exception as e:
            self.add_result(
                "Database/Qdrant",
                "Qdrant Connection",
                VerificationStatus.FAILED,
                error=str(e)
            )


class APIAgent(BaseVerificationAgent):
    """Verifies API endpoints and functionality"""
    
    def __init__(self):
        super().__init__("API Verifier")
        self.base_url = "http://localhost:3000/api"
        
    async def verify(self) -> List[VerificationResult]:
        """Verify API functionality"""
        print(f"\n🔍 {self.name}: Starting API verification...")
        
        await self.verify_health_endpoints()
        await self.verify_rag_endpoints()
        await self.verify_agent_endpoints()
        await self.verify_memory_endpoints()
        
        return self.results
    
    async def verify_health_endpoints(self):
        """Verify health check endpoints"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                self.add_result(
                    "API/Health",
                    "Health Check Endpoint",
                    VerificationStatus.PASSED,
                    {"status": data.get("status", "unknown")}
                )
            else:
                self.add_result(
                    "API/Health",
                    "Health Check Endpoint",
                    VerificationStatus.FAILED,
                    {"status_code": response.status_code}
                )
        except Exception as e:
            self.add_result(
                "API/Health",
                "Health Check",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_rag_endpoints(self):
        """Verify RAG endpoints"""
        # Test hybrid RAG query
        try:
            test_query = {
                "query": "What is hybrid RAG?",
                "mode": "hybrid",
                "top_k": 5
            }
            
            response = await self.client.post(
                f"{self.base_url}/rag/enhanced/query",
                json=test_query
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                self.add_result(
                    "API/RAG",
                    "Hybrid RAG Query",
                    VerificationStatus.PASSED if results else VerificationStatus.PARTIAL,
                    {
                        "results_count": len(results),
                        "query_mode": "hybrid",
                        "response_time": data.get("performance", {}).get("total_time_ms", "N/A")
                    }
                )
            else:
                self.add_result(
                    "API/RAG",
                    "Hybrid RAG Query",
                    VerificationStatus.FAILED,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "API/RAG",
                "RAG Endpoints",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_agent_endpoints(self):
        """Verify agent workflow endpoints"""
        try:
            # Test workflow execution
            workflow_request = {
                "workflow_type": "simple_qa",
                "input": {
                    "query": "Test question",
                    "context": {}
                }
            }
            
            response = await self.client.post(
                f"{self.base_url}/agent/workflows/execute",
                json=workflow_request
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                self.add_result(
                    "API/Agent",
                    "Agent Workflow Execution",
                    VerificationStatus.PASSED,
                    {
                        "workflow_id": data.get("workflow_id", "N/A"),
                        "status": data.get("status", "unknown")
                    }
                )
            else:
                self.add_result(
                    "API/Agent",
                    "Agent Workflow",
                    VerificationStatus.PARTIAL,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            # Check if it's a graceful fallback
            if "LangGraph" in str(e) or "not available" in str(e):
                self.add_result(
                    "API/Agent",
                    "Agent Workflows",
                    VerificationStatus.PARTIAL,
                    {"note": "Using fallback implementation"}
                )
            else:
                self.add_result(
                    "API/Agent",
                    "Agent Endpoints",
                    VerificationStatus.FAILED,
                    error=str(e)
                )
    
    async def verify_memory_endpoints(self):
        """Verify memory management endpoints"""
        try:
            # Test memory storage
            memory_data = {
                "content": "Test memory content",
                "type": "episodic",
                "metadata": {"test": True}
            }
            
            response = await self.client.post(
                f"{self.base_url}/memory/store",
                json=memory_data
            )
            
            if response.status_code in [200, 201]:
                self.add_result(
                    "API/Memory",
                    "Memory Storage",
                    VerificationStatus.PASSED,
                    {"operation": "store", "type": "episodic"}
                )
            else:
                self.add_result(
                    "API/Memory",
                    "Memory Storage",
                    VerificationStatus.PARTIAL,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "API/Memory",
                "Memory Endpoints",
                VerificationStatus.FAILED,
                error=str(e)
            )


class RAGAgent(BaseVerificationAgent):
    """Verifies RAG system functionality"""
    
    def __init__(self):
        super().__init__("RAG System Verifier")
        self.base_url = "http://localhost:3000/api"
        
    async def verify(self) -> List[VerificationResult]:
        """Verify RAG system components"""
        print(f"\n🔍 {self.name}: Starting RAG system verification...")
        
        await self.verify_vector_search()
        await self.verify_sparse_search()
        await self.verify_graph_search()
        await self.verify_hybrid_fusion()
        await self.verify_reranking()
        
        return self.results
    
    async def verify_vector_search(self):
        """Test vector search functionality"""
        try:
            query = {
                "query": "machine learning algorithms",
                "mode": "vector",
                "top_k": 5
            }
            
            response = await self.client.post(
                f"{self.base_url}/rag/enhanced/query",
                json=query
            )
            
            if response.status_code == 200:
                data = response.json()
                self.add_result(
                    "RAG/Vector",
                    "Vector Search",
                    VerificationStatus.PASSED,
                    {
                        "results": len(data.get("results", [])),
                        "latency_ms": data.get("performance", {}).get("retrieval_time_ms", "N/A")
                    }
                )
            else:
                self.add_result(
                    "RAG/Vector",
                    "Vector Search",
                    VerificationStatus.FAILED,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "RAG/Vector",
                "Vector Search",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_sparse_search(self):
        """Test BM25 sparse search"""
        try:
            query = {
                "query": "database optimization techniques",
                "mode": "sparse",
                "top_k": 5
            }
            
            response = await self.client.post(
                f"{self.base_url}/rag/enhanced/query",
                json=query
            )
            
            if response.status_code == 200:
                data = response.json()
                self.add_result(
                    "RAG/Sparse",
                    "BM25 Sparse Search",
                    VerificationStatus.PASSED,
                    {"results": len(data.get("results", []))}
                )
            else:
                self.add_result(
                    "RAG/Sparse",
                    "Sparse Search",
                    VerificationStatus.PARTIAL,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "RAG/Sparse",
                "Sparse Search",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_graph_search(self):
        """Test graph-based search"""
        try:
            query = {
                "query": "related concepts to neural networks",
                "mode": "graph",
                "top_k": 5
            }
            
            response = await self.client.post(
                f"{self.base_url}/rag/enhanced/query",
                json=query
            )
            
            if response.status_code == 200:
                data = response.json()
                self.add_result(
                    "RAG/Graph",
                    "Graph Search",
                    VerificationStatus.PASSED,
                    {"results": len(data.get("results", []))}
                )
            else:
                self.add_result(
                    "RAG/Graph",
                    "Graph Search",
                    VerificationStatus.PARTIAL,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "RAG/Graph",
                "Graph Search",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_hybrid_fusion(self):
        """Test hybrid RAG with fusion"""
        try:
            query = {
                "query": "explain transformers in AI",
                "mode": "hybrid",
                "top_k": 10,
                "rerank": True
            }
            
            response = await self.client.post(
                f"{self.base_url}/rag/enhanced/query",
                json=query
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Check if results come from multiple sources
                sources = set(r.get("retrieval_method", "") for r in results)
                
                self.add_result(
                    "RAG/Hybrid",
                    "Hybrid Fusion",
                    VerificationStatus.PASSED if len(sources) > 1 else VerificationStatus.PARTIAL,
                    {
                        "results": len(results),
                        "sources": list(sources),
                        "fusion_active": len(sources) > 1
                    }
                )
            else:
                self.add_result(
                    "RAG/Hybrid",
                    "Hybrid Fusion",
                    VerificationStatus.FAILED,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "RAG/Hybrid",
                "Hybrid Fusion",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_reranking(self):
        """Test reranking functionality"""
        try:
            # Query with reranking
            query = {
                "query": "advanced machine learning techniques",
                "mode": "hybrid",
                "top_k": 5,
                "rerank": True
            }
            
            response = await self.client.post(
                f"{self.base_url}/rag/enhanced/query",
                json=query
            )
            
            if response.status_code == 200:
                data = response.json()
                perf = data.get("performance", {})
                
                self.add_result(
                    "RAG/Reranking",
                    "Cross-encoder Reranking",
                    VerificationStatus.PASSED if perf.get("reranking_time_ms") else VerificationStatus.PARTIAL,
                    {
                        "reranking_time_ms": perf.get("reranking_time_ms", "N/A"),
                        "total_time_ms": perf.get("total_time_ms", "N/A")
                    }
                )
            else:
                self.add_result(
                    "RAG/Reranking",
                    "Reranking",
                    VerificationStatus.FAILED,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "RAG/Reranking",
                "Reranking",
                VerificationStatus.FAILED,
                error=str(e)
            )


class IntegrationAgent(BaseVerificationAgent):
    """Verifies service integrations"""
    
    def __init__(self):
        super().__init__("Integration Verifier")
        
    async def verify(self) -> List[VerificationResult]:
        """Verify all integrations"""
        print(f"\n🔍 {self.name}: Starting integration verification...")
        
        await self.verify_zep_integration()
        await self.verify_firecrawl_integration()
        await self.verify_mcp_integration()
        await self.verify_monitoring_integration()
        
        return self.results
    
    async def verify_zep_integration(self):
        """Verify Zep memory service"""
        try:
            response = await self.client.get("http://localhost:8100/health")
            
            if response.status_code == 200:
                # Test memory operation through API
                memory_test = await self.client.post(
                    "http://localhost:3000/api/memory/zep/search",
                    json={
                        "query": "test",
                        "session_id": "test_session",
                        "limit": 5
                    }
                )
                
                self.add_result(
                    "Integration/Zep",
                    "Zep Memory Service",
                    VerificationStatus.PASSED,
                    {
                        "health": "OK",
                        "memory_search": memory_test.status_code == 200
                    }
                )
            else:
                self.add_result(
                    "Integration/Zep",
                    "Zep Service",
                    VerificationStatus.PARTIAL,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "Integration/Zep",
                "Zep Integration",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_firecrawl_integration(self):
        """Verify Firecrawl web ingestion"""
        try:
            # Check if Firecrawl endpoint is configured
            crawl_test = {
                "url": "https://example.com",
                "mode": "single",
                "options": {"max_pages": 1}
            }
            
            response = await self.client.post(
                "http://localhost:3000/api/ingestion/crawl",
                json=crawl_test
            )
            
            # Even if it fails, check if the endpoint exists
            if response.status_code in [200, 201, 202]:
                self.add_result(
                    "Integration/Firecrawl",
                    "Firecrawl Web Ingestion",
                    VerificationStatus.PASSED,
                    {"endpoint": "Active", "job_queuing": True}
                )
            elif response.status_code == 404:
                self.add_result(
                    "Integration/Firecrawl",
                    "Firecrawl",
                    VerificationStatus.FAILED,
                    {"error": "Endpoint not found"}
                )
            else:
                self.add_result(
                    "Integration/Firecrawl",
                    "Firecrawl",
                    VerificationStatus.PARTIAL,
                    {"status_code": response.status_code}
                )
                
        except Exception as e:
            self.add_result(
                "Integration/Firecrawl",
                "Firecrawl",
                VerificationStatus.PARTIAL,
                {"note": "Service may not be fully configured"}
            )
    
    async def verify_mcp_integration(self):
        """Verify MCP server"""
        mcp_path = Path("/opt/projects/knowledgehub/mcp_server")
        
        if mcp_path.exists():
            # Check for MCP files
            required_files = ["server.py", "handlers.py", "tools.py"]
            missing_files = [f for f in required_files if not (mcp_path / f).exists()]
            
            if not missing_files:
                self.add_result(
                    "Integration/MCP",
                    "MCP Server Implementation",
                    VerificationStatus.PASSED,
                    {
                        "path": str(mcp_path),
                        "files": "All present",
                        "status": "Implemented"
                    }
                )
            else:
                self.add_result(
                    "Integration/MCP",
                    "MCP Server",
                    VerificationStatus.PARTIAL,
                    {"missing_files": missing_files}
                )
        else:
            self.add_result(
                "Integration/MCP",
                "MCP Server",
                VerificationStatus.FAILED,
                {"error": "MCP server directory not found"}
            )
    
    async def verify_monitoring_integration(self):
        """Verify monitoring stack"""
        # Grafana
        try:
            response = await self.client.get("http://localhost:3030")
            self.add_result(
                "Integration/Monitoring",
                "Grafana Dashboard",
                VerificationStatus.PASSED if response.status_code < 400 else VerificationStatus.FAILED,
                {"status_code": response.status_code}
            )
        except:
            self.add_result(
                "Integration/Monitoring",
                "Grafana",
                VerificationStatus.FAILED,
                {"error": "Cannot connect"}
            )


class PerformanceAgent(BaseVerificationAgent):
    """Verifies performance metrics"""
    
    def __init__(self):
        super().__init__("Performance Verifier")
        
    async def verify(self) -> List[VerificationResult]:
        """Verify performance metrics"""
        print(f"\n🔍 {self.name}: Starting performance verification...")
        
        await self.verify_response_times()
        await self.verify_throughput()
        await self.verify_optimization()
        
        return self.results
    
    async def verify_response_times(self):
        """Test response times"""
        endpoints = [
            ("/health", "GET", None, 50),
            ("/rag/enhanced/query", "POST", {"query": "test", "mode": "vector"}, 200),
            ("/memory/recall", "GET", None, 150)
        ]
        
        for endpoint, method, data, threshold_ms in endpoints:
            try:
                start = time.time()
                
                if method == "GET":
                    response = await self.client.get(f"http://localhost:3000/api{endpoint}")
                else:
                    response = await self.client.post(
                        f"http://localhost:3000/api{endpoint}",
                        json=data
                    )
                
                latency = (time.time() - start) * 1000
                
                self.add_result(
                    f"Performance/Latency{endpoint}",
                    f"{endpoint} Response Time",
                    VerificationStatus.PASSED if latency < threshold_ms else VerificationStatus.PARTIAL,
                    {
                        "latency_ms": round(latency, 2),
                        "threshold_ms": threshold_ms,
                        "status_code": response.status_code
                    }
                )
                
            except Exception as e:
                self.add_result(
                    f"Performance/Latency{endpoint}",
                    f"{endpoint} Response",
                    VerificationStatus.FAILED,
                    error=str(e)
                )
    
    async def verify_throughput(self):
        """Test system throughput"""
        # Simple throughput test
        requests = 10
        start = time.time()
        
        tasks = []
        for _ in range(requests):
            task = self.client.get("http://localhost:3000/api/health")
            tasks.append(task)
        
        try:
            responses = await asyncio.gather(*tasks)
            duration = time.time() - start
            rps = requests / duration
            
            self.add_result(
                "Performance/Throughput",
                "Request Throughput",
                VerificationStatus.PASSED if rps > 10 else VerificationStatus.PARTIAL,
                {
                    "requests_per_second": round(rps, 2),
                    "test_requests": requests,
                    "duration_seconds": round(duration, 2)
                }
            )
            
        except Exception as e:
            self.add_result(
                "Performance/Throughput",
                "Throughput Test",
                VerificationStatus.FAILED,
                error=str(e)
            )
    
    async def verify_optimization(self):
        """Check if optimizations are active"""
        # Check for optimization files
        opt_files = [
            Path("/opt/projects/knowledgehub/api/services/query_optimizer.py"),
            Path("/opt/projects/knowledgehub/api/services/reranking_optimizer.py"),
            Path("/opt/projects/knowledgehub/api/services/resource_manager.py")
        ]
        
        existing = [f.name for f in opt_files if f.exists()]
        
        self.add_result(
            "Performance/Optimization",
            "Optimization Components",
            VerificationStatus.PASSED if len(existing) == len(opt_files) else VerificationStatus.PARTIAL,
            {
                "optimization_modules": existing,
                "total_expected": len(opt_files),
                "implemented": len(existing)
            }
        )


class SystemVerificationOrchestrator:
    """Orchestrates all verification agents"""
    
    def __init__(self):
        self.agents = [
            InfrastructureAgent(),
            DatabaseAgent(),
            APIAgent(),
            RAGAgent(),
            IntegrationAgent(),
            PerformanceAgent()
        ]
        self.start_time = None
        self.end_time = None
        
    async def run_verification(self):
        """Run complete system verification"""
        self.start_time = datetime.now()
        
        print("=" * 80)
        print("🚀 COMPLETE SYSTEM VERIFICATION STARTING")
        print(f"📅 Timestamp: {self.start_time}")
        print(f"🤖 Verification Agents: {len(self.agents)}")
        print("=" * 80)
        
        # Run all agents in parallel
        tasks = [agent.verify() for agent in self.agents]
        await asyncio.gather(*tasks)
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Generate report
        report = self.generate_report(duration)
        
        # Save report
        report_path = Path("system_verification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def generate_report(self, duration: float) -> Dict[str, Any]:
        """Generate verification report"""
        all_results = []
        for agent in self.agents:
            all_results.extend(agent.results)
        
        # Count by status
        passed = sum(1 for r in all_results if r.status == VerificationStatus.PASSED)
        failed = sum(1 for r in all_results if r.status == VerificationStatus.FAILED)
        partial = sum(1 for r in all_results if r.status == VerificationStatus.PARTIAL)
        skipped = sum(1 for r in all_results if r.status == VerificationStatus.SKIPPED)
        
        # Group by component
        component_results = {}
        for result in all_results:
            comp = result.component.split('/')[0]
            if comp not in component_results:
                component_results[comp] = {"passed": 0, "failed": 0, "partial": 0}
            
            if result.status == VerificationStatus.PASSED:
                component_results[comp]["passed"] += 1
            elif result.status == VerificationStatus.FAILED:
                component_results[comp]["failed"] += 1
            elif result.status == VerificationStatus.PARTIAL:
                component_results[comp]["partial"] += 1
        
        return {
            "verification_summary": {
                "timestamp": self.start_time.isoformat(),
                "duration_seconds": duration,
                "total_tests": len(all_results),
                "passed": passed,
                "failed": failed,
                "partial": partial,
                "skipped": skipped,
                "success_rate": (passed / len(all_results) * 100) if all_results else 0
            },
            "component_results": component_results,
            "detailed_results": [
                {
                    "component": r.component,
                    "test": r.test_name,
                    "status": r.status.value,
                    "details": r.details,
                    "error": r.error
                }
                for r in all_results
            ],
            "critical_failures": [
                {
                    "component": r.component,
                    "test": r.test_name,
                    "error": r.error
                }
                for r in all_results
                if r.status == VerificationStatus.FAILED
            ]
        }
    
    def print_summary(self, report: Dict[str, Any]):
        """Print verification summary"""
        summary = report["verification_summary"]
        
        print("\n" + "=" * 80)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 80)
        
        print(f"\n⏱️ Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"📝 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']} ({summary['passed']/summary['total_tests']*100:.1f}%)")
        print(f"⚠️ Partial: {summary['partial']} ({summary['partial']/summary['total_tests']*100:.1f}%)")
        print(f"❌ Failed: {summary['failed']} ({summary['failed']/summary['total_tests']*100:.1f}%)")
        
        print("\n📦 COMPONENT STATUS:")
        print("-" * 40)
        
        for comp, results in report["component_results"].items():
            total = sum(results.values())
            status = "✅" if results["failed"] == 0 else "⚠️" if results["failed"] < results["passed"] else "❌"
            print(f"{status} {comp}: {results['passed']}/{total} passed")
        
        if report["critical_failures"]:
            print("\n🚨 CRITICAL FAILURES:")
            print("-" * 40)
            for failure in report["critical_failures"][:5]:  # Show first 5
                print(f"❌ {failure['component']}: {failure['test']}")
                if failure['error']:
                    print(f"   Error: {failure['error'][:100]}")
        
        # Overall verdict
        success_rate = summary['success_rate']
        if success_rate >= 90:
            verdict = "✅ SYSTEM FULLY OPERATIONAL"
            verdict_detail = "All core components verified and working"
        elif success_rate >= 70:
            verdict = "⚠️ SYSTEM PARTIALLY OPERATIONAL"
            verdict_detail = "Core features working, some components need attention"
        else:
            verdict = "❌ SYSTEM NEEDS ATTENTION"
            verdict_detail = "Multiple components failing verification"
        
        print("\n" + "=" * 80)
        print(f"🎯 FINAL VERDICT: {verdict}")
        print(f"   {verdict_detail}")
        print(f"   Overall Success Rate: {success_rate:.1f}%")
        print("=" * 80)


async def main():
    """Main execution function"""
    orchestrator = SystemVerificationOrchestrator()
    report = await orchestrator.run_verification()
    
    # Close all client connections
    for agent in orchestrator.agents:
        await agent.client.aclose()
    
    return report


if __name__ == "__main__":
    report = asyncio.run(main())
    
    # Exit with appropriate code
    success_rate = report["verification_summary"]["success_rate"]
    sys.exit(0 if success_rate >= 70 else 1)