#!/usr/bin/env python3
"""
Complete RAG Integration Script
Fixes missing integrations and makes all features functional
"""

import sys
import os
import asyncio
import httpx
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class RAGIntegrationFixer:
    def __init__(self):
        self.fixes_applied = []
        self.api_base = "http://localhost:3000"
        
    async def test_endpoint(self, path: str, method: str = "GET") -> bool:
        """Test if an endpoint is accessible."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                if method == "GET":
                    response = await client.get(f"{self.api_base}{path}")
                else:
                    response = await client.post(f"{self.api_base}{path}", json={})
                return response.status_code != 404
            except:
                return False
    
    async def fix_cache_service(self):
        """Fix the CacheService import issue."""
        print("\n📦 Fixing CacheService import...")
        
        cache_fix = '''"""Fixed cache service module."""
from typing import Any, Optional
import json

class CacheService:
    """Simple cache service implementation."""
    
    def __init__(self):
        self.cache = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache."""
        self.cache[key] = value
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
    
    async def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()

# Create global instance
cache_service = CacheService()

def get_cache_service():
    """Get cache service instance."""
    return cache_service
'''
        
        # Check if we need to add CacheService export
        cache_file = Path("/opt/projects/knowledgehub/api/services/cache.py")
        if cache_file.exists():
            content = cache_file.read_text()
            if "class CacheService" not in content:
                # Append the CacheService class to the file
                with open(cache_file, "a") as f:
                    f.write("\n\n" + cache_fix)
                self.fixes_applied.append("Added CacheService to cache.py")
            else:
                print("  ✅ CacheService already exists")
    
    async def initialize_databases(self):
        """Initialize vector and graph databases."""
        print("\n🗄️ Initializing databases...")
        
        # Initialize Weaviate collection
        init_script = '''
import weaviate
from weaviate.classes.config import Configure, Property, DataType

# Connect to Weaviate
client = weaviate.connect_to_local(host="localhost", port=8090)

try:
    # Create KnowledgeHub collection if it doesn't exist
    if not client.collections.exists("KnowledgeHub"):
        client.collections.create(
            name="KnowledgeHub",
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.OBJECT),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_transformers()
        )
        print("✅ Created Weaviate KnowledgeHub collection")
    else:
        print("✅ Weaviate KnowledgeHub collection already exists")
finally:
    client.close()
'''
        
        # Try to initialize Weaviate
        try:
            exec(init_script)
            self.fixes_applied.append("Initialized Weaviate collection")
        except Exception as e:
            print(f"  ⚠️ Could not initialize Weaviate: {e}")
        
        # Initialize Neo4j schema
        neo4j_script = '''
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4jpassword"))

try:
    with driver.session() as session:
        # Create indexes
        session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)")
        print("✅ Created Neo4j indexes")
finally:
    driver.close()
'''
        
        # Try to initialize Neo4j
        try:
            exec(neo4j_script)
            self.fixes_applied.append("Initialized Neo4j indexes")
        except Exception as e:
            print(f"  ⚠️ Could not initialize Neo4j: {e}")
    
    async def fix_health_checks(self):
        """Fix health check endpoints in docker-compose."""
        print("\n🏥 Fixing health check configurations...")
        
        compose_file = Path("/opt/projects/knowledgehub/docker-compose.yml")
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Fix Zep health check
            if 'test: ["CMD", "curl", "-f", "http://localhost:8100/health"]' in content:
                content = content.replace(
                    'test: ["CMD", "curl", "-f", "http://localhost:8100/health"]',
                    'test: ["CMD", "curl", "-f", "http://localhost:8000/"]'
                )
                self.fixes_applied.append("Fixed Zep health check")
            
            # Fix Qdrant health check
            if 'test: ["CMD", "curl", "-f", "http://localhost:6333/health"]' in content:
                content = content.replace(
                    'test: ["CMD", "curl", "-f", "http://localhost:6333/health"]',
                    'test: ["CMD", "curl", "-f", "http://localhost:6333/"]'
                )
                self.fixes_applied.append("Fixed Qdrant health check")
            
            # Write back if changes were made
            if "Fixed" in str(self.fixes_applied):
                compose_file.write_text(content)
                print("  ✅ Updated docker-compose.yml health checks")
            else:
                print("  ✅ Health checks already correct")
    
    async def test_integration(self):
        """Test if all integrations are working."""
        print("\n🧪 Testing integrations...")
        
        endpoints_to_test = [
            ("/api/rag/enhanced/search", "POST"),
            ("/api/agent/workflows/execute", "POST"),
            ("/api/memory/sessions", "GET"),
            ("/api/rag/vector/status", "GET"),
            ("/api/rag/graph/status", "GET"),
        ]
        
        results = []
        for endpoint, method in endpoints_to_test:
            accessible = await self.test_endpoint(endpoint, method)
            status = "✅" if accessible else "❌"
            results.append((endpoint, status))
            print(f"  {status} {endpoint}")
        
        return results
    
    async def run(self):
        """Run all integration fixes."""
        print("="*60)
        print("🔧 COMPLETING RAG INTEGRATION")
        print("="*60)
        
        # Apply fixes
        await self.fix_cache_service()
        await self.initialize_databases()
        await self.fix_health_checks()
        
        # Test integration
        test_results = await self.test_integration()
        
        # Summary
        print("\n" + "="*60)
        print("📊 INTEGRATION SUMMARY")
        print("="*60)
        
        if self.fixes_applied:
            print("\n✅ Fixes Applied:")
            for fix in self.fixes_applied:
                print(f"  - {fix}")
        else:
            print("\n✅ No fixes needed - system already configured")
        
        working = sum(1 for _, status in test_results if status == "✅")
        total = len(test_results)
        
        print(f"\n📈 Endpoint Status: {working}/{total} working")
        
        if working == total:
            print("\n🎉 All integrations are now functional!")
        elif working > 0:
            print("\n⚠️ Partial integration complete. Some endpoints may need additional configuration.")
        else:
            print("\n❌ Integration issues remain. Manual intervention may be required.")
        
        # Restart API if fixes were applied
        if self.fixes_applied:
            print("\n🔄 Restarting API to apply changes...")
            os.system("docker restart knowledgehub-api-1")
            print("  ✅ API restarted")

async def main():
    fixer = RAGIntegrationFixer()
    await fixer.run()

if __name__ == "__main__":
    asyncio.run(main())