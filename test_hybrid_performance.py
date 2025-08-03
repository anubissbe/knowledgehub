#!/usr/bin/env python3
"""
Test SQLite performance gains for hybrid memory system
"""

import asyncio
import time
import random
import string
from pathlib import Path
import aiosqlite
import psycopg2
from datetime import datetime
import json

# Test configuration
NUM_MEMORIES = 1000
SEARCH_QUERIES = 100
TEST_DB_PATH = "/tmp/knowledgehub_perf_test.db"


class PerformanceTester:
    def __init__(self):
        self.results = {
            "sqlite": {"insert": [], "search": []},
            "postgresql": {"insert": [], "search": []},
        }
        
    async def setup_sqlite(self):
        """Setup SQLite test database"""
        Path(TEST_DB_PATH).unlink(missing_ok=True)
        
        async with aiosqlite.connect(TEST_DB_PATH) as db:
            await db.execute("""
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE VIRTUAL TABLE memory_fts USING fts5(
                    id, content, tokenize='porter'
                )
            """)
            
            await db.execute("CREATE INDEX idx_type ON memories(type)")
            await db.commit()
    
    def setup_postgresql(self):
        """Setup PostgreSQL test connection"""
        return psycopg2.connect(
            host="localhost",
            port=5433,
            database="knowledgehub",
            user="knowledgehub",
            password="knowledgehub123"
        )
    
    def generate_memory(self, i: int):
        """Generate test memory data"""
        types = ["code", "error", "decision", "documentation", "general"]
        content = f"Memory {i}: " + " ".join(
            "".join(random.choices(string.ascii_letters, k=10))
            for _ in range(random.randint(10, 50))
        )
        return {
            "id": f"test_{i}_{int(time.time()*1000000)}",
            "content": content,
            "type": random.choice(types),
            "created_at": datetime.utcnow()
        }
    
    async def test_sqlite_insert(self):
        """Test SQLite insert performance"""
        print("\n🔵 Testing SQLite inserts...")
        
        async with aiosqlite.connect(TEST_DB_PATH) as db:
            for i in range(NUM_MEMORIES):
                memory = self.generate_memory(i)
                
                start = time.time()
                
                await db.execute("""
                    INSERT INTO memories (id, content, type)
                    VALUES (?, ?, ?)
                """, (memory["id"], memory["content"], memory["type"]))
                
                await db.execute("""
                    INSERT INTO memory_fts (id, content)
                    VALUES (?, ?)
                """, (memory["id"], memory["content"]))
                
                await db.commit()
                
                elapsed = (time.time() - start) * 1000  # ms
                self.results["sqlite"]["insert"].append(elapsed)
                
                if i % 100 == 0:
                    avg = sum(self.results["sqlite"]["insert"]) / len(self.results["sqlite"]["insert"])
                    print(f"  Inserted {i+1}/{NUM_MEMORIES} - Avg: {avg:.2f}ms")
    
    def test_postgresql_insert(self):
        """Test PostgreSQL insert performance"""
        print("\n🟢 Testing PostgreSQL inserts...")
        
        conn = self.setup_postgresql()
        cur = conn.cursor()
        
        # Create test table
        cur.execute("DROP TABLE IF EXISTS perf_test_memories")
        cur.execute("""
            CREATE TABLE perf_test_memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        for i in range(NUM_MEMORIES):
            memory = self.generate_memory(i)
            
            start = time.time()
            
            cur.execute("""
                INSERT INTO perf_test_memories (id, content, type)
                VALUES (%s, %s, %s)
            """, (memory["id"], memory["content"], memory["type"]))
            
            conn.commit()
            
            elapsed = (time.time() - start) * 1000  # ms
            self.results["postgresql"]["insert"].append(elapsed)
            
            if i % 100 == 0:
                avg = sum(self.results["postgresql"]["insert"]) / len(self.results["postgresql"]["insert"])
                print(f"  Inserted {i+1}/{NUM_MEMORIES} - Avg: {avg:.2f}ms")
        
        cur.close()
        conn.close()
    
    async def test_sqlite_search(self):
        """Test SQLite search performance"""
        print("\n🔵 Testing SQLite searches...")
        
        search_terms = ["memory", "code", "error", "test", "data", "function", "system"]
        
        async with aiosqlite.connect(TEST_DB_PATH) as db:
            for i in range(SEARCH_QUERIES):
                term = random.choice(search_terms)
                
                start = time.time()
                
                cursor = await db.execute("""
                    SELECT m.*, rank
                    FROM memories m
                    JOIN (
                        SELECT id, rank FROM memory_fts 
                        WHERE memory_fts MATCH ?
                        ORDER BY rank
                        LIMIT 10
                    ) fts ON m.id = fts.id
                """, (term,))
                
                results = await cursor.fetchall()
                
                elapsed = (time.time() - start) * 1000  # ms
                self.results["sqlite"]["search"].append(elapsed)
                
                if i % 20 == 0:
                    avg = sum(self.results["sqlite"]["search"]) / len(self.results["sqlite"]["search"])
                    print(f"  Searched {i+1}/{SEARCH_QUERIES} - Avg: {avg:.2f}ms")
    
    def test_postgresql_search(self):
        """Test PostgreSQL search performance"""
        print("\n🟢 Testing PostgreSQL searches...")
        
        search_terms = ["memory", "code", "error", "test", "data", "function", "system"]
        
        conn = self.setup_postgresql()
        cur = conn.cursor()
        
        # Add full-text search
        cur.execute("""
            ALTER TABLE perf_test_memories 
            ADD COLUMN IF NOT EXISTS content_tsvector tsvector
        """)
        cur.execute("""
            UPDATE perf_test_memories 
            SET content_tsvector = to_tsvector('english', content)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_fts 
            ON perf_test_memories USING gin(content_tsvector)
        """)
        conn.commit()
        
        for i in range(SEARCH_QUERIES):
            term = random.choice(search_terms)
            
            start = time.time()
            
            cur.execute("""
                SELECT id, content, type
                FROM perf_test_memories
                WHERE content_tsvector @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(content_tsvector, plainto_tsquery('english', %s)) DESC
                LIMIT 10
            """, (term, term))
            
            results = cur.fetchall()
            
            elapsed = (time.time() - start) * 1000  # ms
            self.results["postgresql"]["search"].append(elapsed)
            
            if i % 20 == 0:
                avg = sum(self.results["postgresql"]["search"]) / len(self.results["postgresql"]["search"])
                print(f"  Searched {i+1}/{SEARCH_QUERIES} - Avg: {avg:.2f}ms")
        
        cur.close()
        conn.close()
    
    def print_results(self):
        """Print performance comparison results"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE COMPARISON RESULTS")
        print("="*60)
        
        # Insert performance
        sqlite_insert_avg = sum(self.results["sqlite"]["insert"]) / len(self.results["sqlite"]["insert"])
        pg_insert_avg = sum(self.results["postgresql"]["insert"]) / len(self.results["postgresql"]["insert"])
        
        print(f"\n📝 INSERT PERFORMANCE (avg per insert):")
        print(f"  SQLite:      {sqlite_insert_avg:.2f}ms")
        print(f"  PostgreSQL:  {pg_insert_avg:.2f}ms")
        print(f"  Speedup:     {pg_insert_avg/sqlite_insert_avg:.1f}x faster")
        
        # Search performance
        sqlite_search_avg = sum(self.results["sqlite"]["search"]) / len(self.results["sqlite"]["search"])
        pg_search_avg = sum(self.results["postgresql"]["search"]) / len(self.results["postgresql"]["search"])
        
        print(f"\n🔍 SEARCH PERFORMANCE (avg per search):")
        print(f"  SQLite:      {sqlite_search_avg:.2f}ms")
        print(f"  PostgreSQL:  {pg_search_avg:.2f}ms")
        print(f"  Speedup:     {pg_search_avg/sqlite_search_avg:.1f}x faster")
        
        # Percentiles
        print(f"\n📈 SEARCH LATENCY PERCENTILES:")
        for db_name, db_results in [("SQLite", self.results["sqlite"]["search"]), 
                                    ("PostgreSQL", self.results["postgresql"]["search"])]:
            sorted_results = sorted(db_results)
            p50 = sorted_results[len(sorted_results)//2]
            p95 = sorted_results[int(len(sorted_results)*0.95)]
            p99 = sorted_results[int(len(sorted_results)*0.99)]
            
            print(f"\n  {db_name}:")
            print(f"    P50: {p50:.2f}ms")
            print(f"    P95: {p95:.2f}ms")
            print(f"    P99: {p99:.2f}ms")
        
        # Save detailed results
        with open("hybrid_performance_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to hybrid_performance_results.json")


async def main():
    """Run performance tests"""
    tester = PerformanceTester()
    
    print("🚀 KnowledgeHub Hybrid Memory Performance Test")
    print(f"Testing with {NUM_MEMORIES} memories and {SEARCH_QUERIES} searches")
    
    # Setup
    await tester.setup_sqlite()
    
    # Test inserts
    await tester.test_sqlite_insert()
    tester.test_postgresql_insert()
    
    # Test searches
    await tester.test_sqlite_search()
    tester.test_postgresql_search()
    
    # Results
    tester.print_results()
    
    # Cleanup
    Path(TEST_DB_PATH).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())