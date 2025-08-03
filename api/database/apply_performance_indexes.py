#!/usr/bin/env python3
"""
Apply Performance Indexes for Memory System

This script applies optimized database indexes for the memory system
based on actual query patterns analysis.
"""

import asyncio
import asyncpg
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import settings


class PerformanceIndexManager:
    """Manages application and monitoring of performance indexes"""
    
    def __init__(self):
        self.connection = None
        self.applied_indexes = []
        self.failed_indexes = []
        
    async def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = await asyncpg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB
            )
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    async def check_existing_indexes(self) -> Dict[str, List[str]]:
        """Check which indexes already exist"""
        query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND (tablename = 'memories' OR tablename = 'memory_sessions')
        ORDER BY tablename, indexname;
        """
        
        try:
            rows = await self.connection.fetch(query)
            existing_indexes = {
                'memories': [],
                'memory_sessions': []
            }
            
            for row in rows:
                existing_indexes[row['tablename']].append(row['indexname'])
            
            print(f"\n📊 Existing Indexes:")
            print(f"   memories table: {len(existing_indexes['memories'])} indexes")
            print(f"   memory_sessions table: {len(existing_indexes['memory_sessions'])} indexes")
            
            return existing_indexes
            
        except Exception as e:
            print(f"❌ Error checking existing indexes: {e}")
            return {'memories': [], 'memory_sessions': []}
    
    async def get_table_stats(self) -> Dict[str, Any]:
        """Get current table statistics"""
        stats_query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_rows,
            n_dead_tup as dead_rows,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM pg_stat_user_tables 
        WHERE tablename IN ('memories', 'memory_sessions');
        """
        
        try:
            rows = await self.connection.fetch(stats_query)
            stats = {}
            
            for row in rows:
                stats[row['tablename']] = dict(row)
            
            print(f"\n📈 Table Statistics:")
            for table, data in stats.items():
                print(f"   {table}:")
                print(f"     Live rows: {data['live_rows']:,}")
                print(f"     Dead rows: {data['dead_rows']:,}")
                print(f"     Total operations: {data['inserts'] + data['updates'] + data['deletes']:,}")
                
            return stats
            
        except Exception as e:
            print(f"❌ Error getting table stats: {e}")
            return {}
    
    async def apply_performance_indexes(self) -> bool:
        """Apply the performance indexes from SQL file"""
        sql_file = Path(__file__).parent / "performance_indexes.sql"
        
        if not sql_file.exists():
            print(f"❌ SQL file not found: {sql_file}")
            return False
        
        try:
            print(f"\n🔧 Applying performance indexes from {sql_file.name}...")
            start_time = time.time()
            
            # Read SQL file
            sql_content = sql_file.read_text()
            
            # Execute the SQL
            await self.connection.execute(sql_content)
            
            execution_time = time.time() - start_time
            print(f"✅ Performance indexes applied successfully")
            print(f"   Execution time: {execution_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"❌ Error applying indexes: {e}")
            return False
    
    async def measure_query_performance(self) -> Dict[str, float]:
        """Measure performance of common query patterns"""
        test_queries = {
            "session_memories": """
                SELECT COUNT(*) FROM memories 
                WHERE session_id = (
                    SELECT id FROM memory_sessions 
                    ORDER BY created_at DESC LIMIT 1
                );
            """,
            
            "importance_filter": """
                SELECT COUNT(*) FROM memories 
                WHERE importance >= 0.8 
                ORDER BY created_at DESC 
                LIMIT 100;
            """,
            
            "type_filter": """
                SELECT COUNT(*) FROM memories 
                WHERE memory_type = 'decision' 
                AND importance >= 0.7 
                ORDER BY created_at DESC 
                LIMIT 50;
            """,
            
            "user_sessions": """
                SELECT COUNT(*) FROM memory_sessions 
                WHERE user_id LIKE 'test-%' 
                AND ended_at IS NULL 
                ORDER BY updated_at DESC;
            """,
            
            "entity_search": """
                SELECT COUNT(*) FROM memories 
                WHERE entities && ARRAY['test', 'system'] 
                LIMIT 100;
            """,
            
            "text_search": """
                SELECT COUNT(*) FROM memories 
                WHERE content ILIKE '%implementation%' 
                OR summary ILIKE '%implementation%' 
                LIMIT 50;
            """
        }
        
        performance_results = {}
        
        print(f"\n⚡ Testing Query Performance:")
        
        for query_name, query in test_queries.items():
            try:
                start_time = time.time()
                
                # Execute query multiple times for average
                for _ in range(3):
                    await self.connection.fetch(query)
                
                avg_time = (time.time() - start_time) / 3
                performance_results[query_name] = avg_time
                
                print(f"   {query_name}: {avg_time:.3f}s")
                
            except Exception as e:
                print(f"   {query_name}: FAILED - {e}")
                performance_results[query_name] = -1
        
        return performance_results
    
    async def analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze index usage statistics"""
        usage_query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            idx_tup_read,
            idx_tup_fetch,
            idx_scan as scans
        FROM pg_stat_user_indexes 
        WHERE schemaname = 'public' 
        AND (tablename = 'memories' OR tablename = 'memory_sessions')
        AND idx_scan > 0
        ORDER BY idx_scan DESC;
        """
        
        try:
            rows = await self.connection.fetch(usage_query)
            
            print(f"\n📊 Index Usage Statistics (Top 10):")
            
            for i, row in enumerate(rows[:10]):
                print(f"   {i+1}. {row['indexname']}")
                print(f"      Table: {row['tablename']}")
                print(f"      Scans: {row['scans']:,}")
                print(f"      Tuples read: {row['idx_tup_read']:,}")
                print("")
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            print(f"❌ Error analyzing index usage: {e}")
            return []
    
    async def get_index_sizes(self) -> Dict[str, Any]:
        """Get index sizes for monitoring"""
        size_query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            pg_size_pretty(pg_relation_size(indexrelid)) as size,
            pg_relation_size(indexrelid) as size_bytes
        FROM pg_stat_user_indexes 
        WHERE schemaname = 'public' 
        AND (tablename = 'memories' OR tablename = 'memory_sessions')
        ORDER BY pg_relation_size(indexrelid) DESC;
        """
        
        try:
            rows = await self.connection.fetch(size_query)
            
            total_size = sum(row['size_bytes'] for row in rows)
            
            print(f"\n💾 Index Storage Usage:")
            print(f"   Total index size: {self._format_bytes(total_size)}")
            print(f"   Number of indexes: {len(rows)}")
            
            print(f"\n   Largest indexes:")
            for row in rows[:5]:
                print(f"     {row['indexname']}: {row['size']}")
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            print(f"❌ Error getting index sizes: {e}")
            return []
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes into human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            print("🔌 Database connection closed")


async def main():
    """Main execution function"""
    print("🚀 Memory System Performance Index Manager\n")
    
    manager = PerformanceIndexManager()
    
    try:
        # Connect to database
        if not await manager.connect():
            return False
        
        # Check current state
        print("=" * 60)
        print("📊 CURRENT DATABASE STATE")
        print("=" * 60)
        
        existing_indexes = await manager.check_existing_indexes()
        table_stats = await manager.get_table_stats()
        
        # Measure performance before optimization
        print("\n" + "=" * 60)
        print("⚡ PERFORMANCE BASELINE")
        print("=" * 60)
        
        before_performance = await manager.measure_query_performance()
        
        # Apply performance indexes
        print("\n" + "=" * 60)
        print("🔧 APPLYING PERFORMANCE INDEXES")
        print("=" * 60)
        
        success = await manager.apply_performance_indexes()
        
        if not success:
            print("❌ Failed to apply performance indexes")
            return False
        
        # Check results
        print("\n" + "=" * 60)
        print("📈 POST-OPTIMIZATION ANALYSIS")
        print("=" * 60)
        
        # Wait a moment for statistics to update
        await asyncio.sleep(2)
        
        after_performance = await manager.measure_query_performance()
        index_usage = await manager.analyze_index_usage()
        index_sizes = await manager.get_index_sizes()
        
        # Performance comparison
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE COMPARISON")
        print("=" * 60)
        
        print(f"{'Query':<20} {'Before':<10} {'After':<10} {'Improvement':<12}")
        print("-" * 60)
        
        for query_name in before_performance:
            before = before_performance.get(query_name, 0)
            after = after_performance.get(query_name, 0)
            
            if before > 0 and after > 0:
                improvement = ((before - after) / before) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{query_name:<20} {before:<10.3f} {after:<10.3f} {improvement_str:<12}")
        
        print("\n✅ Performance optimization completed successfully!")
        print("\n💡 Recommendations:")
        print("   1. Monitor index usage over time")
        print("   2. Run ANALYZE regularly to update statistics")
        print("   3. Consider VACUUM if dead tuple ratio is high")
        print("   4. Monitor slow query logs for additional optimizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
        
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())