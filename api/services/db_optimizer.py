
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text
import asyncio
from typing import List, Dict, Any

class DatabaseOptimizer:
    """Database query optimization utilities"""
    
    def __init__(self):
        # Optimized connection pool
        self.engine = create_async_engine(
            os.getenv("DATABASE_URL"),
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False
        )
        
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def batch_query(self, queries: List[str]) -> List[Dict]:
        """Execute multiple queries in parallel"""
        async with self.async_session() as session:
            tasks = [session.execute(text(query)) for query in queries]
            results = await asyncio.gather(*tasks)
            return [r.mappings().all() for r in results]
    
    async def optimize_pagination(
        self,
        query: str,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """Optimized pagination with cursor"""
        offset = (page - 1) * page_size
        
        # Use cursor-based pagination for better performance
        paginated_query = f"""
        WITH paginated AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY created_at DESC) as row_num
            FROM ({query}) as base_query
        )
        SELECT * FROM paginated
        WHERE row_num BETWEEN {offset + 1} AND {offset + page_size}
        """
        
        async with self.async_session() as session:
            result = await session.execute(text(paginated_query))
            return {
                "data": result.mappings().all(),
                "page": page,
                "page_size": page_size
            }
    
    async def create_indexes(self):
        """Create optimized indexes"""
        indexes = [
            # Memory indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_user_session_type ON memories(user_id, session_id, memory_type)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_created_desc ON memories(created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_relevance ON memories(relevance_score DESC)",
            
            # Document indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_content_gin ON documents USING gin(to_tsvector('english', content))",
            
            # Session indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_active ON sessions(user_id, is_active)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)"
        ]
        
        async with self.async_session() as session:
            for index in indexes:
                try:
                    await session.execute(text(index))
                    await session.commit()
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

db_optimizer = DatabaseOptimizer()
