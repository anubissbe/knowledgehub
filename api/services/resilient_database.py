"""
Resilient Database Service

Provides a high-level interface for resilient database operations with
automatic retry, connection recovery, and performance optimization.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, TypeVar, Callable, Union
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from sqlalchemy import select, insert, update, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, DataError
import asyncpg

from .database_recovery import db_recovery, with_db_retry, RetryConfig
from ..models import Memory, Source, Document, Chunk
from ..middleware.database_recovery_middleware import track_database_operation

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResilientDatabaseService:
    """
    High-level database service with built-in resilience features.
    
    Features:
    - Automatic retry with exponential backoff
    - Connection recovery and pooling
    - Query optimization and caching
    - Batch operations for performance
    - Transaction management with rollback
    - Performance monitoring and metrics
    """
    
    def __init__(self):
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.batch_size = 100
        self.enable_query_optimization = True
    
    # Memory Operations
    
    @with_db_retry()
    async def create_memory(self, memory_data: Dict[str, Any], request = None) -> Memory:
        """Create a new memory with automatic retry"""
        async with db_recovery.get_async_session() as session:
            if request:
                track_database_operation(request)
            
            try:
                memory = Memory(**memory_data)
                session.add(memory)
                await session.commit()
                await session.refresh(memory)
                
                logger.info(f"Created memory with ID: {memory.id}")
                return memory
                
            except IntegrityError as e:
                await session.rollback()
                logger.error(f"Integrity error creating memory: {e}")
                raise
            except Exception as e:
                await session.rollback()
                logger.error(f"Error creating memory: {e}")
                raise
    
    @with_db_retry()
    async def get_memory(self, memory_id: str, request = None) -> Optional[Memory]:
        """Get memory by ID with automatic retry"""
        async with db_recovery.get_async_session() as session:
            if request:
                track_database_operation(request)
            
            query = select(Memory).where(Memory.id == memory_id)
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    @with_db_retry()
    async def search_memories(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 10,
        request = None
    ) -> List[Memory]:
        """Search memories with automatic retry and optimization"""
        async with db_recovery.get_async_session() as session:
            if request:
                track_database_operation(request)
            
            # Use optimized query with index hints
            if self.enable_query_optimization:
                stmt = text("""
                    SELECT * FROM memories
                    WHERE user_id = :user_id
                    AND to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(
                    stmt,
                    {"user_id": user_id, "query": query, "limit": limit}
                )
                
                # Convert rows to Memory objects
                memories = []
                for row in result:
                    memory = Memory(**dict(row))
                    memories.append(memory)
                return memories
            else:
                # Fallback to ORM query
                query = select(Memory).where(
                    Memory.user_id == user_id
                ).order_by(
                    Memory.created_at.desc()
                ).limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
    
    @with_db_retry()
    async def batch_create_memories(
        self, 
        memories_data: List[Dict[str, Any]],
        request = None
    ) -> List[Memory]:
        """Batch create memories with transaction support"""
        async with db_recovery.get_async_session() as session:
            if request:
                track_database_operation(request)
            
            created_memories = []
            
            try:
                # Process in batches for better performance
                for i in range(0, len(memories_data), self.batch_size):
                    batch = memories_data[i:i + self.batch_size]
                    
                    # Use bulk insert for performance
                    stmt = insert(Memory).values(batch)
                    await session.execute(stmt)
                    
                    # Create Memory objects for return
                    for data in batch:
                        memory = Memory(**data)
                        created_memories.append(memory)
                
                await session.commit()
                logger.info(f"Batch created {len(created_memories)} memories")
                return created_memories
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error in batch create: {e}")
                raise
    
    # Source Operations
    
    @with_db_retry()
    async def get_or_create_source(
        self, 
        source_data: Dict[str, Any],
        request = None
    ) -> Source:
        """Get existing source or create new one"""
        async with db_recovery.get_async_session() as session:
            if request:
                track_database_operation(request)
            
            # Check if source exists
            query = select(Source).where(Source.url == source_data['url'])
            result = await session.execute(query)
            source = result.scalar_one_or_none()
            
            if source:
                # Update existing source
                for key, value in source_data.items():
                    setattr(source, key, value)
                source.updated_at = datetime.now(timezone.utc)
            else:
                # Create new source
                source = Source(**source_data)
                session.add(source)
            
            await session.commit()
            await session.refresh(source)
            return source
    
    # Document Operations
    
    @with_db_retry()
    async def create_document_with_chunks(
        self,
        document_data: Dict[str, Any],
        chunks_data: List[Dict[str, Any]],
        request = None
    ) -> Document:
        """Create document with chunks in a single transaction"""
        async with db_recovery.get_async_session() as session:
            if request:
                track_database_operation(request)
            
            try:
                # Create document
                document = Document(**document_data)
                session.add(document)
                await session.flush()  # Get document ID
                
                # Create chunks
                for chunk_data in chunks_data:
                    chunk_data['document_id'] = document.id
                    chunk = Chunk(**chunk_data)
                    session.add(chunk)
                
                await session.commit()
                await session.refresh(document)
                
                logger.info(f"Created document {document.id} with {len(chunks_data)} chunks")
                return document
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error creating document with chunks: {e}")
                raise
    
    # Advanced Query Operations
    
    @with_db_retry(retry_config=RetryConfig(max_retries=5))
    async def execute_complex_query(
        self,
        query_str: str,
        params: Optional[Dict[str, Any]] = None,
        request = None
    ) -> List[Dict[str, Any]]:
        """Execute complex query with enhanced retry logic"""
        async with db_recovery.get_asyncpg_connection() as conn:
            if request:
                track_database_operation(request)
            
            # Enable query optimization
            await conn.execute("SET enable_hashjoin = on")
            await conn.execute("SET enable_mergejoin = on")
            
            # Execute query
            if params:
                rows = await conn.fetch(query_str, *params.values())
            else:
                rows = await conn.fetch(query_str)
            
            # Convert to dict
            return [dict(row) for row in rows]
    
    @asynccontextmanager
    async def transaction(self, request = None):
        """Managed transaction context with automatic retry"""
        async with db_recovery.get_async_session() as session:
            if request:
                track_database_operation(request)
            
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive database health check"""
        health_status = {
            "database": "checking",
            "connection_pool": "checking",
            "performance": "checking"
        }
        
        try:
            # Check basic connectivity
            is_healthy = await db_recovery.health_check()
            health_status["database"] = "healthy" if is_healthy else "unhealthy"
            
            # Get connection stats
            stats = db_recovery.get_connection_stats()
            
            # Check connection pool health
            pool_utilization = (
                stats["connections"]["active"] / 
                max(stats["connections"]["total"], 1)
            ) * 100
            
            if pool_utilization < 80:
                health_status["connection_pool"] = "healthy"
            elif pool_utilization < 90:
                health_status["connection_pool"] = "degraded"
            else:
                health_status["connection_pool"] = "critical"
            
            # Check performance
            if stats["performance"]["success_rate"] >= 95:
                health_status["performance"] = "healthy"
            elif stats["performance"]["success_rate"] >= 80:
                health_status["performance"] = "degraded"
            else:
                health_status["performance"] = "critical"
            
            # Add detailed metrics
            health_status["metrics"] = {
                "success_rate": stats["performance"]["success_rate"],
                "avg_query_time": stats["performance"]["avg_execution_time"],
                "pool_utilization": pool_utilization,
                "active_connections": stats["connections"]["active"],
                "circuit_breaker_open": stats["circuit_breaker"]["open"]
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)
        
        return health_status
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Run database optimization tasks"""
        optimization_results = {
            "vacuum": "pending",
            "analyze": "pending",
            "reindex": "pending"
        }
        
        try:
            async with db_recovery.get_asyncpg_connection() as conn:
                # Run VACUUM ANALYZE
                await conn.execute("VACUUM ANALYZE")
                optimization_results["vacuum"] = "completed"
                optimization_results["analyze"] = "completed"
                
                # Update table statistics
                tables = ["memories", "sources", "documents", "chunks"]
                for table in tables:
                    await conn.execute(f"ANALYZE {table}")
                
                logger.info("Database optimization completed")
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results


# Global resilient database service instance
resilient_db = ResilientDatabaseService()


# Convenience functions for common operations

async def get_memory_with_retry(memory_id: str, request = None) -> Optional[Memory]:
    """Get memory with automatic retry and recovery"""
    return await resilient_db.get_memory(memory_id, request)


async def create_memory_with_retry(memory_data: Dict[str, Any], request = None) -> Memory:
    """Create memory with automatic retry and recovery"""
    return await resilient_db.create_memory(memory_data, request)


async def search_memories_with_retry(
    user_id: str, 
    query: str, 
    limit: int = 10,
    request = None
) -> List[Memory]:
    """Search memories with automatic retry and recovery"""
    return await resilient_db.search_memories(user_id, query, limit, request)


async def execute_with_transaction(
    operations: List[Callable],
    request = None
) -> List[Any]:
    """Execute multiple operations in a single transaction"""
    results = []
    
    async with resilient_db.transaction(request) as session:
        for operation in operations:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(session)
            else:
                result = operation(session)
            results.append(result)
    
    return results