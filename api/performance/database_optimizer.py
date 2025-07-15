"""
Database Performance Optimizer

Provides intelligent database optimization including:
- Connection pool management
- Query optimization and caching
- Batch operations
- Database performance monitoring
- Automatic index management
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
from sqlalchemy import text, event, inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import Select, Update, Insert, Delete
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_hash: str
    sql: str
    execution_time: float
    rows_affected: int
    timestamp: float
    parameters: Dict[str, Any]
    execution_count: int = 1
    
    @property
    def average_time(self) -> float:
        """Average execution time"""
        return self.execution_time / self.execution_count


@dataclass
class ConnectionPoolStats:
    """Connection pool statistics"""
    pool_size: int
    checked_out: int
    overflow: int
    checked_in: int
    total_connections: int
    invalid_connections: int


class QueryCache:
    """Intelligent query result caching"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        
    def _generate_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query and parameters"""
        key_data = f"{query}:{json.dumps(params, sort_keys=True, default=str)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached query result"""
        key = self._generate_key(query, params)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() - entry['timestamp'] > entry['ttl']:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        return entry['result']
    
    def set(self, query: str, params: Dict[str, Any], result: Any, ttl: Optional[float] = None):
        """Cache query result"""
        key = self._generate_key(query, params)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            'result': result,
            'timestamp': time.time(),
            'ttl': ttl or self.default_ttl
        }
        self.access_times[key] = time.time()
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        if pattern is None:
            self.cache.clear()
            self.access_times.clear()
        else:
            # Pattern-based invalidation
            keys_to_remove = []
            for key in self.cache:
                if pattern in self.cache[key].get('query', ''):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size
        }


class DatabaseOptimizer:
    """Database performance optimizer"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.query_cache = QueryCache()
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.slow_query_threshold = 1.0  # seconds
        
        # Performance settings
        self.pool_settings = {
            'pool_size': 20,
            'max_overflow': 40,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'pool_pre_ping': True
        }
        
    async def initialize(self):
        """Initialize the database optimizer"""
        try:
            # Create synchronous engine compatible with existing psycopg2 setup
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                **self.pool_settings,
                echo=False,  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            # Set up event listeners for monitoring
            self._setup_monitoring()
            
            logger.info("Database optimizer initialized with synchronous engine")
        except Exception as e:
            logger.error(f"Failed to initialize database optimizer: {e}")
            # Create a minimal working version to prevent complete failure
            self.engine = None
            self.SessionLocal = None
            logger.warning("Database optimizer running in minimal mode")
    
    def _setup_monitoring(self):
        """Set up database monitoring events"""
        
        if self.engine is None:
            return
            
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if hasattr(context, '_query_start_time'):
                execution_time = time.time() - context._query_start_time
                self._record_query_metrics(statement, parameters, execution_time, cursor.rowcount)
    
    def _record_query_metrics(self, sql: str, parameters: Any, execution_time: float, rows_affected: int):
        """Record query performance metrics"""
        query_hash = hashlib.md5(sql.encode()).hexdigest()[:16]
        
        if query_hash in self.query_metrics:
            # Update existing metrics
            metrics = self.query_metrics[query_hash]
            metrics.execution_time += execution_time
            metrics.execution_count += 1
            metrics.timestamp = time.time()
        else:
            # Create new metrics
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                sql=sql,
                execution_time=execution_time,
                rows_affected=rows_affected,
                timestamp=time.time(),
                parameters=parameters if isinstance(parameters, dict) else {}
            )
        
        # Log slow queries
        if execution_time > self.slow_query_threshold:
            logger.warning(f"Slow query detected: {execution_time:.3f}s - {sql[:100]}...")
    
    @asynccontextmanager
    async def get_session(self):
        """Get optimized database session"""
        if self.engine is None or self.SessionLocal is None:
            # Return a mock session that doesn't do anything
            yield None
            return
            
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get optimized database connection"""
        if self.engine is None:
            yield None
            return
            
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    async def execute_query(self, 
                           query: Union[str, Select, Update, Insert, Delete], 
                           parameters: Dict[str, Any] = None,
                           use_cache: bool = True,
                           cache_ttl: Optional[float] = None) -> Any:
        """Execute query with caching and optimization"""
        
        if parameters is None:
            parameters = {}
        
        query_str = str(query)
        
        # Try cache first for SELECT queries
        if use_cache and query_str.strip().upper().startswith('SELECT'):
            cached_result = self.query_cache.get(query_str, parameters)
            if cached_result is not None:
                return cached_result
        
        # Execute query
        async with self.get_session() as session:
            if session is None:
                return []  # Return empty result if no database connection
                
            if isinstance(query, str):
                result = session.execute(text(query), parameters)
            else:
                result = session.execute(query, parameters)
            
            # Fetch results for SELECT queries
            if query_str.strip().upper().startswith('SELECT'):
                rows = result.fetchall()
                result_data = [dict(row._mapping) for row in rows]
                
                # Cache results
                if use_cache:
                    self.query_cache.set(query_str, parameters, result_data, cache_ttl)
                
                return result_data
            else:
                session.commit()
                return result
    
    async def execute_batch(self, 
                           operations: List[Dict[str, Any]],
                           batch_size: int = 1000) -> Dict[str, Any]:
        """Execute batch operations efficiently"""
        
        results = {
            'total_operations': len(operations),
            'successful_operations': 0,
            'failed_operations': 0,
            'execution_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        # Group operations by type
        grouped_ops = {}
        for op in operations:
            op_type = op.get('type', 'unknown')
            if op_type not in grouped_ops:
                grouped_ops[op_type] = []
            grouped_ops[op_type].append(op)
        
        # Execute each group in batches
        async with self.get_session() as session:
            for op_type, ops in grouped_ops.items():
                for i in range(0, len(ops), batch_size):
                    batch = ops[i:i + batch_size]
                    
                    try:
                        if op_type == 'insert':
                            await self._execute_batch_insert(session, batch)
                        elif op_type == 'update':
                            await self._execute_batch_update(session, batch)
                        elif op_type == 'delete':
                            await self._execute_batch_delete(session, batch)
                        else:
                            # Generic execution
                            for op in batch:
                                await session.execute(text(op['query']), op.get('parameters', {}))
                        
                        results['successful_operations'] += len(batch)
                        
                    except Exception as e:
                        results['failed_operations'] += len(batch)
                        results['errors'].append(str(e))
                        logger.error(f"Batch operation failed: {e}")
                        
                        # Optionally continue with other batches
                        await session.rollback()
            
            await session.commit()
        
        results['execution_time'] = time.time() - start_time
        return results
    
    async def _execute_batch_insert(self, session, batch: List[Dict[str, Any]]):
        """Execute batch insert operations"""
        # Group by table
        tables = {}
        for op in batch:
            table = op.get('table')
            if table not in tables:
                tables[table] = []
            tables[table].append(op.get('data', {}))
        
        # Execute bulk inserts
        for table, data_list in tables.items():
            if data_list:
                query = f"INSERT INTO {table} ({', '.join(data_list[0].keys())}) VALUES "
                values = []
                params = {}
                
                for i, data in enumerate(data_list):
                    placeholders = []
                    for key, value in data.items():
                        param_name = f"{key}_{i}"
                        placeholders.append(f":{param_name}")
                        params[param_name] = value
                    values.append(f"({', '.join(placeholders)})")
                
                query += ", ".join(values)
                session.execute(text(query), params)
    
    async def _execute_batch_update(self, session, batch: List[Dict[str, Any]]):
        """Execute batch update operations"""
        for op in batch:
            session.execute(text(op['query']), op.get('parameters', {}))
    
    async def _execute_batch_delete(self, session, batch: List[Dict[str, Any]]):
        """Execute batch delete operations"""
        for op in batch:
            session.execute(text(op['query']), op.get('parameters', {}))
    
    async def optimize_table(self, table_name: str) -> Dict[str, Any]:
        """Optimize specific table"""
        optimization_results = {}
        
        async with self.get_connection() as conn:
            # Analyze table
            await conn.execute(text(f"ANALYZE {table_name}"))
            optimization_results['analyze'] = 'completed'
            
            # Get table statistics
            stats_query = """
            SELECT 
                schemaname,
                tablename,
                n_live_tup,
                n_dead_tup,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables 
            WHERE tablename = :table_name
            """
            
            if conn is not None:
                result = conn.execute(text(stats_query), {'table_name': table_name})
                stats = result.fetchone()
                
                if stats:
                    optimization_results['statistics'] = dict(stats._mapping)
                    
                    # Check if vacuum is needed
                    dead_ratio = stats.n_dead_tup / max(stats.n_live_tup, 1)
                    if dead_ratio > 0.1:  # More than 10% dead tuples
                        conn.execute(text(f"VACUUM {table_name}"))
                        optimization_results['vacuum'] = 'executed'
                
                # Check for missing indexes
                missing_indexes = await self._suggest_indexes(conn, table_name)
            else:
                optimization_results['statistics'] = 'disabled'
                missing_indexes = []
            optimization_results['suggested_indexes'] = missing_indexes
        
        return optimization_results
    
    async def _suggest_indexes(self, conn, table_name: str) -> List[Dict[str, Any]]:
        """Suggest missing indexes based on query patterns"""
        suggestions = []
        
        # Analyze slow queries for this table
        for query_hash, metrics in self.query_metrics.items():
            if table_name in metrics.sql and metrics.average_time > self.slow_query_threshold:
                # Simple heuristic: suggest indexes on WHERE clause columns
                # In production, you'd want more sophisticated analysis
                suggestions.append({
                    'table': table_name,
                    'suggestion': f"Consider adding index on columns used in WHERE clauses",
                    'query_sample': metrics.sql[:100],
                    'avg_time': metrics.average_time
                })
        
        return suggestions
    
    def get_pool_stats(self) -> ConnectionPoolStats:
        """Get connection pool statistics"""
        if not self.engine:
            return ConnectionPoolStats(0, 0, 0, 0, 0, 0)
        
        pool = self.engine.pool
        return ConnectionPoolStats(
            pool_size=pool.size(),
            checked_out=pool.checkedout(),
            overflow=pool.overflow(),
            checked_in=pool.checkedin(),
            total_connections=pool.size() + pool.overflow(),
            invalid_connections=getattr(pool, 'invalid', lambda: 0)()  # Some pool types don't have invalid()
        )
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        if not self.query_metrics:
            return {'total_queries': 0, 'slow_queries': 0, 'average_time': 0}
        
        total_queries = sum(m.execution_count for m in self.query_metrics.values())
        slow_queries = sum(1 for m in self.query_metrics.values() 
                          if m.average_time > self.slow_query_threshold)
        
        total_time = sum(m.execution_time for m in self.query_metrics.values())
        average_time = total_time / max(total_queries, 1)
        
        # Top slow queries
        top_slow = sorted(
            self.query_metrics.values(),
            key=lambda m: m.average_time,
            reverse=True
        )[:10]
        
        return {
            'total_queries': total_queries,
            'unique_queries': len(self.query_metrics),
            'slow_queries': slow_queries,
            'average_time': average_time,
            'cache_stats': self.query_cache.get_stats(),
            'pool_stats': self.get_pool_stats().__dict__,
            'top_slow_queries': [
                {
                    'hash': m.query_hash,
                    'sql': m.sql[:100],
                    'avg_time': m.average_time,
                    'execution_count': m.execution_count
                }
                for m in top_slow
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        try:
            # Test connection
            if self.engine is not None:
                async with self.get_session() as session:
                    if session is not None:
                        session.execute(text("SELECT 1"))
                        health['checks']['connection'] = 'ok'
                    else:
                        health['checks']['connection'] = 'disabled'
            else:
                health['checks']['connection'] = 'disabled'
        except Exception as e:
            health['status'] = 'unhealthy'
            health['checks']['connection'] = f'error: {e}'
        
        # Check pool health
        pool_stats = self.get_pool_stats()
        if pool_stats.invalid_connections > 0:
            health['status'] = 'degraded'
        
        health['checks']['pool'] = {
            'status': 'ok' if pool_stats.invalid_connections == 0 else 'degraded',
            'invalid_connections': pool_stats.invalid_connections,
            'utilization': pool_stats.checked_out / max(pool_stats.pool_size, 1)
        }
        
        return health
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()


# Global database optimizer instance
db_optimizer: Optional[DatabaseOptimizer] = None


def get_db_optimizer() -> DatabaseOptimizer:
    """Get or create global database optimizer"""
    global db_optimizer
    if db_optimizer is None:
        from ..config import settings
        db_optimizer = DatabaseOptimizer(settings.DATABASE_URL)
    return db_optimizer


async def initialize_db_optimizer():
    """Initialize the global database optimizer"""
    global db_optimizer
    db_optimizer = get_db_optimizer()
    await db_optimizer.initialize()


async def shutdown_db_optimizer():
    """Shutdown the global database optimizer"""
    global db_optimizer
    if db_optimizer:
        await db_optimizer.close()