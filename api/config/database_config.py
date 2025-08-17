"""
Enhanced Database Configuration for Hybrid RAG System.

This module provides comprehensive database configuration management for
the hybrid RAG system with support for multiple database services.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, NullPool
import redis
import asyncio
from contextlib import asynccontextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "prefer"
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    def get_url(self, async_driver: bool = False) -> str:
        """Generate database URL."""
        driver = "postgresql+asyncpg" if async_driver else "postgresql"
        return (
            f"{driver}://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"
        )


@dataclass
class RedisConfig:
    """Configuration for Redis connections."""
    host: str
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 50
    health_check_interval: int = 30
    
    def get_url(self) -> str:
        """Generate Redis URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class VectorDBConfig:
    """Configuration for vector databases."""
    url: str
    collection_name: str
    timeout: int = 60
    retries: int = 3
    batch_size: int = 100
    
    # Weaviate specific
    additional_headers: Dict[str, str] = field(default_factory=dict)
    
    # Qdrant specific (alternative)
    api_key: Optional[str] = None
    prefer_grpc: bool = False


@dataclass
class GraphDBConfig:
    """Configuration for graph databases (Neo4j)."""
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    max_pool_size: int = 50
    connection_timeout: int = 30
    encrypted: bool = True


@dataclass
class TimeSeriesDBConfig:
    """Configuration for TimescaleDB."""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class ZepConfig:
    """Configuration for Zep memory service."""
    api_url: str
    api_key: Optional[str] = None
    session_window: int = 12
    timeout: int = 30
    max_retries: int = 3


@dataclass
class MinIOConfig:
    """Configuration for MinIO object storage."""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str = "knowledgehub"
    secure: bool = False
    region: str = "us-east-1"


class DatabaseManager:
    """Centralized database connection manager."""
    
    def __init__(self):
        self.engines: Dict[str, Any] = {}
        self.session_makers: Dict[str, Any] = {}
        self.redis_pools: Dict[str, redis.ConnectionPool] = {}
        self.configs: Dict[str, Any] = {}
        self._initialized = False
    
    def configure_postgresql(
        self,
        name: str,
        config: DatabaseConfig,
        async_engine: bool = False
    ):
        """Configure PostgreSQL connection."""
        try:
            url = config.get_url(async_driver=async_engine)
            
            if async_engine:
                from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
                engine = create_async_engine(
                    url,
                    poolclass=QueuePool,
                    pool_size=config.pool_size,
                    max_overflow=config.max_overflow,
                    pool_timeout=config.pool_timeout,
                    pool_recycle=config.pool_recycle,
                    pool_pre_ping=True,
                    echo=config.echo,
                )
                self.session_makers[name] = sessionmaker(
                    engine, class_=AsyncSession, expire_on_commit=False
                )
            else:
                engine = create_engine(
                    url,
                    poolclass=QueuePool,
                    pool_size=config.pool_size,
                    max_overflow=config.max_overflow,
                    pool_timeout=config.pool_timeout,
                    pool_recycle=config.pool_recycle,
                    pool_pre_ping=True,
                    echo=config.echo,
                )
                self.session_makers[name] = sessionmaker(
                    autocommit=False, autoflush=False, bind=engine
                )
            
            self.engines[name] = engine
            self.configs[name] = config
            
            logger.info(f"Configured PostgreSQL connection: {name}")
            
        except Exception as e:
            logger.error(f"Failed to configure PostgreSQL {name}: {e}")
            raise
    
    def configure_redis(self, name: str, config: RedisConfig):
        """Configure Redis connection."""
        try:
            pool = redis.ConnectionPool(
                host=config.host,
                port=config.port,
                db=config.db,
                password=config.password,
                ssl=config.ssl,
                socket_timeout=config.socket_timeout,
                socket_connect_timeout=config.socket_connect_timeout,
                max_connections=config.max_connections,
                health_check_interval=config.health_check_interval,
                decode_responses=True
            )
            
            self.redis_pools[name] = pool
            self.configs[name] = config
            
            logger.info(f"Configured Redis connection: {name}")
            
        except Exception as e:
            logger.error(f"Failed to configure Redis {name}: {e}")
            raise
    
    def get_postgres_session(self, name: str = "primary"):
        """Get PostgreSQL session."""
        if name not in self.session_makers:
            raise ValueError(f"PostgreSQL connection '{name}' not configured")
        return self.session_makers[name]()
    
    @asynccontextmanager
    async def get_async_postgres_session(self, name: str = "primary"):
        """Get async PostgreSQL session."""
        if name not in self.session_makers:
            raise ValueError(f"Async PostgreSQL connection '{name}' not configured")
        
        async with self.session_makers[name]() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    def get_redis_client(self, name: str = "cache") -> redis.Redis:
        """Get Redis client."""
        if name not in self.redis_pools:
            raise ValueError(f"Redis connection '{name}' not configured")
        return redis.Redis(connection_pool=self.redis_pools[name])
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all configured databases."""
        health = {
            "status": "healthy",
            "databases": {},
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Check PostgreSQL connections
        for name, engine in self.engines.items():
            try:
                with engine.connect() as conn:
                    conn.execute("SELECT 1")
                health["databases"][f"postgres_{name}"] = {
                    "status": "healthy",
                    "type": "postgresql"
                }
            except Exception as e:
                health["databases"][f"postgres_{name}"] = {
                    "status": "unhealthy",
                    "type": "postgresql",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        # Check Redis connections
        for name, pool in self.redis_pools.items():
            try:
                client = redis.Redis(connection_pool=pool)
                client.ping()
                health["databases"][f"redis_{name}"] = {
                    "status": "healthy",
                    "type": "redis"
                }
            except Exception as e:
                health["databases"][f"redis_{name}"] = {
                    "status": "unhealthy",
                    "type": "redis",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        return health
    
    def close_all(self):
        """Close all database connections."""
        for name, engine in self.engines.items():
            try:
                engine.dispose()
                logger.info(f"Closed PostgreSQL connection: {name}")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL {name}: {e}")
        
        for name, pool in self.redis_pools.items():
            try:
                pool.disconnect()
                logger.info(f"Closed Redis connection: {name}")
            except Exception as e:
                logger.error(f"Error closing Redis {name}: {e}")


# Global database manager instance
db_manager = DatabaseManager()


def initialize_databases():
    """Initialize all database connections from environment variables."""
    try:
        # Primary PostgreSQL (KnowledgeHub)
        primary_config = DatabaseConfig(
            host=os.getenv("DATABASE_HOST", "localhost"),
            port=int(os.getenv("DATABASE_PORT", "5433")),
            database=os.getenv("DATABASE_NAME", "knowledgehub"),
            username=os.getenv("DATABASE_USER", "knowledgehub"),
            password=os.getenv("DATABASE_PASSWORD", "knowledgehub123"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "40")),
            echo=os.getenv("DEBUG", "false").lower() == "true"
        )
        db_manager.configure_postgresql("primary", primary_config)
        
        # TimescaleDB for analytics
        if os.getenv("TIMESCALE_HOST"):
            timescale_config = DatabaseConfig(
                host=os.getenv("TIMESCALE_HOST", "timescale"),
                port=int(os.getenv("TIMESCALE_PORT", "5432")),
                database=os.getenv("TIMESCALE_DATABASE", "knowledgehub_analytics"),
                username=os.getenv("TIMESCALE_USER", "knowledgehub"),
                password=os.getenv("TIMESCALE_PASSWORD", "knowledgehub123"),
                pool_size=int(os.getenv("TIMESCALE_POOL_SIZE", "10")),
                max_overflow=int(os.getenv("TIMESCALE_MAX_OVERFLOW", "20"))
            )
            db_manager.configure_postgresql("timescale", timescale_config)
        
        # Redis cache
        redis_config = RedisConfig(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        )
        db_manager.configure_redis("cache", redis_config)
        
        # Additional Redis for sessions if configured
        if os.getenv("REDIS_SESSION_HOST"):
            session_redis_config = RedisConfig(
                host=os.getenv("REDIS_SESSION_HOST", "redis"),
                port=int(os.getenv("REDIS_SESSION_PORT", "6379")),
                db=int(os.getenv("REDIS_SESSION_DB", "1")),
                password=os.getenv("REDIS_SESSION_PASSWORD")
            )
            db_manager.configure_redis("sessions", session_redis_config)
        
        logger.info("Database connections initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        raise


def get_service_configs() -> Dict[str, Any]:
    """Get configurations for all external services."""
    configs = {}
    
    # Weaviate Vector Database
    if os.getenv("WEAVIATE_URL"):
        configs["weaviate"] = VectorDBConfig(
            url=os.getenv("WEAVIATE_URL", "http://weaviate:8080"),
            collection_name=os.getenv("WEAVIATE_COLLECTION_NAME", "KnowledgeHub"),
            timeout=int(os.getenv("WEAVIATE_TIMEOUT", "60")),
            retries=int(os.getenv("WEAVIATE_RETRIES", "3")),
            batch_size=int(os.getenv("WEAVIATE_BATCH_SIZE", "100"))
        )
    
    # Qdrant Vector Database (alternative)
    if os.getenv("QDRANT_URL"):
        configs["qdrant"] = VectorDBConfig(
            url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "knowledgehub"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "60"))
        )
    
    # Neo4j Graph Database
    if os.getenv("NEO4J_URI"):
        configs["neo4j"] = GraphDBConfig(
            uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "knowledgehub123"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_pool_size=int(os.getenv("NEO4J_MAX_POOL_SIZE", "50")),
            encrypted=os.getenv("NEO4J_ENCRYPTED", "true").lower() == "true"
        )
    
    # Zep Memory Service
    if os.getenv("ZEP_API_URL"):
        configs["zep"] = ZepConfig(
            api_url=os.getenv("ZEP_API_URL", "http://zep:8000"),
            api_key=os.getenv("ZEP_API_KEY"),
            session_window=int(os.getenv("ZEP_SESSION_WINDOW", "12")),
            timeout=int(os.getenv("ZEP_TIMEOUT", "30")),
            max_retries=int(os.getenv("ZEP_MAX_RETRIES", "3"))
        )
    
    # MinIO Object Storage
    if os.getenv("MINIO_ENDPOINT"):
        configs["minio"] = MinIOConfig(
            endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            bucket_name=os.getenv("MINIO_BUCKET_NAME", "knowledgehub"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            region=os.getenv("MINIO_REGION", "us-east-1")
        )
    
    return configs


class DatabaseConnectionPool:
    """Enhanced connection pool management."""
    
    def __init__(self):
        self._pools: Dict[str, Any] = {}
        self._health_checks: Dict[str, bool] = {}
    
    async def get_connection(self, service: str, **kwargs):
        """Get connection with health check and retry logic."""
        if service not in self._pools:
            raise ValueError(f"Service '{service}' not configured")
        
        max_retries = kwargs.get('max_retries', 3)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if service.startswith('postgres'):
                    return db_manager.get_postgres_session(service)
                elif service.startswith('redis'):
                    return db_manager.get_redis_client(service)
                else:
                    raise ValueError(f"Unknown service type: {service}")
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to get {service} connection after {max_retries} retries: {e}")
                    raise
                
                # Wait before retry
                await asyncio.sleep(0.5 * retry_count)
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Comprehensive health check for all services."""
        return await db_manager.health_check()


# Global connection pool
connection_pool = DatabaseConnectionPool()


# Dependency injection functions for FastAPI
def get_postgres_session(name: str = "primary"):
    """FastAPI dependency for PostgreSQL session."""
    session = db_manager.get_postgres_session(name)
    try:
        yield session
    finally:
        session.close()


def get_redis_client(name: str = "cache"):
    """FastAPI dependency for Redis client."""
    return db_manager.get_redis_client(name)


# Migration utilities
def run_migrations(migration_files: List[str]):
    """Run database migrations."""
    session = db_manager.get_postgres_session("primary")
    
    try:
        for migration_file in migration_files:
            logger.info(f"Running migration: {migration_file}")
            
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            
            # Execute migration in chunks (split by ;)
            statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
            
            for statement in statements:
                if statement:
                    session.execute(statement)
            
            session.commit()
            logger.info(f"Migration completed: {migration_file}")
            
    except Exception as e:
        session.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        session.close()


def check_migration_status() -> Dict[str, Any]:
    """Check which migrations have been applied."""
    session = db_manager.get_postgres_session("primary")
    
    try:
        # Check if migration_log table exists
        result = session.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'migration_log'
            )
        """)
        
        if not result.scalar():
            return {"migrations_applied": [], "status": "not_initialized"}
        
        # Get applied migrations
        result = session.execute("""
            SELECT migration_name, completed_at, notes 
            FROM migration_log 
            ORDER BY completed_at
        """)
        
        migrations = [
            {
                "name": row[0],
                "completed_at": row[1].isoformat() if row[1] else None,
                "notes": row[2]
            }
            for row in result.fetchall()
        ]
        
        return {
            "migrations_applied": migrations,
            "status": "ready",
            "total_count": len(migrations)
        }
        
    except Exception as e:
        logger.error(f"Failed to check migration status: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        session.close()