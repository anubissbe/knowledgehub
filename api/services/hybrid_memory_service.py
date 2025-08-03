#!/usr/bin/env python3
"""
Hybrid Memory Service - Combines local SQLite (Nova-style) with distributed KnowledgeHub
"""

import sqlite3
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import aiosqlite
from sqlalchemy.orm import Session
import redis.asyncio as redis
import logging

from ..models.memory import Memory
from ..services.memory_service import MemoryService
from ..utils.token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


class HybridMemoryService:
    """
    Three-tier memory system:
    L1: SQLite (local, <100ms)
    L2: Redis (cache, 100-500ms)  
    L3: PostgreSQL (persistent, >500ms)
    """
    
    def __init__(self, 
                 local_db_path: str = "~/.knowledgehub/memory.db",
                 redis_url: str = "redis://redis:6379",
                 remote_memory_service: Optional[MemoryService] = None):
        
        self.local_db_path = Path(local_db_path).expanduser()
        self.local_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.redis_url = redis_url
        self.remote_service = remote_memory_service or MemoryService()
        self.token_optimizer = TokenOptimizer()
        
        # Performance metrics
        self.metrics = {
            "local_hits": 0,
            "cache_hits": 0,
            "remote_hits": 0,
            "total_queries": 0,
            "token_savings": 0,
            "token_savings_total": 0
        }
        
        # Initialize on first use
        self._initialized = False
        
    async def initialize(self):
        """Initialize all connections"""
        if self._initialized:
            return
            
        # Create local database
        await self._init_local_db()
        
        # Connect to Redis
        self.redis = await redis.from_url(self.redis_url)
        
        self._initialized = True
        logger.info("Hybrid memory service initialized")
        
    async def _init_local_db(self):
        """Initialize SQLite with Nova-style schema"""
        async with aiosqlite.connect(self.local_db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT,
                    metadata TEXT,
                    embedding TEXT,
                    tokens INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    sync_status TEXT DEFAULT 'pending',
                    sync_error TEXT
                )
            """)
            
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    id, content, type, tokenize='porter'
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    source_id TEXT,
                    target_id TEXT,
                    relationship TEXT,
                    strength REAL DEFAULT 1.0,
                    PRIMARY KEY (source_id, target_id)
                )
            """)
            
            # Indexes for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_user_type ON memories(user_id, type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_sync_status ON memories(sync_status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON memories(accessed_at)")
            
            # Schema migration: Add sync_error column if it doesn't exist
            cursor = await db.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in await cursor.fetchall()]
            if 'sync_error' not in columns:
                await db.execute("ALTER TABLE memories ADD COLUMN sync_error TEXT")
                logger.info("Added sync_error column to memories table")
            
            # Add sync_timestamp column if it doesn't exist
            if 'sync_timestamp' not in columns:
                await db.execute("ALTER TABLE memories ADD COLUMN sync_timestamp TIMESTAMP")
                logger.info("Added sync_timestamp column to memories table")
            
            await db.commit()
    
    async def store(self, 
                    user_id: str,
                    content: str,
                    memory_type: str = "general",
                    metadata: Optional[Dict] = None) -> str:
        """
        Store memory with intelligent routing
        """
        await self.initialize()
        
        # Generate ID
        memory_id = self._generate_id(user_id, content)
        
        # Token optimization
        optimized_content, token_count = self._optimize_content(content)
        
        # Store locally first (instant)
        await self._store_local(
            memory_id, user_id, optimized_content, 
            memory_type, metadata, token_count
        )
        
        # Queue for distributed storage
        await self._queue_sync(memory_id)
        
        # Update metrics
        savings = self.token_optimizer.count_tokens(content) - token_count
        self.metrics["token_savings"] += savings
        self.metrics["token_savings_total"] += savings
        
        return memory_id
    
    async def recall(self, 
                     query: str,
                     user_id: Optional[str] = None,
                     memory_type: Optional[str] = None,
                     limit: int = 10) -> List[Dict]:
        """
        Recall memories with cascade lookup
        """
        await self.initialize()
        self.metrics["total_queries"] += 1
        
        # L1: Try local first
        results = await self._search_local(query, user_id, memory_type, limit)
        if results:
            self.metrics["local_hits"] += 1
            return results
        
        # L2: Check Redis cache
        cache_key = self._cache_key(query, user_id, memory_type)
        cached = await self.redis.get(cache_key)
        if cached:
            self.metrics["cache_hits"] += 1
            results = json.loads(cached)
            # Warm up local cache
            asyncio.create_task(self._warm_local_cache(results))
            return results
        
        # L3: Query remote
        results = await self._search_remote(query, user_id, memory_type, limit)
        if results:
            self.metrics["remote_hits"] += 1
            # Cache for next time
            await self._cache_results(cache_key, results)
            asyncio.create_task(self._warm_local_cache(results))
        
        return results
    
    async def _store_local(self, memory_id: str, user_id: str, 
                          content: str, memory_type: str,
                          metadata: Optional[Dict], tokens: int):
        """Store in local SQLite"""
        async with aiosqlite.connect(self.local_db_path) as db:
            await db.execute("""
                INSERT INTO memories 
                (id, user_id, content, type, metadata, tokens)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory_id, user_id, content, memory_type,
                json.dumps(metadata or {}), tokens
            ))
            
            # Update FTS index
            await db.execute("""
                INSERT INTO memory_fts (id, content, type)
                VALUES (?, ?, ?)
            """, (memory_id, content, memory_type))
            
            await db.commit()
    
    async def _search_local(self, query: str, user_id: Optional[str],
                           memory_type: Optional[str], limit: int) -> List[Dict]:
        """Search local SQLite with FTS5"""
        async with aiosqlite.connect(self.local_db_path) as db:
            # Build query
            sql = """
                SELECT m.*, rank
                FROM memories m
                JOIN (
                    SELECT id, rank FROM memory_fts 
                    WHERE memory_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ) fts ON m.id = fts.id
                WHERE 1=1
            """
            params = [query, limit]
            
            if user_id:
                sql += " AND m.user_id = ?"
                params.append(user_id)
            
            if memory_type:
                sql += " AND m.type = ?"
                params.append(memory_type)
            
            sql += " ORDER BY fts.rank, m.accessed_at DESC"
            
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            
            # Update access stats
            if rows:
                ids = [row[0] for row in rows]
                await db.execute(f"""
                    UPDATE memories 
                    SET accessed_at = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE id IN ({','.join(['?']*len(ids))})
                """, ids)
                await db.commit()
            
            # Convert to dicts
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                # Parse metadata JSON if present
                if 'metadata' in result and isinstance(result['metadata'], str):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                results.append(result)
            return results
    
    def _optimize_content(self, content: str) -> tuple[str, int]:
        """
        Optimize content for token efficiency
        Returns (optimized_content, token_count)
        """
        # Remove redundant whitespace
        optimized = ' '.join(content.split())
        
        # TODO: Implement more sophisticated optimization
        # - Summarization for long content
        # - Reference extraction
        # - Compression algorithms
        
        tokens = self.token_optimizer.count_tokens(optimized)
        return optimized, tokens
    
    def _generate_id(self, user_id: str, content: str) -> str:
        """Generate unique memory ID"""
        data = f"{user_id}:{content}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _cache_key(self, query: str, user_id: Optional[str], 
                   memory_type: Optional[str]) -> str:
        """Generate Redis cache key"""
        parts = ["memory", query]
        if user_id:
            parts.append(f"u:{user_id}")
        if memory_type:
            parts.append(f"t:{memory_type}")
        return ":".join(parts)
    
    async def _queue_sync(self, memory_id: str):
        """Queue memory for distributed sync"""
        await self.redis.rpush("memory:sync:queue", memory_id)
    
    async def _cache_results(self, key: str, results: List[Dict]):
        """Cache results in Redis"""
        await self.redis.setex(
            key, 
            3600,  # 1 hour TTL
            json.dumps(results, default=str)
        )
    
    async def _warm_local_cache(self, memories: List[Dict]):
        """Warm local cache with remote results"""
        # TODO: Implement selective caching based on access patterns
        pass
    
    async def _search_remote(self, query: str, user_id: Optional[str],
                            memory_type: Optional[str], limit: int) -> List[Dict]:
        """Search remote KnowledgeHub"""
        # TODO: Implement remote search via API
        return []
    
    async def sync_pending(self):
        """Sync pending memories to distributed storage"""
        async with aiosqlite.connect(self.local_db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM memories 
                WHERE sync_status = 'pending' 
                LIMIT 100
            """)
            
            pending = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for row in pending:
                memory = dict(zip(columns, row))
                
                try:
                    # Sync to remote
                    # TODO: Implement actual remote sync
                    
                    # Mark as synced
                    await db.execute("""
                        UPDATE memories 
                        SET sync_status = 'synced' 
                        WHERE id = ?
                    """, (memory['id'],))
                    
                except Exception as e:
                    logger.error(f"Sync failed for {memory['id']}: {e}")
            
            await db.commit()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_queries = self.metrics["total_queries"] or 1
        
        return {
            "cache_hit_rate": (self.metrics["local_hits"] + self.metrics["cache_hits"]) / total_queries,
            "local_hit_rate": self.metrics["local_hits"] / total_queries,
            "token_savings_total": self.metrics["token_savings"],
            **self.metrics
        }
    
    # Nova-style workflow features
    
    async def track_workflow(self, project_id: str, phase: str, context: Optional[str] = None):
        """Track project workflow phases"""
        async with aiosqlite.connect(self.local_db_path) as db:
            workflow_id = self._generate_id(project_id, phase)
            
            await db.execute("""
                INSERT INTO workflows (id, project_id, phase, context)
                VALUES (?, ?, ?, ?)
            """, (workflow_id, project_id, phase, context))
            
            await db.commit()
    
    async def map_relationships(self, source_id: str, target_id: str, 
                               relationship: str, strength: float = 1.0):
        """Map relationships between memories"""
        async with aiosqlite.connect(self.local_db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO relationships 
                (source_id, target_id, relationship, strength)
                VALUES (?, ?, ?, ?)
            """, (source_id, target_id, relationship, strength))
            
            await db.commit()