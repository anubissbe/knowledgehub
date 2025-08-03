#!/usr/bin/env python3
"""
Memory Sync Service - Handles synchronization between local and distributed storage
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import logging
from collections import defaultdict
import aiosqlite
from sqlalchemy.orm import Session
import redis.asyncio as redis

from ..models.memory import Memory
from ..services.memory_service import MemoryService
from ..database import get_db

logger = logging.getLogger(__name__)


class MemorySyncService:
    """
    Handles bidirectional sync between:
    - Local SQLite (fast, offline)
    - Redis Cache (distributed cache)
    - PostgreSQL (persistent storage)
    """
    
    def __init__(self,
                 local_db_path: str,
                 redis_url: str,
                 sync_interval: int = 300,  # 5 minutes
                 batch_size: int = 100):
        
        self.local_db_path = local_db_path
        self.redis_url = redis_url
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        
        self.redis_client = None
        self.remote_service = MemoryService()
        
        # Sync state
        self.sync_running = False
        self.last_sync = None
        self.sync_stats = defaultdict(int)
        
    async def start(self):
        """Start the sync service"""
        self.redis_client = await redis.from_url(self.redis_url)
        
        # Start sync loop
        asyncio.create_task(self._sync_loop())
        
        logger.info("Memory sync service started")
    
    async def _sync_loop(self):
        """Main sync loop"""
        while True:
            try:
                await self.sync_all()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def sync_all(self):
        """Perform full synchronization"""
        if self.sync_running:
            logger.warning("Sync already in progress, skipping")
            return
        
        self.sync_running = True
        start_time = datetime.utcnow()
        
        try:
            # 1. Upload local changes to remote
            await self._sync_local_to_remote()
            
            # 2. Download remote changes to local
            await self._sync_remote_to_local()
            
            # 3. Clean up old cache entries
            await self._cleanup_cache()
            
            # 4. Update sync metadata
            self.last_sync = datetime.utcnow()
            duration = (self.last_sync - start_time).total_seconds()
            
            logger.info(f"Sync completed in {duration:.2f}s - "
                       f"Uploaded: {self.sync_stats['uploaded']}, "
                       f"Downloaded: {self.sync_stats['downloaded']}")
            
        finally:
            self.sync_running = False
    
    async def _sync_local_to_remote(self):
        """Upload pending local memories to remote"""
        async with aiosqlite.connect(self.local_db_path) as db:
            # Get pending memories
            cursor = await db.execute("""
                SELECT * FROM memories 
                WHERE sync_status = 'pending'
                ORDER BY created_at
                LIMIT ?
            """, (self.batch_size,))
            
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                memory = dict(zip(columns, row))
                
                try:
                    # Upload to remote
                    await self._upload_memory(memory)
                    
                    # Mark as synced
                    await db.execute("""
                        UPDATE memories 
                        SET sync_status = 'synced',
                            sync_timestamp = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (memory['id'],))
                    
                    self.sync_stats['uploaded'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to upload memory {memory['id']}: {e}")
                    
                    # Mark as failed
                    await db.execute("""
                        UPDATE memories 
                        SET sync_status = 'failed',
                            sync_error = ?
                        WHERE id = ?
                    """, (str(e), memory['id']))
            
            await db.commit()
    
    async def _sync_remote_to_local(self):
        """Download new remote memories to local"""
        # Get last sync timestamp
        last_sync = await self._get_last_remote_sync()
        
        # Query remote for new memories
        db = next(get_db())
        try:
            memories = db.query(Memory).filter(
                Memory.created_at > last_sync
            ).limit(self.batch_size).all()
            
            for memory in memories:
                try:
                    await self._download_memory(memory)
                    self.sync_stats['downloaded'] += 1
                except Exception as e:
                    logger.error(f"Failed to download memory {memory.id}: {e}")
            
            # Update last sync timestamp
            if memories:
                await self._set_last_remote_sync(memories[-1].created_at)
                
        finally:
            db.close()
    
    async def _upload_memory(self, memory: Dict):
        """Upload a memory to remote storage"""
        db = next(get_db())
        try:
            # Create remote memory
            # Convert the short hex ID to a proper UUID by padding with zeros
            memory_id = memory['id']
            if len(memory_id) == 16:  # Short hex ID
                # Pad to 32 characters for a valid UUID
                memory_id = memory_id.ljust(32, '0')
                # Format as UUID
                memory_id = f"{memory_id[:8]}-{memory_id[8:12]}-{memory_id[12:16]}-{memory_id[16:20]}-{memory_id[20:32]}"
            
            # Calculate content hash
            import hashlib
            content_hash = hashlib.sha256(memory['content'].encode()).hexdigest()
            
            remote_memory = Memory(
                id=memory_id,
                user_id=memory['user_id'],
                content=memory['content'],
                memory_type=memory.get('type', 'general'),
                meta_data=json.loads(memory.get('metadata', '{}')),
                created_at=datetime.fromisoformat(memory['created_at']),
                session_id='hybrid_sync',  # Use session_id instead of source
                content_hash=content_hash,  # Required field
                context={},  # Default empty context
                tags=[],  # Default empty tags
                knowledge_entities='[]',  # Required JSON field
                knowledge_relations='[]',  # Required JSON field
                related_memories=[],  # Default empty array
                embedding_version='v1.0'  # Default version
            )
            
            # Check if memory already exists
            existing = db.query(Memory).filter(Memory.id == memory_id).first()
            if not existing:
                db.add(remote_memory)
                db.commit()
            else:
                # Update sync status locally without uploading duplicate
                logger.info(f"Memory {memory_id} already exists in remote, marking as synced")
            
            # Also update Redis cache
            await self._update_cache(memory)
            
        finally:
            db.close()
    
    async def _download_memory(self, memory: Memory):
        """Download a memory to local storage"""
        async with aiosqlite.connect(self.local_db_path) as db:
            # Check if already exists
            cursor = await db.execute(
                "SELECT id FROM memories WHERE id = ?", 
                (memory.id,)
            )
            
            if await cursor.fetchone():
                # Update existing
                await db.execute("""
                    UPDATE memories 
                    SET content = ?, 
                        metadata = ?,
                        sync_status = 'synced',
                        sync_timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    memory.content,
                    json.dumps(memory.metadata or {}),
                    memory.id
                ))
            else:
                # Insert new
                await db.execute("""
                    INSERT INTO memories 
                    (id, user_id, content, type, metadata, 
                     created_at, sync_status, sync_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, 'synced', CURRENT_TIMESTAMP)
                """, (
                    memory.id,
                    memory.user_id,
                    memory.content,
                    memory.memory_type,
                    json.dumps(memory.metadata or {}),
                    memory.created_at.isoformat()
                ))
                
                # Update FTS index
                await db.execute("""
                    INSERT INTO memory_fts (id, content, type)
                    VALUES (?, ?, ?)
                """, (memory.id, memory.content, memory.memory_type))
            
            await db.commit()
    
    async def _update_cache(self, memory: Dict):
        """Update Redis cache with memory"""
        cache_key = f"memory:{memory['id']}"
        await self.redis_client.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(memory, default=str)
        )
    
    async def _cleanup_cache(self):
        """Clean up old cache entries"""
        # Remove expired entries (handled by Redis TTL)
        
        # Optionally implement LRU cleanup if needed
        pass
    
    async def _get_last_remote_sync(self) -> datetime:
        """Get timestamp of last remote sync"""
        value = await self.redis_client.get("sync:last_remote")
        if value:
            return datetime.fromisoformat(value.decode())
        
        # Default to 7 days ago
        return datetime.utcnow() - timedelta(days=7)
    
    async def _set_last_remote_sync(self, timestamp: datetime):
        """Set timestamp of last remote sync"""
        await self.redis_client.set(
            "sync:last_remote",
            timestamp.isoformat()
        )
    
    async def force_sync(self, memory_ids: Optional[List[str]] = None):
        """Force sync specific memories or all pending"""
        async with aiosqlite.connect(self.local_db_path) as db:
            if memory_ids:
                # Sync specific memories
                placeholders = ','.join(['?'] * len(memory_ids))
                await db.execute(f"""
                    UPDATE memories 
                    SET sync_status = 'pending'
                    WHERE id IN ({placeholders})
                """, memory_ids)
            else:
                # Mark all failed as pending
                await db.execute("""
                    UPDATE memories 
                    SET sync_status = 'pending'
                    WHERE sync_status = 'failed'
                """)
            
            await db.commit()
        
        # Trigger sync
        await self.sync_all()
    
    async def get_sync_status(self) -> Dict:
        """Get current sync status"""
        async with aiosqlite.connect(self.local_db_path) as db:
            # Count by status
            cursor = await db.execute("""
                SELECT sync_status, COUNT(*) 
                FROM memories 
                GROUP BY sync_status
            """)
            
            status_counts = dict(await cursor.fetchall())
            
            # Get failed memories
            cursor = await db.execute("""
                SELECT id, sync_error, created_at
                FROM memories 
                WHERE sync_status = 'failed'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            failed = await cursor.fetchall()
            
        return {
            "is_syncing": self.sync_running,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "status_counts": status_counts,
            "recent_failures": [
                {"id": f[0], "error": f[1], "created_at": f[2]}
                for f in failed
            ],
            "stats": dict(self.sync_stats)
        }
    
    async def resolve_conflicts(self, strategy: str = "latest"):
        """Resolve sync conflicts"""
        # Implement conflict resolution strategies
        # - latest: Keep most recent version
        # - local: Prefer local version
        # - remote: Prefer remote version
        # - merge: Attempt to merge changes
        
        pass  # TODO: Implement conflict resolution