"""
from ..path_config import MEMORY_SYSTEM_BASE
Memory Sync Service - Bridges local memory system and KnowledgeHub database
"""

import json
import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models import get_db
from ..models.memory import MemoryItem
from ..schemas.memory import MemoryCreate

logger = logging.getLogger(__name__)


class MemorySyncService:
    """Service to synchronize local memory system with KnowledgeHub database"""
    
    def __init__(self, local_memory_path: str = None):
        self.local_memory_path = Path(local_memory_path)
        self.memories_dir = self.local_memory_path / "memories"
        self.synced_entries_file = self.local_memory_path / ".synced_entries.json"
        self.sync_stats = {
            "total_local": 0,
            "total_synced": 0,
            "errors": [],
            "last_sync": None
        }
    
    def _load_synced_entries(self) -> set:
        """Load set of already synced entry IDs"""
        if self.synced_entries_file.exists():
            try:
                with open(self.synced_entries_file, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load synced entries: {e}")
        return set()
    
    def _save_synced_entries(self, synced_ids: set):
        """Save set of synced entry IDs"""
        try:
            with open(self.synced_entries_file, 'w') as f:
                json.dump(list(synced_ids), f)
        except Exception as e:
            logger.error(f"Failed to save synced entries: {e}")
    
    def _get_local_memories(self) -> List[Dict[str, Any]]:
        """Get all memories from local memory system"""
        memories = []
        
        if not self.memories_dir.exists():
            logger.warning(f"Local memory directory not found: {self.memories_dir}")
            return memories
        
        for memory_file in self.memories_dir.glob("*.json"):
            try:
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                    memories.extend(memory_data.get("entries", []))
            except Exception as e:
                logger.error(f"Error reading memory file {memory_file}: {e}")
                self.sync_stats["errors"].append(f"Failed to read {memory_file.name}: {str(e)}")
        
        self.sync_stats["total_local"] = len(memories)
        return memories
    
    def _convert_local_to_db_format(self, local_memory: Dict[str, Any]) -> Optional[MemoryCreate]:
        """Convert local memory format to database format"""
        try:
            content = local_memory.get("content", "")
            memory_id = local_memory.get("id", "")
            
            # Create metadata including original local system fields
            metadata = {
                "source": "local_memory_system",
                "original_id": memory_id,
                "priority": local_memory.get("priority", "MEDIUM"),
                "type": local_memory.get("type", "CONVERSATION"),
                "session_id": local_memory.get("session_id"),
                "file_path": local_memory.get("file_path"),
                "line_number": local_memory.get("line_number"),
                "timestamp": local_memory.get("timestamp"),
                "sync_timestamp": datetime.now().isoformat()
            }
            
            # Add any additional metadata from the local entry
            if "metadata" in local_memory:
                metadata.update(local_memory["metadata"])
            
            # Prepare tags list
            tags = local_memory.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            
            # Add type and priority as tags for easier filtering
            tags.extend([
                f"type:{local_memory.get('type', 'CONVERSATION').lower()}",
                f"priority:{local_memory.get('priority', 'MEDIUM').lower()}",
                "source:local_memory"
            ])
            
            return MemoryCreate(
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                metadata=metadata,
                tags=list(set(tags))  # Remove duplicates
            )
        
        except Exception as e:
            logger.error(f"Error converting local memory to DB format: {e}")
            self.sync_stats["errors"].append(f"Conversion error: {str(e)}")
            return None
    
    
    async def sync_local_to_database(self, batch_size: int = 50) -> Dict[str, Any]:
        """Sync local memories to KnowledgeHub database"""
        logger.info("Starting memory sync from local system to database")
        
        # Get local memories
        local_memories = self._get_local_memories()
        if not local_memories:
            logger.info("No local memories found to sync")
            return self.sync_stats
        
        # Load already synced entries
        synced_ids = self._load_synced_entries()
        
        # Get database session
        db: Session = next(get_db())
        
        try:
            synced_count = 0
            
            for i in range(0, len(local_memories), batch_size):
                batch = local_memories[i:i + batch_size]
                
                for local_memory in batch:
                    memory_id = local_memory.get("id", "")
                    
                    # Skip if already synced
                    if memory_id in synced_ids:
                        continue
                    
                    # Convert to database format
                    db_memory = self._convert_local_to_db_format(local_memory)
                    if not db_memory:
                        continue
                    
                    try:
                        # Check if memory already exists in database
                        existing = db.query(MemoryItem).filter(
                            MemoryItem.content_hash == db_memory.content_hash
                        ).first()
                        
                        if existing:
                            logger.debug(f"Memory with key {db_memory.key} already exists")
                            synced_ids.add(memory_id)
                            continue
                        
                        # Create new memory entry
                        new_memory = MemoryItem(
                            content=db_memory.content,
                            content_hash=db_memory.content_hash,
                            meta_data=db_memory.metadata,
                            tags=db_memory.tags
                        )
                        
                        db.add(new_memory)
                        synced_ids.add(memory_id)
                        synced_count += 1
                        
                        if synced_count % 10 == 0:
                            logger.info(f"Synced {synced_count} memories so far...")
                    
                    except Exception as e:
                        logger.error(f"Error syncing memory {memory_id}: {e}")
                        self.sync_stats["errors"].append(f"Sync error for {memory_id}: {str(e)}")
                        continue
                
                # Commit batch
                try:
                    db.commit()
                    logger.debug(f"Committed batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error committing batch: {e}")
                    db.rollback()
                    self.sync_stats["errors"].append(f"Batch commit error: {str(e)}")
            
            # Save synced entries
            self._save_synced_entries(synced_ids)
            
            # Update stats
            self.sync_stats["total_synced"] = synced_count
            self.sync_stats["last_sync"] = datetime.now().isoformat()
            
            logger.info(f"Memory sync completed: {synced_count} new entries synced")
            
        except Exception as e:
            logger.error(f"Critical error during memory sync: {e}")
            self.sync_stats["errors"].append(f"Critical sync error: {str(e)}")
            db.rollback()
        
        finally:
            db.close()
        
        return self.sync_stats
    
    async def get_unified_memories(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get memories from both local system and database"""
        result = {
            "local_memories": [],
            "database_memories": [],
            "total_count": 0,
            "sources": ["local_memory_system", "knowledgehub_database"]
        }
        
        try:
            # Get local memories
            local_memories = self._get_local_memories()
            result["local_memories"] = local_memories[offset:offset + limit]
            
            # Get database memories
            db: Session = next(get_db())
            try:
                db_memories = db.query(MemoryItem).offset(offset).limit(limit).all()
                result["database_memories"] = [
                    {
                        "id": str(memory.id),
                        "content": memory.content,
                        "content_hash": memory.content_hash,
                        "metadata": memory.meta_data,
                        "tags": memory.tags,
                        "access_count": memory.access_count,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                        "accessed_at": memory.accessed_at.isoformat() if memory.accessed_at else None
                    }
                    for memory in db_memories
                ]
            finally:
                db.close()
            
            result["total_count"] = len(result["local_memories"]) + len(result["database_memories"])
            
        except Exception as e:
            logger.error(f"Error getting unified memories: {e}")
            result["error"] = str(e)
        
        return result
    
    async def search_unified_memories(self, query: str, limit: int = 50) -> Dict[str, Any]:
        """Search memories across both local system and database"""
        result = {
            "query": query,
            "local_matches": [],
            "database_matches": [],
            "total_matches": 0
        }
        
        try:
            # Search local memories
            local_memories = self._get_local_memories()
            query_lower = query.lower()
            
            for memory in local_memories:
                content = memory.get("content", "").lower()
                if query_lower in content:
                    result["local_matches"].append(memory)
                    if len(result["local_matches"]) >= limit // 2:
                        break
            
            # Search database memories
            db: Session = next(get_db())
            try:
                db_memories = db.query(MemoryItem).filter(
                    MemoryItem.content.ilike(f"%{query}%")
                ).limit(limit // 2).all()
                
                result["database_matches"] = [
                    {
                        "id": str(memory.id),
                        "content": memory.content,
                        "content_hash": memory.content_hash,
                        "metadata": memory.meta_data,
                        "tags": memory.tags,
                        "access_count": memory.access_count,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None
                    }
                    for memory in db_memories
                ]
            finally:
                db.close()
            
            result["total_matches"] = len(result["local_matches"]) + len(result["database_matches"])
            
        except Exception as e:
            logger.error(f"Error searching unified memories: {e}")
            result["error"] = str(e)
        
        return result


# Global instance
memory_sync_service = MemorySyncService()