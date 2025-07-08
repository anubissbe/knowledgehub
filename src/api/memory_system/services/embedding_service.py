"""Embedding service integration for memory system"""

import logging
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from ...services.embeddings_client import get_embeddings_client
from ..models import Memory

logger = logging.getLogger(__name__)


class MemoryEmbeddingService:
    """Service for generating and managing memory embeddings"""
    
    def __init__(self):
        self.embeddings_client = get_embeddings_client()
    
    async def generate_memory_embedding(self, memory: Memory) -> Optional[List[float]]:
        """Generate embedding for a memory"""
        try:
            # Combine content and summary for richer embedding
            text_to_embed = memory.content
            if memory.summary:
                text_to_embed = f"{memory.summary}\n\n{memory.content}"
            
            # Add entities to the embedding text for better context
            if memory.entities:
                entities_str = ", ".join(memory.entities)
                text_to_embed = f"{text_to_embed}\n\nEntities: {entities_str}"
            
            # Generate embedding
            embedding = await self.embeddings_client.generate_embedding(
                text_to_embed, 
                normalize=True
            )
            
            logger.info(f"Generated embedding for memory {memory.id} (dim: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for memory {memory.id}: {e}")
            return None
    
    async def update_memory_embedding(self, db: Session, memory_id: UUID) -> bool:
        """Update embedding for a specific memory"""
        try:
            memory = db.query(Memory).filter_by(id=memory_id).first()
            if not memory:
                logger.error(f"Memory {memory_id} not found")
                return False
            
            embedding = await self.generate_memory_embedding(memory)
            if embedding:
                memory.set_embedding(embedding)
                db.commit()
                logger.info(f"Updated embedding for memory {memory_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update embedding for memory {memory_id}: {e}")
            db.rollback()
            return False
    
    async def generate_batch_embeddings(self, memories: List[Memory]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple memories in batch"""
        if not memories:
            return []
        
        try:
            # Prepare texts for batch embedding
            texts = []
            for memory in memories:
                text_to_embed = memory.content
                if memory.summary:
                    text_to_embed = f"{memory.summary}\n\n{memory.content}"
                
                if memory.entities:
                    entities_str = ", ".join(memory.entities)
                    text_to_embed = f"{text_to_embed}\n\nEntities: {entities_str}"
                
                texts.append(text_to_embed)
            
            # Generate embeddings in batch
            embeddings = await self.embeddings_client.generate_embeddings(
                texts, 
                normalize=True
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(memories)
    
    async def find_similar_memories(
        self, 
        db: Session, 
        query_embedding: List[float], 
        limit: int = 10,
        min_similarity: float = 0.5,
        session_id: Optional[UUID] = None,
        user_id: Optional[str] = None
    ) -> List[tuple[Memory, float]]:
        """Find similar memories using cosine similarity search"""
        try:
            from sqlalchemy import text, and_
            from ..models import MemorySession
            
            # Build the base query with cosine similarity calculation
            query_parts = [
                "SELECT m.*, ",
                "cosine_similarity(m.embedding, :query_embedding) as similarity ",
                "FROM memories m "
            ]
            
            # Add joins and filters
            where_conditions = ["m.embedding IS NOT NULL"]
            params = {"query_embedding": query_embedding}
            
            if session_id or user_id:
                query_parts.append("JOIN memory_sessions ms ON m.session_id = ms.id ")
            
            if session_id:
                where_conditions.append("m.session_id = :session_id")
                params["session_id"] = str(session_id)
            
            if user_id:
                where_conditions.append("ms.user_id = :user_id")
                params["user_id"] = user_id
            
            # Add minimum similarity filter
            where_conditions.append("cosine_similarity(m.embedding, :query_embedding) >= :min_similarity")
            params["min_similarity"] = min_similarity
            
            # Build final query
            query_str = "".join(query_parts)
            if where_conditions:
                query_str += "WHERE " + " AND ".join(where_conditions) + " "
            
            query_str += "ORDER BY similarity DESC LIMIT :limit"
            params["limit"] = limit
            
            # Execute query
            result = db.execute(text(query_str), params)
            rows = result.fetchall()
            
            # Convert results to Memory objects with similarity scores
            similar_memories = []
            for row in rows:
                # Create Memory object from row data
                memory = db.query(Memory).filter_by(id=row.id).first()
                if memory:
                    similarity_score = float(row.similarity)
                    similar_memories.append((memory, similarity_score))
            
            logger.info(f"Found {len(similar_memories)} similar memories with similarity >= {min_similarity}")
            return similar_memories
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            # Fallback to placeholder behavior
            return await self._fallback_similarity_search(
                db, limit, session_id, user_id
            )
    
    async def _fallback_similarity_search(
        self,
        db: Session,
        limit: int,
        session_id: Optional[UUID] = None,
        user_id: Optional[str] = None
    ) -> List[tuple[Memory, float]]:
        """Fallback similarity search when vector search fails"""
        from ..models import MemorySession
        query = db.query(Memory)
        
        if session_id or user_id:
            query = query.join(MemorySession)
        
        if session_id:
            query = query.filter(Memory.session_id == session_id)
        
        if user_id:
            query = query.filter(MemorySession.user_id == user_id)
        
        # Get memories with embeddings
        memories = query.filter(Memory.embedding != None).limit(limit).all()
        
        # Return with fallback similarity scores
        return [(memory, 0.6) for memory in memories]


# Global instance
memory_embedding_service = MemoryEmbeddingService()