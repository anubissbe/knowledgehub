"""Embedding service integration for memory system"""

import logging
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from ...services.embeddings_client import get_embeddings_client
from ...services.local_embeddings import get_local_embedding_service
from ..models import MemorySystemMemory

logger = logging.getLogger(__name__)


class MemoryEmbeddingService:
    """Service for generating and managing memory embeddings"""
    
    def __init__(self):
        try:
            self.embeddings_client = get_embeddings_client()
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings client: {e}")
            self.embeddings_client = None
        
        # Always initialize local service as fallback
        self.local_service = get_local_embedding_service()
    
    async def generate_embedding(self, text: str, normalize: bool = True) -> Optional[List[float]]:
        """Generate embedding for text with fallback support"""
        # Try remote service first
        if self.embeddings_client:
            try:
                # Check if service is available
                if await self.embeddings_client.check_availability():
                    return await self.embeddings_client.generate_embedding(text, normalize=normalize)
            except Exception as e:
                logger.warning(f"Remote embeddings service failed: {e}")
        
        # Fallback to local service
        logger.info("Using local embedding service")
        try:
            return await self.local_service.generate_embedding_async(text, normalize=normalize)
        except Exception as e:
            logger.error(f"Failed to generate embedding with local service: {e}")
            return None
    
    async def generate_memory_embedding(self, memory: MemorySystemMemory) -> Optional[List[float]]:
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
            
            # Generate embedding using the method with fallback
            embedding = await self.generate_embedding(
                text_to_embed, 
                normalize=True
            )
            
            if embedding:
                logger.info(f"Generated embedding for memory {memory.id} (dim: {len(embedding)})")
            else:
                logger.warning(f"Failed to generate embedding for memory {memory.id}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for memory {memory.id}: {e}")
            return None
    
    async def update_memory_embedding(self, db: Session, memory_id: UUID) -> bool:
        """Update embedding for a specific memory"""
        try:
            memory = db.query(MemorySystemMemory).filter_by(id=memory_id).first()
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
    
    async def generate_batch_embeddings(self, memories: List[MemorySystemMemory]) -> List[Optional[List[float]]]:
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
    ) -> List[tuple[MemorySystemMemory, float]]:
        """Find similar memories using cosine similarity search in Python"""
        try:
            import numpy as np
            from ..models import MemorySession
            
            # Build query
            query = db.query(MemorySystemMemory)
            
            # Add filters
            if session_id or user_id:
                query = query.join(MemorySession)
            
            if session_id:
                query = query.filter(MemorySystemMemory.session_id == session_id)
            
            if user_id:
                query = query.filter(MemorySession.user_id == user_id)
            
            # Only get memories with embeddings
            memories_with_embeddings = query.filter(MemorySystemMemory.embedding != None).all()
            
            if not memories_with_embeddings:
                logger.info("No memories with embeddings found")
                return []
            
            # Convert query embedding to numpy array
            query_vec = np.array(query_embedding)
            query_norm = np.linalg.norm(query_vec)
            
            if query_norm == 0:
                logger.warning("Query embedding has zero norm")
                return []
            
            # Calculate similarities
            results = []
            for memory in memories_with_embeddings:
                if memory.embedding and len(memory.embedding) == len(query_embedding):
                    # Calculate cosine similarity
                    memory_vec = np.array(memory.embedding)
                    memory_norm = np.linalg.norm(memory_vec)
                    
                    if memory_norm > 0:
                        similarity = np.dot(query_vec, memory_vec) / (query_norm * memory_norm)
                        
                        if similarity >= min_similarity:
                            results.append((memory, float(similarity)))
            
            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply limit
            results = results[:limit]
            
            logger.info(f"Found {len(results)} similar memories with similarity >= {min_similarity}")
            return results
            
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
    ) -> List[tuple[MemorySystemMemory, float]]:
        """Fallback similarity search when vector search fails"""
        from ..models import MemorySession
        query = db.query(MemorySystemMemory)
        
        if session_id or user_id:
            query = query.join(MemorySession)
        
        if session_id:
            query = query.filter(MemorySystemMemory.session_id == session_id)
        
        if user_id:
            query = query.filter(MemorySession.user_id == user_id)
        
        # Get memories with embeddings
        memories = query.filter(MemorySystemMemory.embedding != None).limit(limit).all()
        
        # Return with fallback similarity scores
        return [(memory, 0.6) for memory in memories]


# Global instance
memory_embedding_service = MemoryEmbeddingService()