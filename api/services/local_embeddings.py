"""Local embeddings generation using sentence-transformers"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available - embeddings will return None")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class LocalEmbeddingService:
    """Generate embeddings locally using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialized = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self._initialized = True
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
    
    def generate_embedding(self, text: str, normalize: bool = True) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not self._initialized or self.model is None:
            logger.warning("Embedding model not initialized")
            return None
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, normalize_embeddings=normalize)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_embeddings(self, texts: List[str], normalize: bool = True) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        if not self._initialized or self.model is None:
            logger.warning("Embedding model not initialized")
            return [None] * len(texts)
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(texts, normalize_embeddings=normalize)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)
    
    async def generate_embedding_async(self, text: str, normalize: bool = True) -> Optional[List[float]]:
        """Async wrapper for embedding generation"""
        # Since sentence-transformers is CPU-based, we just call the sync version
        return self.generate_embedding(text, normalize)
    
    async def generate_embeddings_async(self, texts: List[str], normalize: bool = True) -> List[Optional[List[float]]]:
        """Async wrapper for batch embedding generation"""
        return self.generate_embeddings(texts, normalize)


# Global instance
local_embedding_service = LocalEmbeddingService()


def get_local_embedding_service() -> LocalEmbeddingService:
    """Get the global local embedding service instance"""
    return local_embedding_service