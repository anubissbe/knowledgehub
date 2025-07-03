"""Remote embedding generation service"""

import asyncio
import os
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import hashlib
import json

import httpx

from ..shared.config import Config
from ..shared.logging import setup_logging

logger = setup_logging("embeddings")


class EmbeddingService:
    """Service for generating text embeddings using remote service"""
    
    def __init__(self):
        self.config = Config()
        
        # Remote embeddings service URL
        self.embeddings_url = os.getenv(
            "EMBEDDINGS_SERVICE_URL",
            "http://embeddings-real:8100"
        )
        
        # Cache for embeddings
        self.cache: Dict[str, List[float]] = {}
        self.cache_size = 1000
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self):
        """Initialize the embedding service"""
        logger.info(f"Initializing remote embedding service at: {self.embeddings_url}")
        
        # Create HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Test connection
        try:
            response = await self.client.get(f"{self.embeddings_url}/health")
            if response.status_code == 200:
                logger.info("Successfully connected to embeddings service")
            else:
                logger.warning(f"Embeddings service returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to embeddings service: {e}")
            raise
    
    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """Generate embedding for text"""
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                logger.debug("Returning cached embedding")
                return self.cache[cache_key]
        
        # Generate embedding via remote service
        try:
            response = await self.client.post(
                f"{self.embeddings_url}/embeddings",
                json={
                    "texts": [text],
                    "normalize": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data["embeddings"][0]
                
                # Cache result
                if use_cache and len(self.cache) < self.cache_size:
                    self.cache[cache_key] = embedding
                
                return embedding
            else:
                raise Exception(f"Embeddings service error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await self.client.post(
                    f"{self.embeddings_url}/embeddings",
                    json={
                        "texts": batch,
                        "normalize": True
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embeddings.extend(data["embeddings"])
                else:
                    raise Exception(f"Embeddings service error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()
        
        self.cache.clear()


class EmbeddingOptimizer:
    """Optimizer for embedding generation and storage"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    async def optimize_text_for_embedding(
        self,
        text: str,
        max_length: int = 512
    ) -> str:
        """Optimize text for embedding generation"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Truncate if too long
        if len(text) > max_length:
            # Try to truncate at sentence boundary
            sentences = text.split(". ")
            optimized = ""
            
            for sentence in sentences:
                if len(optimized) + len(sentence) <= max_length:
                    optimized += sentence + ". "
                else:
                    break
            
            text = optimized.strip()
            if not text:
                # Fallback to simple truncation
                text = text[:max_length]
        
        return text
    
    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    async def deduplicate_by_similarity(
        self,
        texts: List[str],
        threshold: float = 0.95
    ) -> List[int]:
        """Find duplicate texts by embedding similarity"""
        # Generate embeddings for all texts
        embeddings = await self.embedding_service.generate_embeddings_batch(texts)
        
        # Find duplicates
        duplicates = []
        n = len(embeddings)
        
        for i in range(n):
            if i in duplicates:
                continue
                
            for j in range(i + 1, n):
                if j in duplicates:
                    continue
                
                similarity = self.calculate_similarity(
                    embeddings[i],
                    embeddings[j]
                )
                
                if similarity >= threshold:
                    duplicates.append(j)
        
        return duplicates