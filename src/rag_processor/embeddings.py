"""Embedding generation service"""

import asyncio
import os
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import hashlib
import json

import httpx
from sentence_transformers import SentenceTransformer
import torch

from ..shared.config import Config
from ..shared.logging import setup_logging

logger = setup_logging("embeddings")


class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self):
        self.config = Config()
        self.model: Optional[SentenceTransformer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model configuration
        self.model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Cache for embeddings
        self.cache: Dict[str, List[float]] = {}
        self.cache_size = 1000
        
        # OpenAI client for fallback
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    async def initialize(self):
        """Initialize the embedding service"""
        logger.info(f"Initializing embedding service with model: {self.model_name}")
        
        try:
            # Load local model
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info(f"Loaded model on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            if self.openai_api_key:
                logger.info("Falling back to OpenAI embeddings")
            else:
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
        
        # Generate embedding
        try:
            if self.model:
                embedding = await self._generate_local_embedding(text)
            else:
                embedding = await self._generate_openai_embedding(text)
            
            # Cache result
            if use_cache and len(self.cache) < self.cache_size:
                self.cache[cache_key] = embedding
            
            return embedding
            
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
            
            if self.model:
                batch_embeddings = await self._generate_local_embeddings_batch(batch)
            else:
                # For OpenAI, process one by one (they have their own batching)
                batch_embeddings = []
                for text in batch:
                    embedding = await self._generate_openai_embedding(text)
                    batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _generate_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.model.encode(text, convert_to_numpy=True)
        )
        
        return embedding.tolist()
    
    async def _generate_local_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for batch using local model"""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, convert_to_numpy=True)
        )
        
        return [emb.tolist() for emb in embeddings]
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": "text-embedding-ada-002"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")
            
            data = response.json()
            return data["data"][0]["embedding"]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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