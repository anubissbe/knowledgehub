"""Embeddings service client for GPU-accelerated embeddings via external service"""

import os
import logging
from typing import List, Optional
import httpx
import asyncio

logger = logging.getLogger(__name__)


class EmbeddingServiceClient:
    """Client for external GPU-accelerated embedding service"""
    
    def __init__(self):
        self.embeddings_url = os.getenv("EMBEDDINGS_SERVICE_URL", "http://localhost:8100")
        self.client: Optional[httpx.AsyncClient] = None
        self.model_info = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the embedding service client"""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.embeddings_url,
                timeout=httpx.Timeout(30.0, connect=10.0)
            )
            
            # Check health and get model info
            response = await self.client.get("/health")
            if response.status_code == 200:
                self.model_info = response.json()
                self.initialized = True
                logger.info(f"Embedding service connected - Model: {self.model_info.get('model')}, Device: {self.model_info.get('device')}")
            else:
                logger.error(f"Embedding service health check failed: {response.status_code}")
                raise RuntimeError("Embedding service not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
            
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.initialized:
            raise RuntimeError("Embedding service not initialized")
            
        try:
            response = await self.client.post(
                "/embeddings",
                json={"texts": [text]}
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                if embeddings:
                    return embeddings[0]
                else:
                    logger.error("No embeddings returned")
                    return []
            else:
                logger.error(f"Embedding generation failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.initialized = False
        logger.info("Embedding service client cleaned up")