"""Client for remote embeddings service"""

import aiohttp
import asyncio
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """Client for remote embeddings service"""
    
    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._is_available: Optional[bool] = None
        self._last_check: float = 0
        self._check_interval: int = 60  # seconds
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def check_availability(self) -> bool:
        """Check if embeddings service is available"""
        import time
        current_time = time.time()
        
        # Cache availability check
        if self._is_available is not None and (current_time - self._last_check) < self._check_interval:
            return self._is_available
        
        try:
            await self._ensure_session()
            if self.session is None:
                raise Exception("Session not initialized")
            async with self.session.get(f"{self.base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._is_available = data.get("status") == "healthy"
                else:
                    self._is_available = False
        except Exception as e:
            logger.warning(f"Embeddings service not available: {e}")
            self._is_available = False
        
        self._last_check = current_time
        return self._is_available
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        normalize: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for texts using remote service"""
        
        # Check if service is available
        if not await self.check_availability():
            raise Exception("Embeddings service is not available at " + self.base_url)
        
        try:
            await self._ensure_session()
            
            # Call remote service
            if self.session is None:
                raise Exception("Session not initialized")
            async with self.session.post(
                f"{self.base_url}/embeddings",
                json={
                    "texts": texts,
                    "normalize": normalize
                },
                timeout=30
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Generated {len(texts)} embeddings")
                    return data["embeddings"]
                else:
                    error_text = await resp.text()
                    logger.error(f"Embeddings service error: {resp.status} - {error_text}")
                    raise Exception(f"Embeddings service error: {resp.status} - {error_text}")
                    
        except asyncio.TimeoutError:
            logger.error("Embeddings service timeout")
            raise Exception("Embeddings service timeout after 30 seconds")
        except Exception as e:
            logger.error(f"Error calling embeddings service: {e}")
            raise
    
    async def generate_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """Generate embedding for single text"""
        embeddings = await self.generate_embeddings([text], normalize)
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return []
    
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


# Global instance - will be initialized with config URL
embeddings_client = None

def get_embeddings_client():
    """Get or create embeddings client instance"""
    global embeddings_client
    if embeddings_client is None:
        import os
        base_url = os.getenv("EMBEDDINGS_SERVICE_URL", "http://localhost:8100")
        embeddings_client = EmbeddingsClient(base_url)
    return embeddings_client