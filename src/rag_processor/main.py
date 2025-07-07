"""Fixed RAG processor worker main entry point"""

import asyncio
import os
import json
import signal
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

from .embeddings_client import EmbeddingServiceClient  # Use external embeddings service
from .chunker import SmartChunker
from .health_server import HealthServer
from ..shared.config import Config
from ..shared.logging import setup_logging

import httpx
import weaviate
from redis import asyncio as aioredis
import uuid

# Setup logging
logger = setup_logging("rag_processor")


class RAGProcessor:
    """Main RAG processor that handles chunk processing and embeddings"""
    
    def __init__(self):
        self.config = Config()
        self.embedding_service = EmbeddingServiceClient()
        self.chunker = SmartChunker()
        self.health_port = int(os.getenv("HEALTH_CHECK_PORT", "3013"))
        self.health_server = HealthServer(self.health_port)
        self.redis_client: Optional[aioredis.Redis] = None
        self.weaviate_client: Optional[weaviate.Client] = None
        self.api_client: Optional[httpx.AsyncClient] = None
        self.running = False
        
        # Rate limiting configuration - increased for production workload
        self.rate_limit_requests = 500  # Max requests per minute (increased 10x)
        self.rate_limit_window = 60     # Window in seconds
        self.request_times = []         # Track request timestamps
        self.batch_size = 50            # Process chunks in larger batches (increased 5x)
        self.batch_delay = 0.5          # Reduced delay between batches for faster processing
        
    async def initialize(self):
        """Initialize all connections and services"""
        logger.info("Initializing RAG processor...")
        
        # Initialize Redis
        self.redis_client = await aioredis.from_url(
            self.config.REDIS_URL,
            decode_responses=True
        )
        logger.info("Redis connection established")
        
        # Initialize Weaviate
        weaviate_url = os.getenv("WEAVIATE_URL", "http://knowledgehub-weaviate:8080")
        self.weaviate_client = weaviate.Client(weaviate_url)
        self._ensure_weaviate_schema()
        logger.info("Weaviate connection established")
        
        # Initialize embedding service
        await self.embedding_service.initialize()
        logger.info("Embedding service initialized - GPU acceleration enabled")
        
        # Initialize API client
        api_url = os.getenv("API_URL", "http://api:3000")
        api_key = os.getenv("API_KEY", "dev-api-key-123")
        self.api_client = httpx.AsyncClient(
            base_url=api_url,
            headers={"X-API-Key": api_key},
            timeout=30.0
        )
        logger.info("API client initialized")
        
        # Write health file
        with open('/tmp/health', 'w') as f:
            f.write('healthy')
        
        # Start health server
        await self.health_server.start()
        
        # No consumer group needed for list-based queue
        
        self.running = True
        logger.info("RAG processor initialization complete")
        
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up RAG processor...")
        self.running = False
        
        # Remove health file
        try:
            os.remove('/tmp/health')
        except FileNotFoundError:
            pass
        
        # Stop health server
        await self.health_server.stop()
            
        if self.embedding_service:
            await self.embedding_service.cleanup()
            
        if self.redis_client:
            await self.redis_client.close()
            
        if self.api_client:
            await self.api_client.aclose()
            
        logger.info("RAG processor cleanup complete")
        
    async def run(self):
        """Main processing loop with batching"""
        logger.info("Starting RAG processor main loop with batching...")
        
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Read from Redis list (blocking pop with short timeout for batching)
                result = await self.redis_client.brpop(
                    "rag_processing:normal",
                    timeout=0.1  # Reduced timeout for faster batch formation
                )
                
                if result:
                    _, message = result
                    chunk_data = json.loads(message)
                    batch.append(chunk_data)
                
                # Process batch if it's full or enough time has passed
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and current_time - last_batch_time >= self.batch_delay)
                )
                
                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Remove old timestamps outside the window
        self.request_times = [
            t for t in self.request_times 
            if current_time - t < self.rate_limit_window
        ]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit_requests:
            # Calculate how long to wait
            oldest_request = min(self.request_times)
            wait_time = self.rate_limit_window - (current_time - oldest_request) + 0.1
            
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                # Recursive call to recheck after waiting
                await self._check_rate_limit()
        
        # Record this request
        self.request_times.append(current_time)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of chunks efficiently"""
        logger.info(f"Processing batch of {len(batch)} chunks")
        
        # Collect all chunks to be created
        chunks_to_create = []
        chunk_to_batch_idx = {}  # Map chunk data to batch index
        
        try:
            # Phase 1: Prepare all chunks
            for idx, chunk_data in enumerate(batch):
                source_id = chunk_data.get("source_id")
                job_id = chunk_data.get("job_id")
                chunk = chunk_data.get("chunk", {})
                
                # Apply smart chunking if needed
                processed_chunks = await self.chunker.process_chunk(chunk)
                
                for processed_chunk in processed_chunks:
                    chunk_metadata = {
                        "source_id": source_id,
                        "content": processed_chunk["content"],
                        "chunk_type": processed_chunk.get("type", "text").lower(),
                        "metadata": {
                            "url": processed_chunk.get("url", ""),
                            "job_id": job_id,
                            **processed_chunk.get("metadata", {})
                        }
                    }
                    chunks_to_create.append(chunk_metadata)
                    chunk_to_batch_idx[len(chunks_to_create) - 1] = (idx, processed_chunk)
            
            # Phase 2: Batch create chunks in PostgreSQL
            if chunks_to_create:
                created_chunks = await self._store_chunks_batch(chunks_to_create)
                
                # Phase 3: Generate embeddings and store in Weaviate
                embedding_updates = []
                
                for idx, (chunk_id, chunk_data) in enumerate(created_chunks):
                    batch_idx, processed_chunk = chunk_to_batch_idx[idx]
                    source_id = batch[batch_idx].get("source_id")
                    job_id = batch[batch_idx].get("job_id")
                    
                    # Generate embedding using GPU acceleration
                    embedding = await self.embedding_service.generate_embedding(
                        processed_chunk["content"]
                    )
                    
                    # Store in Weaviate
                    vector_id = await self._store_in_weaviate(
                        content=processed_chunk["content"],
                        embedding=embedding,
                        metadata={
                            "source_id": source_id,
                            "chunk_id": chunk_id,
                            "document_id": chunk_data.get("document_id", ""),
                            "job_id": job_id,
                            "url": processed_chunk.get("url", ""),
                            "type": processed_chunk.get("type", "text"),
                            **processed_chunk.get("metadata", {})
                        }
                    )
                    
                    embedding_updates.append({
                        "chunk_id": chunk_id,
                        "embedding_id": vector_id
                    })
                    
                    logger.info(f"Chunk {chunk_id} created with GPU-accelerated embedding (vector_id: {vector_id})")
                
                # Phase 4: Update chunk embedding IDs
                if embedding_updates:
                    await self._update_chunk_embedding_ids_batch(embedding_updates)
                
            logger.info(f"Successfully processed batch of {len(batch)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Fall back to individual processing
            for chunk_data in batch:
                await self._process_chunk(chunk_data)
        
        # Add minimal delay between batches
        await asyncio.sleep(0.1)  # Reduced from 0.5 for faster processing
                
    async def _process_chunk(self, chunk_data: Dict[str, Any]):
        """Process a single chunk"""
        source_id = chunk_data.get("source_id")
        job_id = chunk_data.get("job_id")
        chunk = chunk_data.get("chunk", {})
        
        logger.info(f"Processing chunk for source {source_id}")
        
        try:
            # Apply smart chunking if needed
            processed_chunks = await self.chunker.process_chunk(chunk)
            
            for processed_chunk in processed_chunks:
                # First create chunk in PostgreSQL to get ID
                chunk_metadata = {
                    "source_id": source_id,
                    "content": processed_chunk["content"],
                    "chunk_type": processed_chunk.get("type", "text").lower(),
                    "metadata": {
                        "url": processed_chunk.get("url", ""),
                        "job_id": job_id,
                        **processed_chunk.get("metadata", {})
                    }
                }
                
                # Store in PostgreSQL and get the chunk ID
                chunk_id = await self._store_chunk_metadata(chunk_metadata)
                
                if chunk_id:
                    # Generate embedding using GPU acceleration
                    embedding = await self.embedding_service.generate_embedding(
                        processed_chunk["content"]
                    )
                    
                    # Store in vector database with the PostgreSQL chunk ID
                    vector_id = await self._store_in_weaviate(
                        content=processed_chunk["content"],
                        embedding=embedding,
                        metadata={
                            "source_id": source_id,
                            "chunk_id": chunk_id,  # Use the PostgreSQL chunk ID
                            "document_id": chunk_metadata.get("document_id", ""),
                            "job_id": job_id,
                            "url": processed_chunk.get("url", ""),
                            "type": processed_chunk.get("type", "text"),
                            **processed_chunk.get("metadata", {})
                        }
                    )
                    
                    # Update chunk with embedding_id
                    await self._update_chunk_embedding_id(chunk_id, vector_id)
                    
                    logger.info(f"Chunk {chunk_id} stored in Weaviate with GPU-accelerated embedding (vector_id: {vector_id})")
            
            logger.info(f"Processed {len(processed_chunks)} chunks for source {source_id}")
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            # Don't raise - continue processing other chunks
            
    async def _store_in_weaviate(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> str:
        """Store content and embedding in Weaviate"""
        try:
            # Create object in Weaviate
            result = self.weaviate_client.data_object.create(
                class_name="Knowledge_chunks",
                data_object={
                    "content": content,
                    "source_id": metadata.get("source_id"),
                    "chunk_id": metadata.get("chunk_id"),  # This is now the PostgreSQL chunk ID
                    "document_id": metadata.get("document_id", ""),
                    "chunk_type": metadata.get("type", "text"),
                    "url": metadata.get("url", ""),
                    "metadata": json.dumps(metadata),
                    "created_at": datetime.utcnow().isoformat() + "Z"
                },
                vector=embedding
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing in Weaviate: {e}")
            raise
    
    async def _store_chunk_metadata(self, chunk_data: Dict[str, Any]) -> Optional[str]:
        """Store chunk metadata in PostgreSQL via API and return chunk ID"""
        try:
            # Apply rate limiting
            await self._check_rate_limit()
            
            response = await self.api_client.post(
                "/api/v1/chunks/",
                json=chunk_data
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                chunk_id = result.get("id")
                logger.debug(f"Chunk metadata stored in PostgreSQL with ID: {chunk_id}")
                return chunk_id
            else:
                logger.error(f"Failed to store chunk metadata: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing chunk metadata: {e}")
            return None
    
    async def _update_chunk_embedding_id(self, chunk_id: str, embedding_id: str):
        """Update chunk with embedding ID"""
        try:
            # Apply rate limiting
            await self._check_rate_limit()
            
            response = await self.api_client.patch(
                f"/api/v1/chunks/{chunk_id}",
                json={"embedding_id": embedding_id}
            )
            
            if response.status_code not in [200, 204]:
                logger.warning(f"Failed to update chunk embedding_id: {response.text}")
                
        except Exception as e:
            logger.warning(f"Error updating chunk embedding_id: {e}")
            # Don't fail if update fails
    
    async def _store_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[tuple]:
        """Store multiple chunks in PostgreSQL via batch API"""
        try:
            # Apply rate limiting
            await self._check_rate_limit()
            
            response = await self.api_client.post(
                "/api/v1/chunks/batch",
                json=chunks
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                created_chunks = []
                
                # Return list of (chunk_id, chunk_data) tuples
                for chunk_info in result.get("chunks", []):
                    created_chunks.append((
                        chunk_info["id"],
                        chunk_info
                    ))
                
                logger.info(f"Batch stored {len(created_chunks)} chunks in PostgreSQL")
                return created_chunks
            else:
                logger.error(f"Failed to store chunks batch: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error storing chunks batch: {e}")
            return []
    
    async def _update_chunk_embedding_ids_batch(self, updates: List[Dict[str, str]]):
        """Update multiple chunks with embedding IDs in batch"""
        try:
            # Apply rate limiting
            await self._check_rate_limit()
            
            response = await self.api_client.patch(
                "/api/v1/chunks/batch/embedding-ids",
                json=updates
            )
            
            if response.status_code not in [200, 204]:
                logger.warning(f"Failed to update chunk embedding_ids batch: {response.text}")
                
        except Exception as e:
            logger.warning(f"Error updating chunk embedding_ids batch: {e}")
            # Don't fail if update fails
    
    def _ensure_weaviate_schema(self):
        """Ensure Weaviate schema exists"""
        try:
            # Check if schema exists
            schema = self.weaviate_client.schema.get()
            
            chunk_class_exists = any(
                cls["class"] == "Knowledge_chunks" for cls in schema.get("classes", [])
            )
            
            if not chunk_class_exists:
                logger.warning("Knowledge_chunks class not found in Weaviate")
                # Don't create it - let the API handle schema creation
                
        except Exception as e:
            logger.error(f"Error checking Weaviate schema: {e}")


async def main():
    """Main entry point"""
    processor = RAGProcessor()
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        processor.running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await processor.initialize()
        await processor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())