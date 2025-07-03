"""RAG processor worker main entry point"""

import asyncio
import os
import json
import signal
from typing import Optional, Dict, Any
from datetime import datetime

from .embeddings_remote import EmbeddingService
from .chunker_simple import SmartChunker
from ..shared.config import Config
from ..shared.logging import setup_logging

# Setup logging
logger = setup_logging("rag_processor")


class RAGProcessor:
    """Main RAG processor that handles chunk processing and embeddings"""
    
    def __init__(self):
        self.config = Config()
        self.embedding_service = EmbeddingService()
        self.chunker = SmartChunker()
        self.running = True
        
        # Redis connection for job queue
        import redis.asyncio as redis
        self.redis = redis.from_url(
            self.config.REDIS_URL,
            decode_responses=True
        )
        
        # API client for database operations
        import httpx
        
        # Get API key from environment or use dev key
        api_key = os.getenv("API_KEY", "dev-api-key-123")
        
        self.api_client = httpx.AsyncClient(
            base_url=self.config.API_URL,
            timeout=30.0,
            headers={
                "X-API-Key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        
        # Weaviate client for vector storage
        import weaviate
        self.weaviate_client = weaviate.Client(
            url=self.config.WEAVIATE_URL,
            timeout_config=(5, 30)
        )
    
    async def start(self):
        """Start the RAG processor worker"""
        logger.info("Starting RAG processor...")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize services
        await self.embedding_service.initialize()
        self._ensure_weaviate_schema()
        
        # Main processing loop
        while self.running:
            try:
                # Get chunk from queue
                chunk_data = await self._get_next_chunk()
                
                if chunk_data:
                    await self._process_chunk(chunk_data)
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.5)
                else:
                    # No chunk available, wait
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        # Cleanup
        await self.cleanup()
    
    async def _get_next_chunk(self) -> Optional[dict]:
        """Get next chunk from Redis queue"""
        try:
            # Use BLPOP for blocking pop
            result = await self.redis.blpop(
                ["rag_processing:high", "rag_processing:normal", "rag_processing:low"],
                timeout=5
            )
            
            if result:
                queue, chunk_str = result
                return json.loads(chunk_str)
                
        except Exception as e:
            logger.error(f"Error getting chunk from queue: {e}")
        
        return None
    
    async def _process_chunk(self, chunk_data: dict):
        """Process a single chunk"""
        source_id = chunk_data.get("source_id")
        job_id = chunk_data.get("job_id")
        chunk = chunk_data.get("chunk", {})
        
        logger.info(f"Processing chunk for source {source_id}")
        
        try:
            # Apply smart chunking if needed
            processed_chunks = await self.chunker.process_chunk(chunk)
            
            for processed_chunk in processed_chunks:
                # Generate embedding
                embedding = await self.embedding_service.generate_embedding(
                    processed_chunk["content"]
                )
                
                # Store in vector database
                vector_id = await self._store_in_weaviate(
                    content=processed_chunk["content"],
                    embedding=embedding,
                    metadata={
                        "source_id": source_id,
                        "job_id": job_id,
                        "url": processed_chunk.get("url", ""),
                        "type": processed_chunk.get("type", "text"),
                        **processed_chunk.get("metadata", {})
                    }
                )
                
                # Store chunk metadata in PostgreSQL
                await self._store_chunk_metadata({
                    "source_id": source_id,
                    "content": processed_chunk["content"],
                    "chunk_type": processed_chunk.get("type", "text").upper(),
                    "embedding_id": vector_id,
                    "metadata": {
                        "url": processed_chunk.get("url", ""),
                        "job_id": job_id,
                        **processed_chunk.get("metadata", {})
                    }
                })
                
                logger.debug(f"Chunk stored in Weaviate with ID: {vector_id} and PostgreSQL")
            
            logger.info(f"Processed {len(processed_chunks)} chunks for source {source_id}")
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            # Could implement retry logic here
    
    async def _store_in_weaviate(
        self,
        content: str,
        embedding: list,
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
                    "chunk_id": metadata.get("chunk_id", str(uuid.uuid4())),
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
    
    async def _store_chunk_metadata(self, chunk_data: Dict[str, Any]):
        """Store chunk metadata in PostgreSQL via API"""
        try:
            response = await self.api_client.post(
                "/api/v1/chunks/",
                json=chunk_data
            )
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to store chunk metadata: {response.text}")
            else:
                logger.debug("Chunk metadata stored in PostgreSQL")
                
        except Exception as e:
            logger.error(f"Error storing chunk metadata: {e}")
            # Don't raise - we don't want to stop processing if PostgreSQL storage fails
    
    
    def _ensure_weaviate_schema(self):
        """Ensure Weaviate schema exists"""
        try:
            # Check if schema exists
            schema = self.weaviate_client.schema.get()
            
            chunk_class_exists = any(
                cls["class"] == "Chunk" for cls in schema.get("classes", [])
            )
            
            if not chunk_class_exists:
                # Create schema
                self.weaviate_client.schema.create_class({
                    "class": "Chunk",
                    "description": "Document chunks for semantic search",
                    "vectorizer": "none",  # We provide our own embeddings
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The chunk content"
                        },
                        {
                            "name": "source_id",
                            "dataType": ["string"],
                            "description": "Source ID"
                        },
                        {
                            "name": "url",
                            "dataType": ["string"],
                            "description": "Source URL"
                        },
                        {
                            "name": "type",
                            "dataType": ["string"],
                            "description": "Chunk type"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"],
                            "description": "JSON metadata"
                        },
                        {
                            "name": "indexed_at",
                            "dataType": ["date"],
                            "description": "Indexing timestamp"
                        }
                    ]
                })
                
                logger.info("Created Weaviate schema for Chunk class")
            
        except Exception as e:
            logger.error(f"Error ensuring Weaviate schema: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up RAG processor...")
        
        try:
            await self.embedding_service.cleanup()
            await self.redis.close()
            await self.api_client.aclose()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main entry point"""
    processor = RAGProcessor()
    await processor.start()


if __name__ == "__main__":
    asyncio.run(main())