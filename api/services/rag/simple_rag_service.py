"""
Simple RAG Service with minimal dependencies
Uses existing infrastructure and makes LlamaIndex optional
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import asyncio
import json

# Core imports that should already exist
from ...models.document import Document as KHDocument, DocumentChunk
from ...services.embedding_service import EmbeddingService
from ...services.vector_store import VectorStore
from ...services.search_service import SearchService
from ...services.real_ai_intelligence import RealAIIntelligence
from ...services import CacheService
from ...database import get_db_session
from ...config import settings

logger = logging.getLogger(__name__)

# Try to import LlamaIndex components, but make them optional
LLAMAINDEX_AVAILABLE = False
try:
    from llama_index.core import (
        VectorStoreIndex,
        Document,
        StorageContext,
        Settings,
        QueryBundle
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMAINDEX_AVAILABLE = True
    logger.info("LlamaIndex components available")
except ImportError as e:
    logger.warning(f"LlamaIndex not available: {e}. Using fallback implementation.")


class SimpleRAGService:
    """
    Simplified RAG service that works with existing infrastructure
    Falls back gracefully when LlamaIndex is not available
    """
    
    def __init__(self):
        self.logger = logger
        self.cache = CacheService(settings.REDIS_URL)
        self.ai_intelligence = RealAIIntelligence()
        
        # Always use existing services
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(
            url=settings.WEAVIATE_URL,
            collection_name=settings.WEAVIATE_COLLECTION_NAME
        )
        self.search_service = SearchService()
        
        # Initialize LlamaIndex if available
        self.llamaindex_initialized = False
        if LLAMAINDEX_AVAILABLE:
            try:
                self._init_llamaindex()
                self.llamaindex_initialized = True
            except Exception as e:
                logger.warning(f"Could not initialize LlamaIndex: {e}")
        
    def _init_llamaindex(self):
        """Initialize LlamaIndex components if available"""
        # Use existing embedding model configuration
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL or "BAAI/bge-base-en-v1.5",
            cache_folder="/tmp/embeddings_cache"
        )
        
        # Configure settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Initialize simple node parser
        self.text_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        logger.info("LlamaIndex components initialized")
        
    async def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        source_type: str = "documentation",
        use_contextual_enrichment: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a document using existing infrastructure
        """
        try:
            # Create chunks using existing chunking logic
            chunks = await self._create_chunks(content, metadata)
            
            # Enrich chunks if requested
            if use_contextual_enrichment:
                chunks = await self._enrich_chunks(chunks, content, metadata)
            
            # Store using existing vector store service
            stored_chunks = []
            for chunk in chunks:
                # Generate embedding using existing service
                embedding = await self.embedding_service.generate_embedding(chunk["content"])
                
                # Store in vector store
                result = await self.vector_store_service.add_embedding(
                    content=chunk["content"],
                    embedding=embedding,
                    metadata={
                        **metadata,
                        **chunk.get("metadata", {}),
                        "chunk_index": chunk["index"],
                        "source_type": source_type
                    }
                )
                stored_chunks.append(result)
            
            # Track metrics
            await self.ai_intelligence.track_performance_metric(
                "document_ingestion",
                execution_time=0.0,
                success=True,
                metadata={
                    "chunk_count": len(chunks),
                    "source_type": source_type,
                    "enrichment": use_contextual_enrichment
                }
            )
            
            return {
                "success": True,
                "chunks_created": len(stored_chunks),
                "metadata": metadata,
                "source_type": source_type
            }
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": metadata
            }
    
    async def _create_chunks(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create chunks from content"""
        chunks = []
        
        if LLAMAINDEX_AVAILABLE and self.llamaindex_initialized:
            # Use LlamaIndex text splitter
            try:
                nodes = self.text_splitter.split_text(content)
                for i, node in enumerate(nodes):
                    chunks.append({
                        "content": node,
                        "index": i,
                        "metadata": {
                            "chunk_method": "llamaindex",
                            **metadata
                        }
                    })
                return chunks
            except Exception as e:
                logger.warning(f"LlamaIndex chunking failed: {e}")
        
        # Fallback to simple chunking
        chunk_size = 512
        chunk_overlap = 50
        
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk_text = content[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "index": len(chunks),
                    "metadata": {
                        "chunk_method": "simple",
                        "start_char": i,
                        "end_char": min(i + chunk_size, len(content)),
                        **metadata
                    }
                })
        
        return chunks
    
    async def _enrich_chunks(
        self,
        chunks: List[Dict[str, Any]],
        full_content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Add contextual enrichment to chunks"""
        enriched_chunks = []
        
        # Get document summary (first 500 chars)
        doc_summary = full_content[:500].replace('\n', ' ')
        
        for chunk in chunks:
            # Create context prefix
            context = f"Context: This is part of {metadata.get('title', 'a document')} "
            context += f"about {metadata.get('description', metadata.get('source_type', 'content'))}. "
            
            # Add enriched content
            enriched_chunk = chunk.copy()
            enriched_chunk["content"] = f"{context}\n\n{chunk['content']}"
            enriched_chunk["metadata"]["contextually_enriched"] = True
            enriched_chunk["metadata"]["enrichment_timestamp"] = datetime.utcnow().isoformat()
            
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    async def query(
        self,
        query_text: str,
        user_id: str,
        project_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_hybrid: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a RAG query using existing search infrastructure
        """
        try:
            # Check cache
            cache_key = f"rag_query:{query_text}:{user_id}:{project_id}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Build search filters
            search_filters = filters or {}
            if project_id:
                search_filters["project_id"] = project_id
            
            # Use existing hybrid search
            if use_hybrid:
                search_results = await self.search_service.hybrid_search(
                    query=query_text,
                    user_id=user_id,
                    top_k=top_k,
                    filters=search_filters
                )
            else:
                search_results = await self.search_service.vector_search(
                    query=query_text,
                    user_id=user_id,
                    top_k=top_k,
                    filters=search_filters
                )
            
            # Format as RAG response
            response = await self._format_rag_response(
                query_text,
                search_results,
                user_id,
                project_id
            )
            
            # Cache result
            await self.cache.set(cache_key, response, ttl=300)
            
            # Track metrics
            await self.ai_intelligence.track_performance_metric(
                "rag_query",
                execution_time=0.0,
                success=True,
                metadata={
                    "result_count": len(search_results.get("results", [])),
                    "has_project_context": bool(project_id),
                    "use_hybrid": use_hybrid
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "response": "I encountered an error while searching for information.",
                "source_nodes": [],
                "metadata": {
                    "query": query_text,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    async def _format_rag_response(
        self,
        query: str,
        search_results: Dict[str, Any],
        user_id: str,
        project_id: Optional[str]
    ) -> Dict[str, Any]:
        """Format search results as a RAG response"""
        if not search_results.get("results"):
            return {
                "response": "I couldn't find any relevant information for your query.",
                "source_nodes": [],
                "metadata": {
                    "query": query,
                    "user_id": user_id,
                    "project_id": project_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        # Build response from top results
        response_parts = ["Based on the available information:\n"]
        source_nodes = []
        
        for i, result in enumerate(search_results["results"][:3]):
            # Extract key information
            content = result.get("content", "")
            score = result.get("score", 0.0)
            metadata = result.get("metadata", {})
            
            # Add to response
            response_parts.append(f"\n{i+1}. {content[:200]}...")
            
            # Format as source node
            source_nodes.append({
                "text": content,
                "score": score,
                "metadata": metadata
            })
        
        return {
            "response": "\n".join(response_parts),
            "source_nodes": source_nodes,
            "metadata": {
                "query": query,
                "user_id": user_id,
                "project_id": project_id,
                "timestamp": datetime.utcnow().isoformat(),
                "total_results": len(search_results.get("results", [])),
                "implementation": "simple_rag"
            }
        }
    
    async def update_index_stats(self):
        """Get index statistics from existing services"""
        try:
            # Get stats from vector store
            vector_stats = await self.vector_store_service.get_stats()
            
            stats = {
                "total_documents": vector_stats.get("total_chunks", 0),
                "implementation": "simple_rag",
                "llamaindex_available": LLAMAINDEX_AVAILABLE,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            await self.cache.set("rag_index_stats", stats, ttl=3600)
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}


# Singleton instance
_rag_service = None


def get_rag_service() -> SimpleRAGService:
    """Get singleton RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = SimpleRAGService()
    return _rag_service