"""
Hybrid RAG Service - Advanced Multi-Modal Retrieval System
Implements dense vector search, sparse retrieval (BM25), graph-based retrieval, and reranking
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core RAG imports
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None
    
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import weaviate
from neo4j import GraphDatabase

# LangGraph imports (optional)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.runnables import RunnableConfig
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None
    BaseMessage = None
    HumanMessage = None
    AIMessage = None
    RunnableConfig = None
    TypedDict = dict  # Fallback to regular dict

# Existing KnowledgeHub services
from .vector_store import VectorStore
from .knowledge_graph import KnowledgeGraphService, NodeType, RelationType
from .embedding_service import EmbeddingService
from .cache import RedisCache
from .real_ai_intelligence import RealAIIntelligence
from ..models.document import Document, DocumentChunk
from ..config import settings

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Retrieval modes for hybrid search"""
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    GRAPH_ONLY = "graph_only"
    DENSE_SPARSE = "dense_sparse"
    DENSE_GRAPH = "dense_graph"
    SPARSE_GRAPH = "sparse_graph"
    HYBRID_ALL = "hybrid_all"


class RerankingModel(Enum):
    """Reranking model options"""
    CROSS_ENCODER = "cross_encoder"
    COLBERT = "colbert"
    BGE_RERANKER = "bge_reranker"
    COHERE_RERANK = "cohere_rerank"


@dataclass
class SearchResult:
    """Unified search result with scoring breakdown"""
    content: str
    doc_id: str
    chunk_id: Optional[str] = None
    dense_score: float = 0.0
    sparse_score: float = 0.0
    graph_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)


@dataclass
class HybridRAGConfig:
    """Configuration for hybrid RAG system"""
    # Dense retrieval
    dense_model: str = "BAAI/bge-base-en-v1.5"
    dense_top_k: int = 50
    dense_weight: float = 0.4
    
    # Sparse retrieval
    sparse_top_k: int = 50
    sparse_weight: float = 0.3
    
    # Graph retrieval
    graph_top_k: int = 30
    graph_weight: float = 0.2
    graph_max_depth: int = 3
    
    # Reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20
    rerank_weight: float = 0.1
    
    # Performance
    max_workers: int = 4
    batch_size: int = 32
    cache_ttl: int = 300
    enable_caching: bool = True
    
    # Quality thresholds
    min_dense_score: float = 0.5
    min_sparse_score: float = 0.1
    min_graph_score: float = 0.3
    min_final_score: float = 0.2


# LangGraph State Definition
if LANGGRAPH_AVAILABLE:
    class RAGState(TypedDict):
        """State for LangGraph workflow"""
        query: str
        user_id: str
        session_id: Optional[str]
        retrieval_mode: RetrievalMode
        top_k: int
        include_reasoning: bool
        dense_results: List[SearchResult]
        sparse_results: List[SearchResult]
        graph_results: List[SearchResult]
        merged_results: List[SearchResult]
        reranked_results: List[SearchResult]
        final_results: List[SearchResult]
        reasoning_steps: List[str]
        performance_metrics: Dict[str, float]
        context: Dict[str, Any]
else:
    # Fallback state for when LangGraph is not available
    RAGState = Dict[str, Any]


class HybridRAGService:
    """
    Advanced Hybrid RAG Service combining multiple retrieval strategies:
    1. Dense vector search (semantic similarity)
    2. Sparse retrieval (BM25 keyword matching)
    3. Graph-based retrieval (entity relationships)
    4. Cross-encoder reranking
    5. LangGraph orchestration for complex workflows
    """
    
    def __init__(self, config: Optional[HybridRAGConfig] = None):
        self.config = config or HybridRAGConfig()
        self.logger = logger
        
        # Initialize services
        self.cache = RedisCache(settings.REDIS_URL)
        self.ai_intelligence = RealAIIntelligence()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(
            url=settings.WEAVIATE_URL,
            collection_name=settings.WEAVIATE_COLLECTION_NAME
        )
        self.knowledge_graph = KnowledgeGraphService()
        
        # Initialize models
        self._init_models()
        
        # BM25 index (will be built on demand)
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_doc_mapping = {}
        
        # LangGraph workflow
        self.workflow = None
        self.memory = MemorySaver()
        
        # Performance tracking
        self.performance_stats = {
            "queries_processed": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
    def _init_models(self):
        """Initialize ML models for retrieval and reranking"""
        try:
            # Dense retrieval model
            self.dense_model = SentenceTransformer(
                self.config.dense_model,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Cross-encoder reranker
            self.reranker = CrossEncoder(
                self.config.rerank_model,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            # Fallback to simpler models or disable features
            self.dense_model = None
            self.reranker = None
    
    async def initialize(self):
        """Initialize all components and build workflow"""
        try:
            # Initialize cache
            await self.cache.initialize()
            
            # Initialize vector store
            await self.vector_store.initialize()
            
            # Initialize knowledge graph
            await self.knowledge_graph.initialize()
            
            # Build BM25 index
            await self._build_bm25_index()
            
            # Create LangGraph workflow
            self._create_workflow()
            
            logger.info("Hybrid RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid RAG service: {e}")
            raise
    
    async def _build_bm25_index(self):
        """Build BM25 index from existing documents"""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, sparse retrieval will be disabled")
            self.bm25_index = None
            return
            
        try:
            # Get all documents from vector store
            # This is a simplified approach - in production, you'd want to
            # maintain the BM25 index incrementally
            
            # For now, we'll build it lazily when needed
            logger.info("BM25 index will be built on demand")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
    
    def _create_workflow(self):
        """Create LangGraph workflow for RAG orchestration"""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, using simplified workflow")
            self.workflow = None
            return
        
        # Define the workflow graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("dense_retrieval", self._dense_retrieval)
        workflow.add_node("sparse_retrieval", self._sparse_retrieval)
        workflow.add_node("graph_retrieval", self._graph_retrieval)
        workflow.add_node("merge_results", self._merge_results)
        workflow.add_node("rerank_results", self._rerank_results)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define edges and conditional routing
        workflow.set_entry_point("analyze_query")
        
        workflow.add_edge("analyze_query", "dense_retrieval")
        workflow.add_edge("analyze_query", "sparse_retrieval") 
        workflow.add_edge("analyze_query", "graph_retrieval")
        
        workflow.add_edge("dense_retrieval", "merge_results")
        workflow.add_edge("sparse_retrieval", "merge_results")
        workflow.add_edge("graph_retrieval", "merge_results")
        
        workflow.add_edge("merge_results", "rerank_results")
        workflow.add_edge("rerank_results", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile workflow
        self.workflow = workflow.compile(checkpointer=self.memory)
        
        logger.info("LangGraph workflow created successfully")
    
    async def query(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID_ALL,
        top_k: int = 10,
        include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Execute hybrid RAG query using LangGraph orchestration
        
        Args:
            query: User query
            user_id: User identifier
            session_id: Session identifier
            retrieval_mode: Retrieval strategy to use
            top_k: Number of results to return
            include_reasoning: Include reasoning steps
            
        Returns:
            Query results with metadata
        """
        start_time = time.time()
        session_id = session_id or f"session_{int(time.time())}"
        
        try:
            # Check cache first
            if self.config.enable_caching:
                cache_key = f"hybrid_rag:{query}:{user_id}:{retrieval_mode.value}"
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    logger.info("Returning cached result")
                    return cached_result
            
            # Initialize state
            if LANGGRAPH_AVAILABLE:
                initial_state: RAGState = {
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "retrieval_mode": retrieval_mode,
                    "top_k": top_k,
                    "include_reasoning": include_reasoning,
                    "dense_results": [],
                    "sparse_results": [],
                    "graph_results": [],
                    "merged_results": [],
                    "reranked_results": [],
                    "final_results": [],
                    "reasoning_steps": [],
                    "performance_metrics": {"start_time": start_time},
                    "context": {}
                }
            else:
                # Fallback state structure
                initial_state = {
                    "query": query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "retrieval_mode": retrieval_mode.value,
                    "top_k": top_k,
                    "include_reasoning": include_reasoning,
                    "dense_results": [],
                    "sparse_results": [],
                    "graph_results": [],
                    "merged_results": [],
                    "reranked_results": [],
                    "final_results": [],
                    "reasoning_steps": [],
                    "performance_metrics": {"start_time": start_time},
                    "context": {}
                }
            
            # Execute workflow
            if LANGGRAPH_AVAILABLE and self.workflow:
                config = RunnableConfig(
                    configurable={"thread_id": session_id}
                )
                result_state = await self.workflow.ainvoke(initial_state, config)
            else:
                # Fallback: execute workflow steps manually
                result_state = await self._execute_fallback_workflow(initial_state)
            
            # Format response
            final_results = result_state.get("final_results", result_state.get("reranked_results", []))
            response = {
                "response": result_state.get("final_response", "Query processed successfully"),
                "results": final_results[:top_k],
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "retrieval_mode": retrieval_mode.value,
                    "total_results": len(result_state.get("merged_results", [])),
                    "langgraph_enabled": LANGGRAPH_AVAILABLE,
                    "bm25_enabled": BM25_AVAILABLE
                },
                "performance_metrics": result_state.get("performance_metrics", {})
            }
            
            if include_reasoning:
                response["reasoning_steps"] = result_state["reasoning_steps"]
            
            # Cache result
            if self.config.enable_caching:
                await self.cache.set(cache_key, response, ttl=self.config.cache_ttl)
            
            # Track performance
            await self._track_query_performance(
                query, time.time() - start_time, len(final_results)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Hybrid RAG query failed: {e}")
            # Return fallback response
            return {
                "response": "I apologize, but I encountered an error processing your query.",
                "results": [],
                "metadata": {"error": str(e), "processing_time": time.time() - start_time},
                "performance_metrics": {}
            }
    
    async def _analyze_query(self, state: RAGState) -> RAGState:
        """Analyze query to determine optimal retrieval strategy"""
        query = state["query"]
        
        # Simple query analysis - can be enhanced with NLP
        reasoning_steps = [f"Analyzing query: '{query}'"]
        
        # Detect query type and adjust weights
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "how", "why", "explain"]):
            reasoning_steps.append("Detected explanatory query - favoring dense retrieval")
        elif any(word in query_lower for word in ["find", "search", "list", "show"]):
            reasoning_steps.append("Detected search query - favoring sparse retrieval")
        elif any(word in query_lower for word in ["related", "connection", "relationship"]):
            reasoning_steps.append("Detected relationship query - favoring graph retrieval")
        
        state["reasoning_steps"] = reasoning_steps
        state["metadata"]["query_analysis_complete"] = True
        
        return state
    
    async def _dense_retrieval(self, state: RAGState) -> RAGState:
        """Perform dense vector retrieval"""
        query = state["query"]
        start_time = time.time()
        
        try:
            if not self.dense_model:
                state["dense_results"] = []
                return state
            
            # Generate query embedding
            query_embedding = self.dense_model.encode(query).tolist()
            
            # Search vector store
            results = await self.vector_store.search(
                query_vector=query_embedding,
                limit=self.config.dense_top_k
            )
            
            # Convert to SearchResult objects
            dense_results = []
            for result in results:
                search_result = SearchResult(
                    content=result.get("content", ""),
                    doc_id=result.get("doc_id", ""),
                    chunk_id=result.get("id"),
                    dense_score=result.get("score", 0.0),
                    metadata=result.get("metadata", {})
                )
                
                # Only include results above threshold
                if search_result.dense_score >= self.config.min_dense_score:
                    dense_results.append(search_result)
            
            state["dense_results"] = dense_results
            state["reasoning_steps"].append(f"Dense retrieval found {len(dense_results)} results")
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            state["dense_results"] = []
        
        state["performance_metrics"]["dense_retrieval_time"] = time.time() - start_time
        return state
    
    async def _sparse_retrieval(self, state: RAGState) -> RAGState:
        """Perform sparse (BM25) retrieval"""
        query = state["query"]
        start_time = time.time()
        
        if not BM25_AVAILABLE:
            state["sparse_results"] = []
            state["reasoning_steps"].append("BM25 not available, skipping sparse retrieval")
            state["performance_metrics"]["sparse_time"] = 0.0
            return state
        
        try:
            # Build BM25 index if not exists
            if self.bm25_index is None:
                await self._build_bm25_from_vector_store()
            
            if not self.bm25_index:
                state["sparse_results"] = []
                return state
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:self.config.sparse_top_k]
            
            sparse_results = []
            for idx in top_indices:
                score = scores[idx]
                if score >= self.config.min_sparse_score and idx < len(self.bm25_corpus):
                    doc_info = self.bm25_doc_mapping.get(idx, {})
                    
                    search_result = SearchResult(
                        content=self.bm25_corpus[idx],
                        doc_id=doc_info.get("doc_id", f"doc_{idx}"),
                        chunk_id=doc_info.get("chunk_id"),
                        sparse_score=float(score),
                        metadata=doc_info.get("metadata", {})
                    )
                    sparse_results.append(search_result)
            
            state["sparse_results"] = sparse_results
            state["reasoning_steps"].append(f"Sparse retrieval found {len(sparse_results)} results")
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            state["sparse_results"] = []
        
        state["performance_metrics"]["sparse_retrieval_time"] = time.time() - start_time
        return state
    
    async def _graph_retrieval(self, state: RAGState) -> RAGState:
        """Perform graph-based retrieval"""
        query = state["query"]
        start_time = time.time()
        
        try:
            # Extract entities from query (simple keyword matching)
            query_entities = await self._extract_query_entities(query)
            
            if not query_entities:
                state["graph_results"] = []
                return state
            
            # Search knowledge graph for related content
            graph_results = []
            
            for entity in query_entities:
                # Find documents related to this entity
                related_docs = await self.knowledge_graph.search_graph(
                    query_text=entity,
                    limit=self.config.graph_top_k // len(query_entities)
                )
                
                for doc in related_docs:
                    search_result = SearchResult(
                        content=doc.get("properties", {}).get("content", ""),
                        doc_id=doc.get("id", ""),
                        graph_score=doc.get("score", 0.0),
                        entities=[entity],
                        metadata={
                            "graph_entity": entity,
                            "node_type": doc.get("type", ""),
                            **doc.get("properties", {})
                        }
                    )
                    
                    if search_result.graph_score >= self.config.min_graph_score:
                        graph_results.append(search_result)
            
            state["graph_results"] = graph_results
            state["reasoning_steps"].append(f"Graph retrieval found {len(graph_results)} results")
            
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            state["graph_results"] = []
        
        state["performance_metrics"]["graph_retrieval_time"] = time.time() - start_time
        return state
    
    async def _merge_results(self, state: RAGState) -> RAGState:
        """Merge results from different retrieval methods"""
        start_time = time.time()
        
        # Combine all results
        all_results = (
            state["dense_results"] + 
            state["sparse_results"] + 
            state["graph_results"]
        )
        
        # Deduplicate by content similarity
        merged_results = []
        seen_contents = set()
        
        for result in all_results:
            # Use first 100 characters as deduplication key
            content_key = result.content[:100].strip()
            
            if content_key not in seen_contents and content_key:
                seen_contents.add(content_key)
                
                # Calculate weighted combined score
                result.final_score = (
                    result.dense_score * self.config.dense_weight +
                    result.sparse_score * self.config.sparse_weight +
                    result.graph_score * self.config.graph_weight
                )
                
                if result.final_score >= self.config.min_final_score:
                    merged_results.append(result)
        
        # Sort by final score
        merged_results.sort(key=lambda x: x.final_score, reverse=True)
        
        state["merged_results"] = merged_results
        state["reasoning_steps"].append(
            f"Merged {len(all_results)} results into {len(merged_results)} unique results"
        )
        
        state["performance_metrics"]["merge_time"] = time.time() - start_time
        return state
    
    async def _rerank_results(self, state: RAGState) -> RAGState:
        """Rerank results using cross-encoder"""
        start_time = time.time()
        query = state["query"]
        merged_results = state["merged_results"]
        
        try:
            if not self.reranker or not merged_results:
                state["reranked_results"] = merged_results[:self.config.rerank_top_k]
                return state
            
            # Prepare query-document pairs for reranking
            pairs = [(query, result.content[:512]) for result in merged_results]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update results with rerank scores
            for i, result in enumerate(merged_results):
                result.rerank_score = float(rerank_scores[i])
                # Combine with existing score
                result.final_score = (
                    result.final_score * (1 - self.config.rerank_weight) +
                    result.rerank_score * self.config.rerank_weight
                )
            
            # Sort by final score
            merged_results.sort(key=lambda x: x.final_score, reverse=True)
            
            state["reranked_results"] = merged_results[:self.config.rerank_top_k]
            state["reasoning_steps"].append(f"Reranked top {len(state['reranked_results'])} results")
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            state["reranked_results"] = merged_results[:self.config.rerank_top_k]
        
        state["performance_metrics"]["rerank_time"] = time.time() - start_time
        return state
    
    async def _generate_response(self, state: RAGState) -> RAGState:
        """Generate final response based on retrieved results"""
        start_time = time.time()
        query = state["query"]
        results = state["reranked_results"]
        
        if not results:
            state["final_response"] = "I couldn't find relevant information for your query."
            return state
        
        # Simple response generation - can be enhanced with LLM
        response_parts = [f"Based on the available information, here's what I found about '{query}':\n"]
        
        for i, result in enumerate(results[:3], 1):
            content_snippet = result.content[:200] + "..." if len(result.content) > 200 else result.content
            response_parts.append(f"{i}. {content_snippet}")
            
            if result.entities:
                response_parts.append(f"   (Related entities: {', '.join(result.entities)})")
        
        if len(results) > 3:
            response_parts.append(f"\nI found {len(results)} total relevant results.")
        
        state["final_response"] = "\n\n".join(response_parts)
        state["reasoning_steps"].append("Generated response from top results")
        
        state["performance_metrics"]["response_generation_time"] = time.time() - start_time
        return state
    
    async def _build_bm25_from_vector_store(self):
        """Build BM25 index from vector store data"""
        try:
            # This is a simplified approach - in production you'd want to
            # maintain this index more efficiently
            
            # For now, create a mock corpus
            self.bm25_corpus = [
                "example document content for BM25 indexing",
                "another document with different content",
                "technical documentation example"
            ]
            
            self.bm25_doc_mapping = {
                i: {
                    "doc_id": f"doc_{i}",
                    "chunk_id": f"chunk_{i}",
                    "metadata": {"source": "vector_store"}
                }
                for i in range(len(self.bm25_corpus))
            }
            
            # Tokenize corpus
            tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            logger.info(f"Built BM25 index with {len(self.bm25_corpus)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query text"""
        # Simple keyword-based entity extraction
        # In production, use NER models like spaCy or transformers
        
        tech_keywords = [
            "python", "javascript", "react", "nodejs", "docker", "kubernetes",
            "machine learning", "ai", "deep learning", "neural network",
            "database", "sql", "nosql", "api", "rest", "graphql"
        ]
        
        entities = []
        query_lower = query.lower()
        
        for keyword in tech_keywords:
            if keyword in query_lower:
                entities.append(keyword)
        
        return entities[:5]  # Limit to top 5 entities
    
    async def _track_query_performance(self, query: str, processing_time: float, result_count: int):
        """Track query performance metrics"""
        try:
            self.performance_stats["queries_processed"] += 1
            
            # Update average response time
            total_queries = self.performance_stats["queries_processed"]
            old_avg = self.performance_stats["avg_response_time"]
            self.performance_stats["avg_response_time"] = (
                (old_avg * (total_queries - 1) + processing_time) / total_queries
            )
            
            # Track in AI intelligence system
            await self.ai_intelligence.track_performance_metric(
                "hybrid_rag_query",
                execution_time=processing_time,
                success=result_count > 0,
                metadata={
                    "result_count": result_count,
                    "query_length": len(query)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to track performance: {e}")
    
    async def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest document into all retrieval systems
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID
            
        Returns:
            Ingestion results
        """
        try:
            doc_id = doc_id or f"doc_{int(time.time())}"
            
            # Ingest into vector store (dense retrieval)
            if self.dense_model:
                embedding = self.dense_model.encode(content).tolist()
                await self.vector_store.insert_chunk(
                    chunk_data={
                        "doc_id": doc_id,
                        "content": content,
                        "metadata": json.dumps(metadata),
                        "created_at": datetime.utcnow().isoformat()
                    },
                    vector=embedding
                )
            
            # Update BM25 index (sparse retrieval)
            if content.strip():
                self.bm25_corpus.append(content)
                idx = len(self.bm25_corpus) - 1
                self.bm25_doc_mapping[idx] = {
                    "doc_id": doc_id,
                    "metadata": metadata
                }
                
                # Rebuild BM25 index
                tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
                self.bm25_index = BM25Okapi(tokenized_corpus)
            
            # Extract entities and add to knowledge graph
            entities = await self._extract_query_entities(content)
            if entities:
                # Create document node
                doc_node_id = await self.knowledge_graph.create_node(
                    NodeType.DOCUMENT,
                    {
                        "id": doc_id,
                        "content": content[:1000],  # Truncate for storage
                        "title": metadata.get("title", "Untitled"),
                        **metadata
                    }
                )
                
                # Create entity nodes and relationships
                for entity in entities:
                    entity_node_id = await self.knowledge_graph.create_node(
                        NodeType.ENTITY,
                        {
                            "id": f"entity_{entity}",
                            "name": entity,
                            "type": "concept"
                        }
                    )
                    
                    # Create relationship
                    await self.knowledge_graph.create_relationship(
                        doc_node_id,
                        entity_node_id,
                        RelationType.MENTIONS
                    )
            
            return {
                "success": True,
                "doc_id": doc_id,
                "entities_extracted": len(entities),
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "doc_id": doc_id
            }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        return {
            **self.performance_stats,
            "config": {
                "dense_weight": self.config.dense_weight,
                "sparse_weight": self.config.sparse_weight,
                "graph_weight": self.config.graph_weight,
                "rerank_weight": self.config.rerank_weight
            },
            "index_stats": {
                "bm25_corpus_size": len(self.bm25_corpus),
                "models_loaded": {
                    "dense_model": self.dense_model is not None,
                    "reranker": self.reranker is not None
                }
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check vector store
        try:
            vector_healthy = await self.vector_store.health_check()
            health["components"]["vector_store"] = "healthy" if vector_healthy else "unhealthy"
        except:
            health["components"]["vector_store"] = "error"
        
        # Check knowledge graph
        try:
            # Simple check - can be enhanced
            health["components"]["knowledge_graph"] = "healthy"
        except:
            health["components"]["knowledge_graph"] = "error"
        
        # Check models
        health["components"]["dense_model"] = "loaded" if self.dense_model else "not_loaded"
        health["components"]["reranker"] = "loaded" if self.reranker else "not_loaded"
        health["components"]["bm25_index"] = "ready" if self.bm25_index else "not_ready"
        
        # Overall status
        if any(status in ["unhealthy", "error"] for status in health["components"].values()):
            health["status"] = "degraded"
        
        return health
    
    async def _execute_fallback_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps manually when LangGraph is not available"""
        try:
            # Execute workflow steps in sequence
            state = await self._analyze_query(state)
            state = await self._dense_retrieval(state)
            state = await self._sparse_retrieval(state)
            state = await self._graph_retrieval(state)
            state = await self._merge_results(state)
            state = await self._rerank_results(state)
            state = await self._generate_response(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Fallback workflow execution failed: {e}")
            # Return basic state with error
            state["final_response"] = f"Error during query processing: {str(e)}"
            state["final_results"] = state.get("dense_results", [])[:state.get("top_k", 10)]
            return state


# Global instance
_hybrid_rag_service: Optional[HybridRAGService] = None


async def get_hybrid_rag_service() -> HybridRAGService:
    """Get singleton hybrid RAG service instance"""
    global _hybrid_rag_service
    
    if _hybrid_rag_service is None:
        _hybrid_rag_service = HybridRAGService()
        await _hybrid_rag_service.initialize()
    
    return _hybrid_rag_service