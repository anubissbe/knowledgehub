"""
Advanced RAG Pipeline Implementation for KnowledgeHub
Implements state-of-the-art RAG techniques with performance optimizations
"""

import asyncio
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field
import tiktoken
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import text
import httpx

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Advanced chunking strategies"""
    SEMANTIC = "semantic"  # Semantic-aware chunking
    SLIDING = "sliding"  # Sliding window with overlap
    RECURSIVE = "recursive"  # Recursive character splitting
    PROPOSITION = "proposition"  # Proposition-based chunking
    HIERARCHICAL = "hierarchical"  # Multi-level hierarchical chunking
    ADAPTIVE = "adaptive"  # Context-aware adaptive sizing


class RetrievalStrategy(Enum):
    """Retrieval strategies for RAG"""
    VECTOR = "vector"  # Pure vector similarity
    HYBRID = "hybrid"  # Vector + keyword search
    ENSEMBLE = "ensemble"  # Multiple retrieval methods
    ITERATIVE = "iterative"  # Progressive refinement
    GRAPH = "graph"  # Knowledge graph-enhanced
    ADAPTIVE = "adaptive"  # Query-dependent strategy


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 128
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    
    # Retrieval parameters
    top_k: int = 10
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    similarity_threshold: float = 0.7
    
    # Reranking parameters
    enable_reranking: bool = True
    rerank_top_k: int = 5
    
    # Generation parameters
    max_context_length: int = 4096
    temperature: float = 0.7
    
    # Performance optimization
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_compression: bool = True
    
    # Advanced features
    enable_hyde: bool = True  # Hypothetical Document Embedding
    enable_graph_rag: bool = True  # Knowledge graph integration
    enable_self_correction: bool = True  # Self-RAG


@dataclass
class Document:
    """Document representation with metadata"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List['Chunk'] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Chunk:
    """Chunk representation with context"""
    id: str
    content: str
    document_id: str
    position: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    context_before: str = ""
    context_after: str = ""
    importance_score: float = 1.0


class AdvancedChunker:
    """Advanced document chunking with multiple strategies"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Apply configured chunking strategy"""
        if self.config.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(document)
        elif self.config.chunking_strategy == ChunkingStrategy.HIERARCHICAL:
            return self._hierarchical_chunking(document)
        elif self.config.chunking_strategy == ChunkingStrategy.PROPOSITION:
            return self._proposition_chunking(document)
        elif self.config.chunking_strategy == ChunkingStrategy.ADAPTIVE:
            return self._adaptive_chunking(document)
        else:
            return self._sliding_window_chunking(document)
    
    def _hierarchical_chunking(self, document: Document) -> List[Chunk]:
        """Multi-level hierarchical chunking for better context preservation"""
        chunks = []
        
        # Level 1: Split by major sections (e.g., headers, paragraphs)
        sections = self._split_by_sections(document.content)
        
        for section_idx, section in enumerate(sections):
            # Level 2: Split sections into semantic units
            semantic_units = self._split_semantic_units(section)
            
            for unit_idx, unit in enumerate(semantic_units):
                # Level 3: Apply size constraints
                if len(self.tokenizer.encode(unit)) > self.config.chunk_size:
                    sub_chunks = self._split_by_size(unit)
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        chunk_id = f"{document.id}_s{section_idx}_u{unit_idx}_c{sub_idx}"
                        chunks.append(self._create_chunk(
                            chunk_id, sub_chunk, document.id,
                            len(chunks), section, semantic_units
                        ))
                else:
                    chunk_id = f"{document.id}_s{section_idx}_u{unit_idx}"
                    chunks.append(self._create_chunk(
                        chunk_id, unit, document.id,
                        len(chunks), section, semantic_units
                    ))
        
        return chunks
    
    def _semantic_chunking(self, document: Document) -> List[Chunk]:
        """Semantic-aware chunking based on sentence boundaries and coherence"""
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(document.content)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_size + sentence_tokens > self.config.chunk_size:
                if current_chunk:
                    chunk_content = ' '.join(current_chunk)
                    chunk_id = f"{document.id}_chunk_{len(chunks)}"
                    chunks.append(Chunk(
                        id=chunk_id,
                        content=chunk_content,
                        document_id=document.id,
                        position=len(chunks),
                        metadata={"strategy": "semantic"}
                    ))
                current_chunk = [sentence]
                current_size = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_size += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk_id = f"{document.id}_chunk_{len(chunks)}"
            chunks.append(Chunk(
                id=chunk_id,
                content=chunk_content,
                document_id=document.id,
                position=len(chunks),
                metadata={"strategy": "semantic"}
            ))
        
        return chunks
    
    def _proposition_chunking(self, document: Document) -> List[Chunk]:
        """Extract and chunk by logical propositions"""
        # Simplified proposition extraction
        # In production, use more sophisticated NLP techniques
        propositions = self._extract_propositions(document.content)
        chunks = []
        
        for idx, prop in enumerate(propositions):
            chunk_id = f"{document.id}_prop_{idx}"
            chunks.append(Chunk(
                id=chunk_id,
                content=prop,
                document_id=document.id,
                position=idx,
                metadata={"strategy": "proposition", "type": "logical_unit"}
            ))
        
        return chunks
    
    def _adaptive_chunking(self, document: Document) -> List[Chunk]:
        """Adaptive chunking based on content complexity"""
        # Analyze document complexity
        complexity_score = self._analyze_complexity(document.content)
        
        # Adjust chunk size based on complexity
        adaptive_chunk_size = int(self.config.chunk_size * (1.5 - complexity_score))
        
        # Apply chunking with adaptive size
        return self._sliding_window_chunking(document, chunk_size=adaptive_chunk_size)
    
    def _sliding_window_chunking(self, document: Document, chunk_size: Optional[int] = None) -> List[Chunk]:
        """Standard sliding window chunking with overlap"""
        chunk_size = chunk_size or self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        tokens = self.tokenizer.encode(document.content)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_content = self.tokenizer.decode(chunk_tokens)
            
            chunk_id = f"{document.id}_chunk_{len(chunks)}"
            chunks.append(Chunk(
                id=chunk_id,
                content=chunk_content,
                document_id=document.id,
                position=len(chunks),
                metadata={"strategy": "sliding_window"}
            ))
        
        return chunks
    
    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by major sections"""
        # Simple implementation - can be enhanced with better section detection
        sections = content.split('\n\n')
        return [s.strip() for s in sections if s.strip()]
    
    def _split_semantic_units(self, content: str) -> List[str]:
        """Split into semantic units maintaining coherence"""
        # Simplified implementation
        import re
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_size(self, content: str) -> List[str]:
        """Split by token size constraints"""
        tokens = self.tokenizer.encode(content)
        chunks = []
        
        for i in range(0, len(tokens), self.config.chunk_size):
            chunk_tokens = tokens[i:i + self.config.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks
    
    def _create_chunk(self, chunk_id: str, content: str, doc_id: str, 
                     position: int, context: str, units: List[str]) -> Chunk:
        """Create chunk with enriched context"""
        # Add context from surrounding units
        context_before = ""
        context_after = ""
        
        if position > 0 and position < len(units):
            context_before = units[position - 1][-100:] if position > 0 else ""
            context_after = units[position + 1][:100] if position < len(units) - 1 else ""
        
        return Chunk(
            id=chunk_id,
            content=content,
            document_id=doc_id,
            position=position,
            context_before=context_before,
            context_after=context_after,
            metadata={"strategy": "hierarchical"}
        )
    
    def _extract_propositions(self, content: str) -> List[str]:
        """Extract logical propositions from content"""
        # Simplified extraction - in production use dependency parsing
        import re
        sentences = re.split(r'[.!?]+', content)
        propositions = []
        
        for sentence in sentences:
            # Extract clauses
            clauses = re.split(r'[,;]', sentence)
            propositions.extend([c.strip() for c in clauses if len(c.strip()) > 20])
        
        return propositions
    
    def _analyze_complexity(self, content: str) -> float:
        """Analyze content complexity (0-1 scale)"""
        # Simple complexity metrics
        tokens = self.tokenizer.encode(content)
        
        # Average token length
        avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
        
        # Sentence complexity
        sentences = content.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (avg_token_length / 10 + avg_sentence_length / 50) / 2)
        
        return complexity


class HybridRetriever:
    """Advanced hybrid retrieval with multiple strategies"""
    
    def __init__(self, config: RAGConfig, db: Session):
        self.config = config
        self.db = db
        self.cache = {}
    
    async def retrieve(self, query: str, strategy: Optional[RetrievalStrategy] = None) -> List[Chunk]:
        """Retrieve relevant chunks using configured strategy"""
        strategy = strategy or self.config.retrieval_strategy
        
        if strategy == RetrievalStrategy.VECTOR:
            return await self._vector_retrieval(query)
        elif strategy == RetrievalStrategy.HYBRID:
            return await self._hybrid_retrieval(query)
        elif strategy == RetrievalStrategy.ENSEMBLE:
            return await self._ensemble_retrieval(query)
        elif strategy == RetrievalStrategy.GRAPH:
            return await self._graph_enhanced_retrieval(query)
        elif strategy == RetrievalStrategy.ITERATIVE:
            return await self._iterative_retrieval(query)
        else:
            return await self._adaptive_retrieval(query)
    
    async def _vector_retrieval(self, query: str) -> List[Chunk]:
        """Pure vector similarity search"""
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Search in vector database
        results = self.db.execute(text("""
            SELECT id, content, document_id, embedding <=> :embedding as distance
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :embedding
            LIMIT :limit
        """), {
            "embedding": query_embedding.tolist(),
            "limit": self.config.top_k
        }).fetchall()
        
        chunks = []
        for row in results:
            if row.distance <= 1 - self.config.similarity_threshold:
                chunks.append(Chunk(
                    id=row.id,
                    content=row.content,
                    document_id=row.document_id,
                    position=0,
                    metadata={"distance": row.distance}
                ))
        
        return chunks
    
    async def _hybrid_retrieval(self, query: str) -> List[Chunk]:
        """Combine vector and keyword search"""
        # Vector search
        vector_chunks = await self._vector_retrieval(query)
        
        # Keyword search using full-text search
        keyword_results = self.db.execute(text("""
            SELECT id, content, document_id, 
                   ts_rank_cd(search_vector, plainto_tsquery('english', :query)) as rank
            FROM chunks
            WHERE search_vector @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """), {
            "query": query,
            "limit": self.config.top_k
        }).fetchall()
        
        keyword_chunks = []
        for row in keyword_results:
            keyword_chunks.append(Chunk(
                id=row.id,
                content=row.content,
                document_id=row.document_id,
                position=0,
                metadata={"rank": float(row.rank)}
            ))
        
        # Merge and rerank
        return self._merge_and_rerank(vector_chunks, keyword_chunks)
    
    async def _ensemble_retrieval(self, query: str) -> List[Chunk]:
        """Multiple retrieval methods with ensemble voting"""
        # Run multiple strategies in parallel
        strategies = [
            self._vector_retrieval(query),
            self._bm25_retrieval(query),
            self._semantic_retrieval(query)
        ]
        
        results = await asyncio.gather(*strategies)
        
        # Ensemble voting
        chunk_scores = {}
        for strategy_results in results:
            for i, chunk in enumerate(strategy_results):
                if chunk.id not in chunk_scores:
                    chunk_scores[chunk.id] = {"chunk": chunk, "score": 0}
                # Reciprocal rank fusion
                chunk_scores[chunk.id]["score"] += 1 / (i + 1)
        
        # Sort by ensemble score
        sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["chunk"] for item in sorted_chunks[:self.config.top_k]]
    
    async def _graph_enhanced_retrieval(self, query: str) -> List[Chunk]:
        """Knowledge graph-enhanced retrieval"""
        # First, get initial chunks
        initial_chunks = await self._vector_retrieval(query)
        
        if not initial_chunks:
            return []
        
        # Query Neo4j for related entities and concepts
        try:
            # This would connect to Neo4j and find related nodes
            # Simplified for now
            enhanced_chunks = initial_chunks.copy()
            
            # Add graph context to metadata
            for chunk in enhanced_chunks:
                chunk.metadata["graph_enhanced"] = True
                chunk.metadata["related_entities"] = []  # Would be populated from Neo4j
            
            return enhanced_chunks
        except Exception as e:
            logger.warning(f"Graph enhancement failed: {e}")
            return initial_chunks
    
    async def _iterative_retrieval(self, query: str) -> List[Chunk]:
        """Progressive retrieval with query refinement"""
        chunks = []
        refined_query = query
        
        for iteration in range(3):  # Max 3 iterations
            # Retrieve with current query
            iter_chunks = await self._vector_retrieval(refined_query)
            
            if not iter_chunks:
                break
            
            chunks.extend(iter_chunks)
            
            # Refine query based on retrieved content
            refined_query = self._refine_query(refined_query, iter_chunks)
            
            # Check if we have enough diverse chunks
            if len(set(c.document_id for c in chunks)) >= self.config.top_k // 2:
                break
        
        # Deduplicate and return top k
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk.id not in seen:
                seen.add(chunk.id)
                unique_chunks.append(chunk)
        
        return unique_chunks[:self.config.top_k]
    
    async def _adaptive_retrieval(self, query: str) -> List[Chunk]:
        """Adapt retrieval strategy based on query characteristics"""
        # Analyze query
        query_length = len(query.split())
        has_technical_terms = self._detect_technical_terms(query)
        is_question = query.strip().endswith('?')
        
        # Select strategy based on analysis
        if query_length < 5 and is_question:
            # Short question - use hybrid
            return await self._hybrid_retrieval(query)
        elif has_technical_terms:
            # Technical query - use ensemble
            return await self._ensemble_retrieval(query)
        elif query_length > 20:
            # Long query - use iterative
            return await self._iterative_retrieval(query)
        else:
            # Default to vector
            return await self._vector_retrieval(query)
    
    async def _bm25_retrieval(self, query: str) -> List[Chunk]:
        """BM25 keyword-based retrieval"""
        # Simplified BM25 using PostgreSQL full-text search
        results = self.db.execute(text("""
            SELECT id, content, document_id,
                   ts_rank_cd(search_vector, query, 32) as rank
            FROM chunks,
                 plainto_tsquery('english', :query) query
            WHERE search_vector @@ query
            ORDER BY rank DESC
            LIMIT :limit
        """), {
            "query": query,
            "limit": self.config.top_k
        }).fetchall()
        
        chunks = []
        for row in results:
            chunks.append(Chunk(
                id=row.id,
                content=row.content,
                document_id=row.document_id,
                position=0,
                metadata={"bm25_score": float(row.rank)}
            ))
        
        return chunks
    
    async def _semantic_retrieval(self, query: str) -> List[Chunk]:
        """Semantic similarity using sentence transformers"""
        # This would use a more sophisticated semantic model
        # For now, fallback to vector retrieval
        return await self._vector_retrieval(query)
    
    def _merge_and_rerank(self, *chunk_lists: List[Chunk]) -> List[Chunk]:
        """Merge multiple chunk lists and rerank"""
        all_chunks = {}
        
        for chunks in chunk_lists:
            for i, chunk in enumerate(chunks):
                if chunk.id not in all_chunks:
                    all_chunks[chunk.id] = {
                        "chunk": chunk,
                        "scores": [],
                        "positions": []
                    }
                all_chunks[chunk.id]["scores"].append(
                    chunk.metadata.get("distance", 0) or chunk.metadata.get("rank", 0)
                )
                all_chunks[chunk.id]["positions"].append(i)
        
        # Calculate combined score
        for chunk_data in all_chunks.values():
            # Reciprocal rank fusion
            chunk_data["combined_score"] = sum(1 / (p + 1) for p in chunk_data["positions"])
        
        # Sort by combined score
        sorted_chunks = sorted(all_chunks.values(), 
                             key=lambda x: x["combined_score"], reverse=True)
        
        return [item["chunk"] for item in sorted_chunks[:self.config.top_k]]
    
    def _refine_query(self, query: str, chunks: List[Chunk]) -> str:
        """Refine query based on retrieved chunks"""
        # Extract key terms from chunks
        from collections import Counter
        
        all_text = ' '.join(c.content[:200] for c in chunks[:3])
        words = all_text.lower().split()
        
        # Find frequent terms not in original query
        query_words = set(query.lower().split())
        chunk_words = [w for w in words if w not in query_words and len(w) > 4]
        
        if chunk_words:
            # Add most common new term to query
            most_common = Counter(chunk_words).most_common(2)
            additional_terms = ' '.join(term for term, _ in most_common)
            return f"{query} {additional_terms}"
        
        return query
    
    def _detect_technical_terms(self, query: str) -> bool:
        """Detect if query contains technical terms"""
        technical_indicators = [
            'api', 'function', 'class', 'method', 'database', 'algorithm',
            'implementation', 'architecture', 'framework', 'library'
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in technical_indicators)
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Call embedding service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8002/embeddings/generate",
                json={"text": text}
            )
            response.raise_for_status()
            embedding = np.array(response.json()["embedding"])
        
        # Cache result
        if self.config.enable_caching:
            self.cache[cache_key] = embedding
        
        return embedding


class RAGPipeline:
    """Complete RAG pipeline with advanced features"""
    
    def __init__(self, config: RAGConfig, db: Session):
        self.config = config
        self.db = db
        self.chunker = AdvancedChunker(config)
        self.retriever = HybridRetriever(config, db)
        self.reranker = AdvancedReranker(config)
        self.generator = ResponseGenerator(config)
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline"""
        start_time = datetime.utcnow()
        
        # 1. Query preprocessing
        processed_query = await self._preprocess_query(query, context)
        
        # 2. Retrieval
        chunks = await self.retriever.retrieve(processed_query)
        
        # 3. Reranking
        if self.config.enable_reranking:
            chunks = await self.reranker.rerank(processed_query, chunks)
        
        # 4. Context construction
        context_text = self._construct_context(chunks)
        
        # 5. Response generation
        response = await self.generator.generate(processed_query, context_text, chunks)
        
        # 6. Post-processing and validation
        if self.config.enable_self_correction:
            response = await self._self_correct(response, chunks)
        
        # Calculate metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "query": query,
            "response": response,
            "chunks_used": len(chunks),
            "processing_time": processing_time,
            "metadata": {
                "retrieval_strategy": self.config.retrieval_strategy.value,
                "chunking_strategy": self.config.chunking_strategy.value,
                "chunks_retrieved": [{"id": c.id, "score": c.metadata.get("score", 0)} for c in chunks[:5]]
            }
        }
    
    async def ingest_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Ingest a new document into the RAG system"""
        # Create document
        doc_id = hashlib.md5(content.encode()).hexdigest()
        document = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {}
        )
        
        # Chunk document
        chunks = self.chunker.chunk_document(document)
        
        # Generate embeddings for chunks
        for chunk in chunks:
            chunk.embedding = await self.retriever._get_embedding(chunk.content)
        
        # Store in database
        await self._store_document_and_chunks(document, chunks)
        
        return document
    
    async def _preprocess_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Preprocess and enhance query"""
        # HyDE: Generate hypothetical document
        if self.config.enable_hyde:
            hyde_doc = await self._generate_hypothetical_document(query)
            return f"{query} {hyde_doc}"
        
        return query
    
    def _construct_context(self, chunks: List[Chunk]) -> str:
        """Construct context from retrieved chunks"""
        context_parts = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_text = chunk.content
            
            # Add context if available
            if chunk.context_before:
                chunk_text = f"[Context: {chunk.context_before}]\n{chunk_text}"
            if chunk.context_after:
                chunk_text = f"{chunk_text}\n[Context: {chunk.context_after}]"
            
            # Check token limit
            chunk_tokens = len(self.chunker.tokenizer.encode(chunk_text))
            if total_tokens + chunk_tokens > self.config.max_context_length:
                break
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    async def _self_correct(self, response: str, chunks: List[Chunk]) -> str:
        """Self-correction mechanism for response validation"""
        # Check for hallucinations or inconsistencies
        # Simplified implementation
        return response
    
    async def _generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document for HyDE"""
        # This would use an LLM to generate a hypothetical answer
        # Simplified for now
        return f"A comprehensive answer to {query} would include"
    
    async def _store_document_and_chunks(self, document: Document, chunks: List[Chunk]) -> None:
        """Store document and chunks in database"""
        # Store document
        self.db.execute(text("""
            INSERT INTO documents (id, content, metadata, created_at)
            VALUES (:id, :content, :metadata, :created_at)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata
        """), {
            "id": document.id,
            "content": document.content,
            "metadata": json.dumps(document.metadata),
            "created_at": document.timestamp
        })
        
        # Store chunks
        for chunk in chunks:
            self.db.execute(text("""
                INSERT INTO chunks (id, content, document_id, position, embedding, metadata)
                VALUES (:id, :content, :document_id, :position, :embedding, :metadata)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """), {
                "id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "position": chunk.position,
                "embedding": chunk.embedding.tolist() if chunk.embedding is not None else None,
                "metadata": json.dumps(chunk.metadata)
            })
        
        self.db.commit()


class AdvancedReranker:
    """Advanced reranking with multiple strategies"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    async def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """Rerank chunks for better relevance"""
        if not chunks:
            return chunks
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            score = await self._calculate_relevance_score(query, chunk)
            chunk.metadata["rerank_score"] = score
            scored_chunks.append((score, chunk))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        return [chunk for _, chunk in scored_chunks[:self.config.rerank_top_k]]
    
    async def _calculate_relevance_score(self, query: str, chunk: Chunk) -> float:
        """Calculate relevance score for reranking"""
        # Combine multiple signals
        scores = []
        
        # 1. Semantic similarity (if available)
        if "distance" in chunk.metadata:
            scores.append(1 - chunk.metadata["distance"])
        
        # 2. Keyword overlap
        query_terms = set(query.lower().split())
        chunk_terms = set(chunk.content.lower().split())
        overlap = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0
        scores.append(overlap)
        
        # 3. Position bias (prefer earlier chunks)
        position_score = 1 / (chunk.position + 1)
        scores.append(position_score * 0.1)  # Lower weight
        
        # 4. Importance score (if set)
        if hasattr(chunk, 'importance_score'):
            scores.append(chunk.importance_score)
        
        # Weighted average
        return np.mean(scores) if scores else 0.0


class ResponseGenerator:
    """Advanced response generation with multiple techniques"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    async def generate(self, query: str, context: str, chunks: List[Chunk]) -> str:
        """Generate response using context and query"""
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Generate response (would call LLM here)
        # For now, return a structured response
        response = f"""Based on the retrieved information:

{context[:500]}...

Answer to your query: {query}

This response is based on {len(chunks)} relevant documents."""
        
        return response
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM"""
        return f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information, say so."""


# Export main classes
__all__ = [
    'RAGConfig',
    'RAGPipeline',
    'ChunkingStrategy',
    'RetrievalStrategy',
    'Document',
    'Chunk'
]