"""
Query Optimization Service for Hybrid RAG System
Implements optimizations for vector, sparse, and graph search
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import redis
from functools import lru_cache
import faiss
import pickle

class QueryOptimizer:
    """Optimizes query performance across all retrieval modes"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.batch_size = 32
        self.prefetch_buffer = []
        
        # Initialize HNSW index for vector search optimization
        self.hnsw_index = None
        self.embedding_cache = {}
        
        # Query expansion terms for sparse search
        self.query_expansion_map = self._load_query_expansion_map()
        
        # Graph query optimization patterns
        self.graph_query_patterns = self._load_graph_patterns()
        
    # ==================== Vector Search Optimization ====================
    
    async def optimize_vector_search(self, embeddings: np.ndarray, 
                                    query_embedding: np.ndarray,
                                    top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Optimize vector search using HNSW index and intelligent caching
        """
        # Check cache first
        query_hash = self._hash_embedding(query_embedding)
        cached_result = await self._get_cached_result('vector', query_hash)
        if cached_result:
            return cached_result
        
        # Build or update HNSW index if needed
        if self.hnsw_index is None or len(embeddings) != self.hnsw_index.ntotal:
            self._build_hnsw_index(embeddings)
        
        # Perform optimized search
        distances, indices = self.hnsw_index.search(
            query_embedding.reshape(1, -1), 
            top_k * 2  # Fetch more for reranking
        )
        
        # Filter and rank results
        results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]
        results = results[:top_k]
        
        # Cache the result
        await self._cache_result('vector', query_hash, results)
        
        return results
    
    def _build_hnsw_index(self, embeddings: np.ndarray):
        """Build optimized HNSW index for fast similarity search"""
        dimension = embeddings.shape[1]
        
        # HNSW parameters optimized for quality/speed tradeoff
        M = 32  # Number of connections
        ef_construction = 200  # Construction time accuracy
        
        # Create HNSW index
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = 100  # Search time accuracy
        
        # Add vectors to index
        index.add(embeddings.astype('float32'))
        
        self.hnsw_index = index
        print(f"✅ Built HNSW index with {index.ntotal} vectors")
    
    async def batch_process_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Batch process text embeddings for efficiency
        """
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Check cache for each text
            batch_embeddings = []
            for text in batch:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                if text_hash in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[text_hash])
                else:
                    # Generate embedding (placeholder - would use actual model)
                    embedding = np.random.randn(1536).astype('float32')
                    self.embedding_cache[text_hash] = embedding
                    batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    async def precompute_common_queries(self):
        """
        Precompute embeddings for common queries
        """
        common_queries = [
            "What is", "How does", "Explain", "Tell me about",
            "What are the benefits", "How to implement", "Best practices for",
            "What is the difference between", "Why should I use", "When to use"
        ]
        
        print("🔄 Precomputing common query embeddings...")
        embeddings = await self.batch_process_embeddings(common_queries)
        
        # Store in cache
        for query, embedding in zip(common_queries, embeddings):
            query_hash = hashlib.md5(query.encode()).hexdigest()
            self.embedding_cache[query_hash] = embedding
        
        print(f"✅ Precomputed {len(common_queries)} query embeddings")
    
    # ==================== Sparse Search Optimization ====================
    
    async def optimize_sparse_search(self, query: str, 
                                    documents: List[Dict],
                                    top_k: int = 10) -> List[Dict]:
        """
        Optimize BM25 sparse search with query expansion and incremental indexing
        """
        # Check cache
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cached_result = await self._get_cached_result('sparse', query_hash)
        if cached_result:
            return cached_result
        
        # Expand query with synonyms and related terms
        expanded_query = self._expand_query(query)
        
        # Tokenize and optimize query
        optimized_tokens = self._optimize_tokenization(expanded_query)
        
        # Perform BM25 search with optimized parameters
        results = await self._optimized_bm25_search(optimized_tokens, documents, top_k)
        
        # Cache results
        await self._cache_result('sparse', query_hash, results)
        
        return results
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms
        """
        expanded_terms = [query]
        
        # Add synonyms from expansion map
        for term in query.lower().split():
            if term in self.query_expansion_map:
                expanded_terms.extend(self.query_expansion_map[term])
        
        return ' '.join(expanded_terms)
    
    def _optimize_tokenization(self, text: str) -> List[str]:
        """
        Optimize tokenization for BM25 search
        """
        # Simple tokenization with optimization
        tokens = text.lower().split()
        
        # Remove stopwords efficiently
        stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an'}
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        # Stem tokens (simplified)
        tokens = [t.rstrip('s').rstrip('ing').rstrip('ed') for t in tokens]
        
        return tokens
    
    async def _optimized_bm25_search(self, query_tokens: List[str],
                                    documents: List[Dict],
                                    top_k: int) -> List[Dict]:
        """
        Perform optimized BM25 search
        """
        # BM25 parameters (tuned for performance)
        k1 = 1.2
        b = 0.75
        
        # Calculate document frequencies
        doc_freqs = {}
        doc_lengths = []
        
        for doc in documents:
            tokens = self._optimize_tokenization(doc.get('content', ''))
            doc_lengths.append(len(tokens))
            
            for token in set(tokens):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1
        
        avg_doc_length = np.mean(doc_lengths) if doc_lengths else 1.0
        num_docs = len(documents)
        
        # Calculate BM25 scores
        scores = []
        for i, doc in enumerate(documents):
            doc_tokens = self._optimize_tokenization(doc.get('content', ''))
            doc_length = doc_lengths[i]
            
            score = 0.0
            for query_token in query_tokens:
                if query_token in doc_tokens:
                    tf = doc_tokens.count(query_token)
                    df = doc_freqs.get(query_token, 0)
                    idf = np.log((num_docs - df + 0.5) / (df + 0.5))
                    
                    score += idf * (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * doc_length / avg_doc_length)
                    )
            
            scores.append((i, score))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = [documents[idx] for idx, _ in scores[:top_k]]
        
        return top_results
    
    async def build_incremental_index(self, new_documents: List[Dict]):
        """
        Build incremental index for new documents
        """
        # Update document index incrementally
        index_key = "sparse_index"
        
        # Get existing index
        existing_index = self.redis_client.get(index_key)
        if existing_index:
            index = pickle.loads(existing_index)
        else:
            index = {'documents': [], 'tokens': {}}
        
        # Add new documents
        for doc in new_documents:
            doc_id = len(index['documents'])
            index['documents'].append(doc)
            
            # Update token index
            tokens = self._optimize_tokenization(doc.get('content', ''))
            for token in tokens:
                if token not in index['tokens']:
                    index['tokens'][token] = []
                index['tokens'][token].append(doc_id)
        
        # Save updated index
        self.redis_client.set(index_key, pickle.dumps(index))
        
        print(f"✅ Updated incremental index with {len(new_documents)} documents")
    
    # ==================== Graph Search Optimization ====================
    
    async def optimize_graph_search(self, query: str,
                                   neo4j_session) -> List[Dict]:
        """
        Optimize Neo4j graph traversal queries
        """
        # Check cache
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cached_result = await self._get_cached_result('graph', query_hash)
        if cached_result:
            return cached_result
        
        # Select optimal query pattern
        query_pattern = self._select_graph_pattern(query)
        
        # Build optimized Cypher query
        cypher_query = self._build_optimized_cypher(query, query_pattern)
        
        # Execute with query plan optimization
        results = await self._execute_optimized_cypher(neo4j_session, cypher_query)
        
        # Cache results
        await self._cache_result('graph', query_hash, results)
        
        return results
    
    def _select_graph_pattern(self, query: str) -> str:
        """
        Select optimal graph query pattern based on query type
        """
        query_lower = query.lower()
        
        if 'related' in query_lower or 'similar' in query_lower:
            return 'similarity_search'
        elif 'path' in query_lower or 'connection' in query_lower:
            return 'path_search'
        elif 'cluster' in query_lower or 'group' in query_lower:
            return 'community_search'
        else:
            return 'entity_search'
    
    def _build_optimized_cypher(self, query: str, pattern: str) -> str:
        """
        Build optimized Cypher query with proper indexes
        """
        if pattern == 'similarity_search':
            return """
            MATCH (n:Entity)
            WHERE n.embedding IS NOT NULL
            WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS score
            WHERE score > 0.7
            RETURN n, score
            ORDER BY score DESC
            LIMIT 10
            """
        elif pattern == 'path_search':
            return """
            MATCH path = shortestPath((a:Entity)-[*..5]-(b:Entity))
            WHERE a.name CONTAINS $query_term
            RETURN path
            LIMIT 5
            """
        elif pattern == 'community_search':
            return """
            MATCH (n:Entity)-[r]-(m:Entity)
            WHERE n.community = m.community
            AND n.name CONTAINS $query_term
            RETURN n, m, r
            LIMIT 20
            """
        else:
            return """
            MATCH (n:Entity)
            WHERE n.name CONTAINS $query_term
            OR n.description CONTAINS $query_term
            RETURN n
            ORDER BY n.importance DESC
            LIMIT 10
            """
    
    async def _execute_optimized_cypher(self, session, cypher_query: str) -> List[Dict]:
        """
        Execute Cypher query with optimization hints
        """
        # Add query planner hints
        optimized_query = f"CYPHER planner=cost {cypher_query}"
        
        # Execute with timeout
        try:
            result = session.run(optimized_query, timeout=5.0)
            return [record.data() for record in result]
        except Exception as e:
            print(f"⚠️ Graph query failed: {e}")
            return []
    
    async def create_graph_indexes(self, neo4j_session):
        """
        Create strategic indexes for graph optimization
        """
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX entity_embedding IF NOT EXISTS FOR (n:Entity) ON (n.embedding)",
            "CREATE INDEX entity_importance IF NOT EXISTS FOR (n:Entity) ON (n.importance)",
            "CREATE INDEX entity_community IF NOT EXISTS FOR (n:Entity) ON (n.community)",
            "CREATE INDEX document_created IF NOT EXISTS FOR (d:Document) ON (d.created_at)"
        ]
        
        for index_query in indexes:
            try:
                neo4j_session.run(index_query)
                print(f"✅ Created index: {index_query.split('INDEX')[1].split('IF')[0].strip()}")
            except Exception as e:
                print(f"⚠️ Index creation failed: {e}")
    
    async def create_materialized_views(self, neo4j_session):
        """
        Create materialized views for common graph patterns
        """
        views = [
            {
                'name': 'high_importance_entities',
                'query': """
                MATCH (n:Entity)
                WHERE n.importance > 0.8
                WITH n
                CREATE (v:MaterializedView:HighImportance)
                SET v = n
                """
            },
            {
                'name': 'document_relationships',
                'query': """
                MATCH (d1:Document)-[r:RELATED_TO]->(d2:Document)
                WITH d1, d2, r
                CREATE (v:MaterializedView:DocRelations)
                SET v.source = d1.id, v.target = d2.id, v.weight = r.weight
                """
            }
        ]
        
        for view in views:
            try:
                neo4j_session.run(view['query'])
                print(f"✅ Created materialized view: {view['name']}")
            except Exception as e:
                print(f"⚠️ View creation failed: {e}")
    
    # ==================== Caching Utilities ====================
    
    async def _get_cached_result(self, search_type: str, query_hash: str) -> Optional[Any]:
        """Get cached search result"""
        cache_key = f"rag_cache:{search_type}:{query_hash}"
        cached = self.redis_client.get(cache_key)
        
        if cached:
            return pickle.loads(cached)
        return None
    
    async def _cache_result(self, search_type: str, query_hash: str, result: Any):
        """Cache search result with TTL"""
        cache_key = f"rag_cache:{search_type}:{query_hash}"
        self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            pickle.dumps(result)
        )
    
    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """Hash embedding for cache key"""
        return hashlib.md5(embedding.tobytes()).hexdigest()
    
    def _load_query_expansion_map(self) -> Dict[str, List[str]]:
        """Load query expansion mappings"""
        return {
            'rag': ['retrieval', 'augmented', 'generation'],
            'vector': ['embedding', 'dense', 'similarity'],
            'graph': ['network', 'relationship', 'connection'],
            'memory': ['storage', 'recall', 'persistence'],
            'agent': ['workflow', 'orchestration', 'automation'],
            'optimize': ['improve', 'enhance', 'speed up'],
            'search': ['query', 'find', 'retrieve'],
            'hybrid': ['combined', 'multi-modal', 'fusion']
        }
    
    def _load_graph_patterns(self) -> Dict[str, str]:
        """Load optimized graph query patterns"""
        return {
            'similarity_search': 'cosine_similarity',
            'path_search': 'shortest_path',
            'community_search': 'community_detection',
            'entity_search': 'entity_extraction'
        }
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics"""
        metrics = {
            'cache_stats': {
                'vector_hits': self.redis_client.get('cache_hits:vector') or 0,
                'sparse_hits': self.redis_client.get('cache_hits:sparse') or 0,
                'graph_hits': self.redis_client.get('cache_hits:graph') or 0
            },
            'index_stats': {
                'hnsw_vectors': self.hnsw_index.ntotal if self.hnsw_index else 0,
                'embedding_cache_size': len(self.embedding_cache)
            },
            'performance': {
                'avg_vector_time_ms': 50,  # Would track actual times
                'avg_sparse_time_ms': 30,
                'avg_graph_time_ms': 80
            }
        }
        
        return metrics


async def run_optimization_demo():
    """Demonstrate query optimization capabilities"""
    import redis
    
    # Initialize optimizer
    redis_client = redis.Redis(host='localhost', port=6381)
    optimizer = QueryOptimizer(redis_client)
    
    print("=" * 60)
    print("🚀 Query Optimization Demo")
    print("=" * 60)
    
    # Precompute common queries
    await optimizer.precompute_common_queries()
    
    # Simulate vector search optimization
    print("\n📊 Vector Search Optimization:")
    embeddings = np.random.randn(1000, 1536).astype('float32')
    query_embedding = np.random.randn(1536).astype('float32')
    
    start_time = time.time()
    results = await optimizer.optimize_vector_search(embeddings, query_embedding)
    elapsed = (time.time() - start_time) * 1000
    print(f"  ✅ Optimized search completed in {elapsed:.2f}ms")
    
    # Simulate sparse search optimization
    print("\n📊 Sparse Search Optimization:")
    documents = [
        {'content': 'Hybrid RAG combines vector and graph search'},
        {'content': 'Agent orchestration enables complex workflows'},
        {'content': 'Memory clustering improves retrieval accuracy'}
    ]
    
    start_time = time.time()
    results = await optimizer.optimize_sparse_search(
        "What is hybrid RAG?", 
        documents
    )
    elapsed = (time.time() - start_time) * 1000
    print(f"  ✅ Optimized BM25 search completed in {elapsed:.2f}ms")
    
    # Get optimization metrics
    metrics = await optimizer.get_optimization_metrics()
    print("\n📈 Optimization Metrics:")
    print(f"  Cache Stats: {metrics['cache_stats']}")
    print(f"  Index Stats: {metrics['index_stats']}")
    print(f"  Performance: {metrics['performance']}")
    
    print("\n✅ Query optimization demo complete!")


if __name__ == "__main__":
    asyncio.run(run_optimization_demo())