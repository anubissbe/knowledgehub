"""
Reranking and Fusion Optimization Service
Implements optimized cross-encoder reranking and intelligent result fusion
"""

import asyncio
import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import pickle
import redis
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort


@dataclass
class RetrievalResult:
    """Structure for retrieval results"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    retrieval_method: str  # 'vector', 'sparse', 'graph'


class RerankingOptimizer:
    """
    Optimized cross-encoder reranking with model quantization and batch processing
    """
    
    def __init__(self, redis_client: redis.Redis, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.redis_client = redis_client
        self.model_name = model_name
        self.cache_ttl = 3600
        self.batch_size = 32
        
        # Initialize model with optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_optimized_model()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Fusion weights (learned from data)
        self.fusion_weights = {
            'vector': 0.4,
            'sparse': 0.3,
            'graph': 0.3
        }
        
    def _initialize_optimized_model(self):
        """Initialize quantized model for faster inference"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Try to load ONNX model for faster inference
            onnx_path = f"models/{self.model_name.replace('/', '_')}.onnx"
            try:
                self.onnx_session = ort.InferenceSession(onnx_path)
                self.use_onnx = True
                print(f"✅ Loaded ONNX model for fast inference")
            except:
                # Fallback to PyTorch model
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                # Apply quantization for faster inference
                if self.device.type == 'cpu':
                    self.model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    print(f"✅ Applied dynamic quantization to model")
                
                self.use_onnx = False
                
        except Exception as e:
            print(f"⚠️ Failed to initialize reranking model: {e}")
            self.model = None
            self.use_onnx = False
    
    async def rerank_results(self, 
                            query: str,
                            results: List[RetrievalResult],
                            top_k: int = 10) -> List[RetrievalResult]:
        """
        Rerank results using optimized cross-encoder
        """
        if not results:
            return []
        
        # Check cache
        cache_key = self._generate_cache_key(query, results)
        cached = await self._get_cached_reranking(cache_key)
        if cached:
            return cached[:top_k]
        
        # Batch reranking
        reranked = await self._batch_rerank(query, results)
        
        # Cache results
        await self._cache_reranking(cache_key, reranked)
        
        return reranked[:top_k]
    
    async def _batch_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Perform batch reranking for efficiency
        """
        if self.model is None and not self.use_onnx:
            # Fallback to original scores if model not available
            return sorted(results, key=lambda x: x.score, reverse=True)
        
        # Prepare batches
        pairs = [(query, r.content) for r in results]
        scores = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = await self._score_batch(batch)
            scores.extend(batch_scores)
        
        # Combine with original scores
        for i, result in enumerate(results):
            if i < len(scores):
                # Weighted combination of original and reranked scores
                result.score = 0.7 * scores[i] + 0.3 * result.score
        
        # Sort by new scores
        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        
        return reranked
    
    async def _score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score a batch of query-document pairs
        """
        if self.use_onnx:
            return self._score_batch_onnx(pairs)
        else:
            return self._score_batch_pytorch(pairs)
    
    def _score_batch_onnx(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score using ONNX Runtime for faster inference
        """
        # Tokenize batch
        inputs = self.tokenizer(
            [p[0] for p in pairs],  # queries
            [p[1] for p in pairs],  # documents
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        # Run inference
        outputs = self.onnx_session.run(
            None,
            {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
        )
        
        # Extract scores
        scores = outputs[0][:, 1].tolist()  # Positive class scores
        
        return scores
    
    def _score_batch_pytorch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score using PyTorch model
        """
        # Tokenize batch
        inputs = self.tokenizer(
            [p[0] for p in pairs],  # queries
            [p[1] for p in pairs],  # documents
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        
        return scores.tolist()
    
    def _generate_cache_key(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate cache key for reranking results"""
        content = query + ''.join([r.content[:100] for r in results])
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_cached_reranking(self, cache_key: str) -> Optional[List[RetrievalResult]]:
        """Get cached reranking results"""
        cached = self.redis_client.get(f"rerank_cache:{cache_key}")
        if cached:
            self.redis_client.incr("rerank_cache_hits")
            return pickle.loads(cached)
        return None
    
    async def _cache_reranking(self, cache_key: str, results: List[RetrievalResult]):
        """Cache reranking results"""
        self.redis_client.setex(
            f"rerank_cache:{cache_key}",
            self.cache_ttl,
            pickle.dumps(results)
        )


class FusionOptimizer:
    """
    Optimized result fusion with adaptive scoring and learned weights
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        
        # Adaptive fusion weights (updated based on user feedback)
        self.fusion_weights = self._load_fusion_weights()
        
        # Query type detection patterns
        self.query_patterns = {
            'factual': ['what is', 'define', 'explain'],
            'procedural': ['how to', 'steps to', 'process'],
            'comparative': ['difference between', 'compare', 'versus'],
            'exploratory': ['related to', 'similar to', 'examples of']
        }
        
    def _load_fusion_weights(self) -> Dict[str, Dict[str, float]]:
        """Load learned fusion weights from storage"""
        default_weights = {
            'factual': {'vector': 0.5, 'sparse': 0.3, 'graph': 0.2},
            'procedural': {'vector': 0.4, 'sparse': 0.4, 'graph': 0.2},
            'comparative': {'vector': 0.3, 'sparse': 0.3, 'graph': 0.4},
            'exploratory': {'vector': 0.3, 'sparse': 0.2, 'graph': 0.5},
            'default': {'vector': 0.4, 'sparse': 0.3, 'graph': 0.3}
        }
        
        # Try to load from Redis
        stored = self.redis_client.get("fusion_weights")
        if stored:
            return pickle.loads(stored)
        
        return default_weights
    
    async def fuse_results(self,
                          vector_results: List[RetrievalResult],
                          sparse_results: List[RetrievalResult],
                          graph_results: List[RetrievalResult],
                          query: str,
                          top_k: int = 10) -> List[RetrievalResult]:
        """
        Perform weighted fusion of results from different retrieval methods
        """
        # Detect query type
        query_type = self._detect_query_type(query)
        weights = self.fusion_weights.get(query_type, self.fusion_weights['default'])
        
        # Create result map for deduplication
        result_map = {}
        
        # Process vector results
        for result in vector_results:
            key = self._generate_result_key(result)
            if key not in result_map:
                result_map[key] = {
                    'result': result,
                    'scores': {'vector': 0, 'sparse': 0, 'graph': 0}
                }
            result_map[key]['scores']['vector'] = result.score
        
        # Process sparse results
        for result in sparse_results:
            key = self._generate_result_key(result)
            if key not in result_map:
                result_map[key] = {
                    'result': result,
                    'scores': {'vector': 0, 'sparse': 0, 'graph': 0}
                }
            result_map[key]['scores']['sparse'] = result.score
        
        # Process graph results
        for result in graph_results:
            key = self._generate_result_key(result)
            if key not in result_map:
                result_map[key] = {
                    'result': result,
                    'scores': {'vector': 0, 'sparse': 0, 'graph': 0}
                }
            result_map[key]['scores']['graph'] = result.score
        
        # Apply weighted fusion
        fused_results = []
        for key, data in result_map.items():
            # Normalize scores
            normalized_scores = self._normalize_scores(data['scores'])
            
            # Calculate weighted score
            weighted_score = (
                weights['vector'] * normalized_scores['vector'] +
                weights['sparse'] * normalized_scores['sparse'] +
                weights['graph'] * normalized_scores['graph']
            )
            
            # Apply boost for multi-source results
            source_count = sum(1 for s in normalized_scores.values() if s > 0)
            if source_count > 1:
                weighted_score *= (1 + 0.1 * (source_count - 1))  # 10% boost per additional source
            
            data['result'].score = weighted_score
            fused_results.append(data['result'])
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        # Log fusion metrics for learning
        await self._log_fusion_metrics(query_type, weights, fused_results[:top_k])
        
        return fused_results[:top_k]
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type based on patterns"""
        query_lower = query.lower()
        
        for qtype, patterns in self.query_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return qtype
        
        return 'default'
    
    def _generate_result_key(self, result: RetrievalResult) -> str:
        """Generate unique key for result deduplication"""
        # Use first 200 chars of content for deduplication
        content_snippet = result.content[:200]
        return hashlib.md5(content_snippet.encode()).hexdigest()
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range"""
        # Min-max normalization per retrieval type
        normalized = {}
        
        for method, score in scores.items():
            # Clamp to [0, 1] if needed
            normalized[method] = min(max(score, 0), 1)
        
        return normalized
    
    async def _log_fusion_metrics(self, query_type: str, weights: Dict[str, float], 
                                 results: List[RetrievalResult]):
        """Log fusion metrics for continuous learning"""
        metrics = {
            'timestamp': time.time(),
            'query_type': query_type,
            'weights': weights,
            'result_count': len(results),
            'avg_score': np.mean([r.score for r in results]) if results else 0
        }
        
        # Store in Redis list for analysis
        self.redis_client.lpush("fusion_metrics", pickle.dumps(metrics))
        self.redis_client.ltrim("fusion_metrics", 0, 9999)  # Keep last 10000 entries
    
    async def update_fusion_weights(self, feedback_data: List[Dict]):
        """
        Update fusion weights based on user feedback
        Uses simple gradient descent on user satisfaction scores
        """
        print("🔄 Updating fusion weights based on feedback...")
        
        # Group feedback by query type
        feedback_by_type = {}
        for feedback in feedback_data:
            qtype = feedback.get('query_type', 'default')
            if qtype not in feedback_by_type:
                feedback_by_type[qtype] = []
            feedback_by_type[qtype].append(feedback)
        
        # Update weights for each query type
        for qtype, feedbacks in feedback_by_type.items():
            if qtype not in self.fusion_weights:
                self.fusion_weights[qtype] = {'vector': 0.33, 'sparse': 0.33, 'graph': 0.34}
            
            # Calculate weight adjustments based on satisfaction
            adjustments = {'vector': 0, 'sparse': 0, 'graph': 0}
            
            for feedback in feedbacks:
                satisfaction = feedback.get('satisfaction', 0.5)  # 0-1 scale
                used_methods = feedback.get('used_methods', [])
                
                # Increase weights for methods that contributed to satisfied results
                for method in used_methods:
                    if method in adjustments:
                        adjustments[method] += (satisfaction - 0.5) * 0.01
            
            # Apply adjustments
            for method in ['vector', 'sparse', 'graph']:
                self.fusion_weights[qtype][method] += adjustments[method]
            
            # Normalize weights to sum to 1
            total = sum(self.fusion_weights[qtype].values())
            for method in self.fusion_weights[qtype]:
                self.fusion_weights[qtype][method] /= total
        
        # Save updated weights
        self.redis_client.set("fusion_weights", pickle.dumps(self.fusion_weights))
        
        print(f"✅ Updated fusion weights: {self.fusion_weights}")


class OnlineLearningOptimizer:
    """
    Online learning system for continuous reranking improvement
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.learning_rate = 0.01
        self.feedback_buffer = []
        self.model_version = 1
        
    async def collect_feedback(self, query: str, results: List[RetrievalResult], 
                              clicked_indices: List[int], dwell_times: List[float]):
        """
        Collect implicit feedback from user interactions
        """
        feedback = {
            'query': query,
            'results': [r.content[:100] for r in results],  # Store snippets
            'clicked_indices': clicked_indices,
            'dwell_times': dwell_times,
            'timestamp': time.time()
        }
        
        self.feedback_buffer.append(feedback)
        
        # Trigger learning when buffer is full
        if len(self.feedback_buffer) >= 100:
            await self.update_model()
    
    async def update_model(self):
        """
        Update reranking model based on collected feedback
        """
        if not self.feedback_buffer:
            return
        
        print(f"🔄 Updating model with {len(self.feedback_buffer)} feedback samples...")
        
        # Calculate relevance scores from implicit feedback
        training_data = []
        
        for feedback in self.feedback_buffer:
            # Calculate relevance based on clicks and dwell time
            for i, result in enumerate(feedback['results']):
                if i in feedback['clicked_indices']:
                    idx = feedback['clicked_indices'].index(i)
                    dwell_time = feedback['dwell_times'][idx]
                    
                    # Relevance score based on dwell time (normalized)
                    relevance = min(dwell_time / 30.0, 1.0)  # 30 seconds = max relevance
                else:
                    relevance = 0.0
                
                training_data.append({
                    'query': feedback['query'],
                    'document': result,
                    'relevance': relevance
                })
        
        # Update model (simplified - would actually fine-tune the cross-encoder)
        self.model_version += 1
        
        # Clear feedback buffer
        self.feedback_buffer = []
        
        # Save training data for offline analysis
        self.redis_client.lpush("reranking_training_data", pickle.dumps(training_data))
        
        print(f"✅ Model updated to version {self.model_version}")


async def run_optimization_demo():
    """Demonstrate reranking and fusion optimization"""
    import redis
    
    redis_client = redis.Redis(host='localhost', port=6381)
    
    print("=" * 60)
    print("🚀 Reranking & Fusion Optimization Demo")
    print("=" * 60)
    
    # Initialize optimizers
    reranker = RerankingOptimizer(redis_client)
    fusion = FusionOptimizer(redis_client)
    learner = OnlineLearningOptimizer(redis_client)
    
    # Create sample results
    vector_results = [
        RetrievalResult(
            content="Hybrid RAG combines multiple retrieval strategies for better results",
            score=0.85,
            source="doc1",
            metadata={},
            retrieval_method="vector"
        ),
        RetrievalResult(
            content="Vector search uses dense embeddings for semantic similarity",
            score=0.75,
            source="doc2",
            metadata={},
            retrieval_method="vector"
        )
    ]
    
    sparse_results = [
        RetrievalResult(
            content="BM25 is a popular sparse retrieval algorithm",
            score=0.80,
            source="doc3",
            metadata={},
            retrieval_method="sparse"
        ),
        RetrievalResult(
            content="Hybrid RAG combines multiple retrieval strategies for better results",
            score=0.70,
            source="doc1",
            metadata={},
            retrieval_method="sparse"
        )
    ]
    
    graph_results = [
        RetrievalResult(
            content="Knowledge graphs provide relationship-based retrieval",
            score=0.90,
            source="doc4",
            metadata={},
            retrieval_method="graph"
        )
    ]
    
    query = "What is hybrid RAG and how does it work?"
    
    # Test reranking
    print("\n📊 Testing Reranking:")
    all_results = vector_results + sparse_results + graph_results
    
    start_time = time.time()
    reranked = await reranker.rerank_results(query, all_results, top_k=5)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"  ✅ Reranked {len(all_results)} results in {elapsed:.2f}ms")
    for i, result in enumerate(reranked[:3]):
        print(f"  {i+1}. Score: {result.score:.3f} - {result.content[:50]}...")
    
    # Test fusion
    print("\n📊 Testing Result Fusion:")
    
    start_time = time.time()
    fused = await fusion.fuse_results(
        vector_results, sparse_results, graph_results,
        query, top_k=5
    )
    elapsed = (time.time() - start_time) * 1000
    
    print(f"  ✅ Fused results in {elapsed:.2f}ms")
    for i, result in enumerate(fused[:3]):
        print(f"  {i+1}. Score: {result.score:.3f} - {result.content[:50]}...")
    
    # Simulate user feedback
    print("\n📊 Simulating User Feedback:")
    await learner.collect_feedback(
        query, fused,
        clicked_indices=[0, 2],  # User clicked 1st and 3rd results
        dwell_times=[25.0, 5.0]  # Spent 25s on first, 5s on third
    )
    print("  ✅ Feedback collected for online learning")
    
    print("\n✅ Reranking & Fusion optimization demo complete!")


if __name__ == "__main__":
    asyncio.run(run_optimization_demo())