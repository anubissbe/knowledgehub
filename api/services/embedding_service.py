"""
Advanced Embedding Service for AI Memory System.

This service handles text-to-vector embedding generation using multiple models,
caching, and optimization for memory retrieval and clustering.
"""

import logging
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Try to import embedding libraries, with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..services.cache import redis_client
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("embedding_service")


class EmbeddingModel(str, Enum):
    """Available embedding models."""
    SENTENCE_BERT = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_BERT_LARGE = "sentence-transformers/all-mpnet-base-v2"
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    HUGGINGFACE_BERT = "bert-base-uncased"
    CODE_BERT = "microsoft/codebert-base"


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: List[float]
    model: str
    dimensions: int
    processing_time: float
    cached: bool
    metadata: Dict[str, Any]


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""
    
    def __init__(self, redis_client, cache_ttl: int = 86400):  # 24 hours default
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.cache_prefix = "embedding:"
    
    def _get_cache_key(self, text: str, model: str, version: str = "v1.0") -> str:
        """Generate cache key for text + model combination."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{self.cache_prefix}{model}:{version}:{text_hash}"
    
    async def get(self, text: str, model: str, version: str = "v1.0") -> Optional[List[float]]:
        """Get cached embedding."""
        try:
            cache_key = self._get_cache_key(text, model, version)
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(self, text: str, model: str, embeddings: List[float], version: str = "v1.0"):
        """Cache embeddings."""
        try:
            cache_key = self._get_cache_key(text, model, version)
            await self.redis.set(cache_key, json.dumps(embeddings), expiry=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")


class EmbeddingService:
    """
    Advanced embedding service supporting multiple models and optimization.
    
    Features:
    - Multiple embedding models (Sentence-BERT, OpenAI, Hugging Face)
    - Caching for performance
    - Batch processing
    - Memory optimization
    - Model fallbacks
    - Context-aware embeddings
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.cache = EmbeddingCache(redis_client)
        self.models = {}
        self.model_info = {}
        self._initialized = False
        
        # Model configurations
        self.model_configs = {
            EmbeddingModel.SENTENCE_BERT: {
                "dimensions": 384,
                "max_length": 512,
                "available": SENTENCE_TRANSFORMERS_AVAILABLE
            },
            EmbeddingModel.SENTENCE_BERT_LARGE: {
                "dimensions": 768,
                "max_length": 512,
                "available": SENTENCE_TRANSFORMERS_AVAILABLE
            },
            EmbeddingModel.OPENAI_ADA: {
                "dimensions": 1536,
                "max_length": 8192,
                "available": OPENAI_AVAILABLE and bool(self.config.OPENAI_API_KEY)
            },
            EmbeddingModel.OPENAI_3_SMALL: {
                "dimensions": 1536,
                "max_length": 8192,
                "available": OPENAI_AVAILABLE and bool(self.config.OPENAI_API_KEY)
            },
            EmbeddingModel.OPENAI_3_LARGE: {
                "dimensions": 3072,
                "max_length": 8192,
                "available": OPENAI_AVAILABLE and bool(self.config.OPENAI_API_KEY)
            }
        }
        
        logger.info("Initialized EmbeddingService")
    
    async def initialize(self, default_model: Optional[EmbeddingModel] = None):
        """Initialize embedding models."""
        if self._initialized:
            return
        
        try:
            await self.cache.redis.initialize()
            
            # Set default model
            self.default_model = default_model or EmbeddingModel.SENTENCE_BERT
            
            # Initialize available models
            await self._load_available_models()
            
            self._initialized = True
            logger.info(f"EmbeddingService initialized with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}")
            raise
    
    async def _load_available_models(self):
        """Load available embedding models based on installed libraries."""
        
        # Load Sentence-BERT models
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                for model_name in [EmbeddingModel.SENTENCE_BERT, EmbeddingModel.SENTENCE_BERT_LARGE]:
                    if self.model_configs[model_name]["available"]:
                        model = SentenceTransformer(model_name.value)
                        self.models[model_name] = model
                        self.model_info[model_name] = {
                            "type": "sentence_bert",
                            "loaded": True,
                            "dimensions": self.model_configs[model_name]["dimensions"]
                        }
                        logger.info(f"Loaded Sentence-BERT model: {model_name.value}")
            except Exception as e:
                logger.warning(f"Failed to load Sentence-BERT models: {e}")
        
        # Configure OpenAI models (no loading required)
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                openai.api_key = self.config.OPENAI_API_KEY
                for model_name in [EmbeddingModel.OPENAI_ADA, EmbeddingModel.OPENAI_3_SMALL, EmbeddingModel.OPENAI_3_LARGE]:
                    self.model_info[model_name] = {
                        "type": "openai",
                        "loaded": True,
                        "dimensions": self.model_configs[model_name]["dimensions"]
                    }
                logger.info("Configured OpenAI embedding models")
            except Exception as e:
                logger.warning(f"Failed to configure OpenAI models: {e}")
        
        # Load HuggingFace models
        if TRANSFORMERS_AVAILABLE:
            try:
                # We'll load these on-demand to save memory
                self.model_info[EmbeddingModel.HUGGINGFACE_BERT] = {
                    "type": "huggingface",
                    "loaded": False,
                    "dimensions": 768
                }
                self.model_info[EmbeddingModel.CODE_BERT] = {
                    "type": "huggingface",
                    "loaded": False,
                    "dimensions": 768
                }
                logger.info("Configured HuggingFace embedding models")
            except Exception as e:
                logger.warning(f"Failed to configure HuggingFace models: {e}")
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[EmbeddingModel] = None,
        use_cache: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            model: Embedding model to use
            use_cache: Whether to use caching
            context: Additional context for context-aware embeddings
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        model = model or self.default_model
        
        if not self._initialized:
            await self.initialize()
        
        # Prepare text
        processed_text = self._preprocess_text(text, context)
        
        # Check cache first
        cached_embeddings = None
        if use_cache:
            cached_embeddings = await self.cache.get(processed_text, model.value)
            if cached_embeddings:
                return EmbeddingResult(
                    embeddings=cached_embeddings,
                    model=model.value,
                    dimensions=len(cached_embeddings),
                    processing_time=time.time() - start_time,
                    cached=True,
                    metadata={"context": context}
                )
        
        # Generate new embedding
        try:
            embeddings = await self._generate_embedding_by_model(processed_text, model)
            
            # Cache result
            if use_cache:
                await self.cache.set(processed_text, model.value, embeddings)
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embeddings=embeddings,
                model=model.value,
                dimensions=len(embeddings),
                processing_time=processing_time,
                cached=False,
                metadata={"context": context, "text_length": len(text)}
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed for model {model.value}: {e}")
            
            # Try fallback model
            fallback_model = self._get_fallback_model(model)
            if fallback_model and fallback_model != model:
                logger.info(f"Trying fallback model: {fallback_model.value}")
                return await self.generate_embedding(text, fallback_model, use_cache, context)
            
            raise
    
    async def generate_batch_embeddings(
        self,
        texts: List[str],
        model: Optional[EmbeddingModel] = None,
        use_cache: bool = True,
        context: Optional[Dict[str, Any]] = None,
        batch_size: int = 32
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts efficiently."""
        
        if not texts:
            return []
        
        model = model or self.default_model
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.generate_embedding(text, model, use_cache, context)
                for text in batch_texts
            ])
            results.extend(batch_results)
        
        return results
    
    def _preprocess_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Preprocess text for embedding generation."""
        # Basic cleaning
        processed = text.strip()
        
        # Add context if provided
        if context:
            context_str = self._format_context(context)
            if context_str:
                processed = f"{context_str}\n\n{processed}"
        
        return processed
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context into text for context-aware embeddings."""
        context_parts = []
        
        # Add session context
        if "session_id" in context:
            context_parts.append(f"Session: {context['session_id']}")
        
        # Add memory type context
        if "memory_type" in context:
            context_parts.append(f"Type: {context['memory_type']}")
        
        # Add tags context
        if "tags" in context and context["tags"]:
            context_parts.append(f"Tags: {', '.join(context['tags'])}")
        
        # Add temporal context
        if "timestamp" in context:
            context_parts.append(f"Time: {context['timestamp']}")
        
        return "Context: " + " | ".join(context_parts) if context_parts else ""
    
    async def _generate_embedding_by_model(self, text: str, model: EmbeddingModel) -> List[float]:
        """Generate embedding using specific model."""
        
        if model not in self.model_info:
            raise ValueError(f"Model {model.value} not available")
        
        model_info = self.model_info[model]
        
        if model_info["type"] == "sentence_bert":
            return await self._generate_sentence_bert_embedding(text, model)
        elif model_info["type"] == "openai":
            return await self._generate_openai_embedding(text, model)
        elif model_info["type"] == "huggingface":
            return await self._generate_huggingface_embedding(text, model)
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
    
    async def _generate_sentence_bert_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Generate embedding using Sentence-BERT."""
        if model not in self.models:
            raise ValueError(f"Sentence-BERT model {model.value} not loaded")
        
        embedding_model = self.models[model]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: embedding_model.encode([text])
        )
        
        return embeddings[0].tolist()
    
    async def _generate_openai_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = await asyncio.to_thread(
                openai.Embedding.create,
                input=text,
                model=model.value
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def _generate_huggingface_embedding(self, text: str, model: EmbeddingModel) -> List[float]:
        """Generate embedding using HuggingFace model."""
        # Load model on-demand
        if model not in self.models:
            tokenizer = AutoTokenizer.from_pretrained(model.value)
            model_obj = AutoModel.from_pretrained(model.value)
            self.models[model] = (tokenizer, model_obj)
        
        tokenizer, model_obj = self.models[model]
        
        # Tokenize and encode
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model_obj(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings[0].tolist()
    
    def _get_fallback_model(self, failed_model: EmbeddingModel) -> Optional[EmbeddingModel]:
        """Get fallback model if primary model fails."""
        fallback_order = [
            EmbeddingModel.SENTENCE_BERT,
            EmbeddingModel.OPENAI_ADA,
            EmbeddingModel.SENTENCE_BERT_LARGE,
            EmbeddingModel.HUGGINGFACE_BERT
        ]
        
        for model in fallback_order:
            if model != failed_model and model in self.model_info:
                return model
        
        return None
    
    async def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        return {
            model.value: {
                **info,
                "config": self.model_configs.get(model, {})
            }
            for model, info in self.model_info.items()
        }
    
    async def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        method: str = "cosine"
    ) -> float:
        """Calculate similarity between two embeddings."""
        
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have same dimensions")
        
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)
        
        if method == "cosine":
            # Cosine similarity
            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        
        elif method == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(arr1 - arr2)
            return float(1 / (1 + distance))
        
        elif method == "manhattan":
            # Manhattan distance (converted to similarity)
            distance = np.sum(np.abs(arr1 - arr2))
            return float(1 / (1 + distance))
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    async def find_similar_embeddings(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 10,
        method: str = "cosine"
    ) -> List[Tuple[int, float]]:
        """Find most similar embeddings from candidates."""
        
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.calculate_similarity(
                query_embedding, candidate, method
            )
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    async def cleanup(self):
        """Clean up resources."""
        # Clear loaded models to free memory
        self.models.clear()
        self.model_info.clear()
        self._initialized = False
        logger.info("EmbeddingService cleaned up")


# Global embedding service instance
embedding_service = EmbeddingService()