"""
Resilient ML Service with Circuit Breaker Protection

Provides a resilient interface to ML model services with automatic
fallback, caching, and performance optimization.
"""

import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
from sentence_transformers import SentenceTransformer

from .external_service_client import ml_model_client, redis_client
from .circuit_breaker import circuit_breaker, ServiceType, CircuitConfig
from ..config import settings

logger = logging.getLogger(__name__)


class ResilientMLService:
    """
    Resilient ML service with circuit breaker protection.
    
    Features:
    - Circuit breaker protection for ML services
    - Local model fallback
    - Result caching
    - Batch processing optimization
    - Performance monitoring
    """
    
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour for embeddings
        self.enable_cache = True
        self.enable_local_fallback = True
        self.batch_size = 32
        
        # Local fallback model
        self.local_model: Optional[SentenceTransformer] = None
        self.model_name = settings.EMBEDDING_MODEL
        
        # Performance tracking
        self.embedding_count = 0
        self.cache_hits = 0
        self.fallback_count = 0
        self.total_tokens_processed = 0
    
    def _initialize_local_model(self):
        """Initialize local fallback model"""
        if not self.local_model and self.enable_local_fallback:
            try:
                logger.info(f"Loading local fallback model: {self.model_name}")
                self.local_model = SentenceTransformer(self.model_name)
                logger.info("Local fallback model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                self.enable_local_fallback = False
    
    async def generate_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings with resilient ML service.
        
        Attempts in order:
        1. Cache lookup
        2. Remote ML service
        3. Local model fallback
        """
        self.embedding_count += 1
        self.total_tokens_processed += sum(len(text.split()) for text in texts)
        
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache and self.enable_cache:
            for i, text in enumerate(texts):
                cache_key = self._generate_embedding_cache_key(text)
                cached_embedding = await self._get_from_cache(cache_key)
                
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    self.cache_hits += 1
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                # Try remote ML service
                new_embeddings = await self._generate_remote_embeddings(uncached_texts)
                
                # Update results and cache
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding
                    
                    if use_cache and self.enable_cache:
                        cache_key = self._generate_embedding_cache_key(texts[idx])
                        await self._save_to_cache(cache_key, embedding)
                
            except Exception as e:
                logger.error(f"Remote ML service failed: {e}")
                
                # Try local fallback
                if self.enable_local_fallback:
                    self.fallback_count += 1
                    fallback_embeddings = await self._generate_local_embeddings(uncached_texts)
                    
                    # Update results
                    for idx, embedding in zip(uncached_indices, fallback_embeddings):
                        embeddings[idx] = embedding
                else:
                    # Generate zero embeddings as last resort
                    for idx in uncached_indices:
                        embeddings[idx] = [0.0] * settings.EMBEDDING_DIMENSION
        
        return embeddings
    
    @circuit_breaker(
        "ml_inference",
        service_type=ServiceType.ML_MODEL,
        config=CircuitConfig(
            failure_threshold=3,
            timeout=120.0,
            error_timeout=60.0
        )
    )
    async def _generate_remote_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings using remote ML service with circuit breaker"""
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = await ml_model_client.post(
                "/embeddings",
                json_data={
                    "texts": batch,
                    "model": self.model_name
                }
            )
            
            if "embeddings" in response:
                all_embeddings.extend(response["embeddings"])
            else:
                raise ValueError("Invalid response from ML service")
        
        return all_embeddings
    
    async def _generate_local_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings using local fallback model"""
        logger.info(f"Using local fallback model for {len(texts)} texts")
        
        # Initialize model if needed
        if not self.local_model:
            self._initialize_local_model()
        
        if not self.local_model:
            # Return zero embeddings if model unavailable
            return [[0.0] * settings.EMBEDDING_DIMENSION] * len(texts)
        
        try:
            # Generate embeddings with local model
            embeddings = self.local_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Convert to list format
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            return [[0.0] * settings.EMBEDDING_DIMENSION] * len(texts)
    
    async def classify_text(
        self,
        text: str,
        labels: List[str],
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Classify text with resilient ML service.
        
        Returns confidence scores for each label.
        """
        # Generate cache key
        cache_key = self._generate_classification_cache_key(text, labels)
        
        # Check cache
        if use_cache and self.enable_cache:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self.cache_hits += 1
                return cached_result
        
        try:
            # Try remote ML service
            result = await self._classify_remote(text, labels)
            
            # Cache result
            if use_cache and self.enable_cache:
                await self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Remote classification failed: {e}")
            
            # Fallback to embedding-based classification
            return await self._classify_fallback(text, labels)
    
    @circuit_breaker(
        "ml_inference",
        service_type=ServiceType.ML_MODEL,
        config=CircuitConfig(
            failure_threshold=3,
            timeout=60.0
        )
    )
    async def _classify_remote(
        self,
        text: str,
        labels: List[str]
    ) -> Dict[str, float]:
        """Classify text using remote ML service with circuit breaker"""
        
        response = await ml_model_client.post(
            "/classify",
            json_data={
                "text": text,
                "labels": labels,
                "model": "zero-shot-classification"
            }
        )
        
        if "scores" in response:
            return dict(zip(labels, response["scores"]))
        else:
            raise ValueError("Invalid response from ML service")
    
    async def _classify_fallback(
        self,
        text: str,
        labels: List[str]
    ) -> Dict[str, float]:
        """Fallback classification using embeddings"""
        logger.info("Using embedding-based fallback classification")
        
        # Generate embeddings for text and labels
        all_texts = [text] + labels
        embeddings = await self.generate_embeddings(all_texts, use_cache=False)
        
        text_embedding = np.array(embeddings[0])
        label_embeddings = np.array(embeddings[1:])
        
        # Calculate cosine similarities
        similarities = []
        for label_embedding in label_embeddings:
            similarity = np.dot(text_embedding, label_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(label_embedding)
            )
            similarities.append(max(0, similarity))  # Ensure non-negative
        
        # Normalize to sum to 1
        total = sum(similarities)
        if total > 0:
            scores = [s / total for s in similarities]
        else:
            scores = [1.0 / len(labels)] * len(labels)
        
        return dict(zip(labels, scores))
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text with resilient ML service.
        """
        try:
            # Try remote ML service
            return await self._extract_entities_remote(text, entity_types)
        except Exception as e:
            logger.error(f"Remote entity extraction failed: {e}")
            
            # Fallback to pattern-based extraction
            return self._extract_entities_fallback(text, entity_types)
    
    @circuit_breaker(
        "ml_inference",
        service_type=ServiceType.ML_MODEL,
        config=CircuitConfig(failure_threshold=3, timeout=60.0)
    )
    async def _extract_entities_remote(
        self,
        text: str,
        entity_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Extract entities using remote ML service with circuit breaker"""
        
        response = await ml_model_client.post(
            "/entities",
            json_data={
                "text": text,
                "entity_types": entity_types or ["PERSON", "ORG", "LOC", "DATE"]
            }
        )
        
        if "entities" in response:
            return response["entities"]
        else:
            raise ValueError("Invalid response from ML service")
    
    def _extract_entities_fallback(
        self,
        text: str,
        entity_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Fallback entity extraction using patterns"""
        logger.info("Using pattern-based fallback entity extraction")
        
        entities = []
        
        # Simple pattern-based extraction
        import re
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "EMAIL",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # URL patterns
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for match in re.finditer(url_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "URL",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # Date patterns (simple)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        for match in re.finditer(date_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "DATE",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.7
            })
        
        return entities
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get result from cache"""
        try:
            cached = await redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        return None
    
    async def _save_to_cache(self, key: str, value: Any):
        """Save result to cache"""
        try:
            await redis_client.set(
                key,
                json.dumps(value),
                ttl=self.cache_ttl
            )
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")
    
    def _generate_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{self.model_name}:{text_hash}"
    
    def _generate_classification_cache_key(
        self,
        text: str,
        labels: List[str]
    ) -> str:
        """Generate cache key for classification"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        labels_hash = hashlib.md5(
            json.dumps(sorted(labels)).encode()
        ).hexdigest()[:16]
        return f"classify:{text_hash}:{labels_hash}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ML service statistics"""
        cache_hit_rate = (
            (self.cache_hits / self.embedding_count * 100)
            if self.embedding_count > 0 else 0
        )
        
        fallback_rate = (
            (self.fallback_count / self.embedding_count * 100)
            if self.embedding_count > 0 else 0
        )
        
        # Get circuit breaker status
        from .circuit_breaker import circuit_manager
        breaker = circuit_manager.get_circuit_breaker("ml_inference")
        breaker_status = breaker.get_status() if breaker else None
        
        return {
            "embedding_count": self.embedding_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "fallback_count": self.fallback_count,
            "fallback_rate": fallback_rate,
            "total_tokens_processed": self.total_tokens_processed,
            "local_model_available": self.local_model is not None,
            "cache_enabled": self.enable_cache,
            "fallback_enabled": self.enable_local_fallback,
            "circuit_breaker": breaker_status
        }


# Global resilient ML service
resilient_ml = ResilientMLService()