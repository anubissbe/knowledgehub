"""
AI-Powered Recommendation Engine with Cross-Domain Knowledge Synthesis
Author: Pol Verbruggen - Adaptive Quantization & Model Pruning Expert

This module implements intelligent recommendation systems that leverage
cross-domain knowledge synthesis and quantized models for efficient inference.
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
from collections import defaultdict, deque
import heapq

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """Types of recommendations the system can generate"""
    CONTENT = "content"              # Content recommendations
    ACTION = "action"               # Next action recommendations  
    WORKFLOW = "workflow"           # Process/workflow recommendations
    RESOURCE = "resource"           # Resource allocation recommendations
    OPTIMIZATION = "optimization"   # Performance optimization suggestions
    LEARNING = "learning"          # Learning and improvement recommendations

class RecommendationStrategy(Enum):
    """Different strategies for generating recommendations"""
    COLLABORATIVE = "collaborative"         # User behavior similarity
    CONTENT_BASED = "content_based"        # Feature similarity
    HYBRID = "hybrid"                      # Combined approach
    KNOWLEDGE_GRAPH = "knowledge_graph"    # Graph-based recommendations
    CROSS_DOMAIN = "cross_domain"          # Cross-domain knowledge transfer
    TEMPORAL = "temporal"                  # Time-based patterns

@dataclass
class RecommendationRequest:
    """Structure for recommendation requests"""
    user_id: str
    request_id: str
    context: Dict[str, Any]
    recommendation_type: RecommendationType
    strategy: RecommendationStrategy
    max_recommendations: int = 10
    diversity_threshold: float = 0.7
    explanation_level: str = "basic"  # "none", "basic", "detailed"
    timestamp: float = None

@dataclass
class Recommendation:
    """Individual recommendation structure"""
    item_id: str
    title: str
    description: str
    confidence: float
    relevance_score: float
    diversity_score: float
    explanation: List[str]
    metadata: Dict[str, Any]
    cross_domain_sources: List[str]

@dataclass
class RecommendationResponse:
    """Response structure for recommendations"""
    request_id: str
    user_id: str
    recommendations: List[Recommendation]
    total_candidates: int
    processing_time_ms: float
    strategy_used: str
    explanation: List[str]
    metadata: Dict[str, Any]

class CrossDomainKnowledgeBridge:
    """
    Bridge for transferring knowledge across different domains
    """
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        
        # Domain embeddings for different knowledge areas
        self.domain_embeddings = {
            'code': torch.randn(embedding_dim),
            'documentation': torch.randn(embedding_dim),
            'performance': torch.randn(embedding_dim),
            'security': torch.randn(embedding_dim),
            'workflow': torch.randn(embedding_dim),
            'user_behavior': torch.randn(embedding_dim),
            'system_metrics': torch.randn(embedding_dim)
        }
        
        # Cross-domain mapping network (quantized for speed)
        self.mapping_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.Tanh()
        )
        
        # Knowledge transfer history
        self.transfer_history = defaultdict(list)
        
    def compute_domain_similarity(self, source_domain: str, target_domain: str) -> float:
        """Compute similarity between different domains"""
        if source_domain not in self.domain_embeddings or target_domain not in self.domain_embeddings:
            return 0.0
        
        source_emb = self.domain_embeddings[source_domain]
        target_emb = self.domain_embeddings[target_domain]
        
        similarity = F.cosine_similarity(source_emb, target_emb, dim=0)
        return similarity.item()
    
    def transfer_knowledge(self, 
                          source_knowledge: Dict[str, Any], 
                          source_domain: str,
                          target_domain: str) -> Dict[str, Any]:
        """Transfer knowledge from source to target domain"""
        
        similarity = self.compute_domain_similarity(source_domain, target_domain)
        
        # Apply domain-specific transformations
        transferred_knowledge = {}
        
        if 'patterns' in source_knowledge:
            # Transfer patterns with domain adaptation
            transferred_knowledge['adapted_patterns'] = self._adapt_patterns(
                source_knowledge['patterns'], source_domain, target_domain, similarity
            )
        
        if 'metrics' in source_knowledge:
            # Transfer metrics with scaling
            transferred_knowledge['scaled_metrics'] = self._scale_metrics(
                source_knowledge['metrics'], similarity
            )
        
        if 'best_practices' in source_knowledge:
            # Transfer best practices with domain mapping
            transferred_knowledge['mapped_practices'] = self._map_best_practices(
                source_knowledge['best_practices'], source_domain, target_domain
            )
        
        # Record transfer for learning
        self.transfer_history[f"{source_domain}->{target_domain}"].append({
            'timestamp': time.time(),
            'similarity': similarity,
            'knowledge_types': list(source_knowledge.keys()),
            'success_score': similarity  # Simplified success metric
        })
        
        return transferred_knowledge
    
    def _adapt_patterns(self, patterns: List[Dict], source_domain: str, target_domain: str, similarity: float) -> List[Dict]:
        """Adapt patterns from source to target domain"""
        adapted_patterns = []
        
        for pattern in patterns:
            adapted_pattern = pattern.copy()
            
            # Scale confidence based on domain similarity
            if 'confidence' in adapted_pattern:
                adapted_pattern['confidence'] *= similarity
            
            # Add domain transfer metadata
            adapted_pattern['source_domain'] = source_domain
            adapted_pattern['target_domain'] = target_domain
            adapted_pattern['transfer_confidence'] = similarity
            
            adapted_patterns.append(adapted_pattern)
        
        return adapted_patterns
    
    def _scale_metrics(self, metrics: Dict[str, float], similarity: float) -> Dict[str, float]:
        """Scale metrics based on domain similarity"""
        return {
            key: value * similarity for key, value in metrics.items()
        }
    
    def _map_best_practices(self, practices: List[str], source_domain: str, target_domain: str) -> List[str]:
        """Map best practices between domains"""
        mapped_practices = []
        
        # Domain-specific mappings (simplified)
        domain_mappings = {
            ('code', 'documentation'): lambda x: x.replace('function', 'section').replace('variable', 'parameter'),
            ('performance', 'security'): lambda x: x.replace('speed', 'safety').replace('optimize', 'secure'),
            ('workflow', 'user_behavior'): lambda x: x.replace('process', 'interaction').replace('step', 'action')
        }
        
        mapping_key = (source_domain, target_domain)
        if mapping_key in domain_mappings:
            mapper = domain_mappings[mapping_key]
            mapped_practices = [mapper(practice) for practice in practices]
        else:
            # Default: add domain context
            mapped_practices = [f"In {target_domain} context: {practice}" for practice in practices]
        
        return mapped_practices

class QuantizedRecommendationModel(nn.Module):
    """
    Quantized neural network for fast recommendation generation
    """
    
    def __init__(self, 
                 user_features: int = 128,
                 item_features: int = 256,
                 context_features: int = 64,
                 hidden_dim: int = 512,
                 output_dim: int = 1):
        super().__init__()
        
        total_input = user_features + item_features + context_features
        
        self.network = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        # Quantization parameters for different scenarios
        self.quantization_bits = {
            'realtime': 4,    # 4-bit for real-time recommendations
            'batch': 8,       # 8-bit for batch processing
            'offline': 16     # 16-bit for offline training
        }
    
    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass through the recommendation network"""
        combined = torch.cat([user_emb, item_emb, context_emb], dim=-1)
        return self.network(combined)
    
    def quantized_forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor, context_emb: torch.Tensor, mode: str = 'realtime') -> torch.Tensor:
        """Quantized forward pass for fast inference"""
        bits = self.quantization_bits.get(mode, 8)
        
        # Simple quantization for demonstration
        def quantize(x):
            scale = 2**bits - 1
            return torch.round(x * scale) / scale
        
        # Quantize inputs
        user_emb = quantize(user_emb)
        item_emb = quantize(item_emb)
        context_emb = quantize(context_emb)
        
        return self.forward(user_emb, item_emb, context_emb)

class IntelligentRecommendationEngine:
    """
    Main recommendation engine with cross-domain knowledge synthesis
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.model_config = model_config
        
        # Initialize quantized recommendation model
        self.recommendation_model = QuantizedRecommendationModel(
            user_features=model_config.get('user_features', 128),
            item_features=model_config.get('item_features', 256),
            context_features=model_config.get('context_features', 64),
            hidden_dim=model_config.get('hidden_dim', 512)
        ).to(device)
        
        # Cross-domain knowledge bridge
        self.knowledge_bridge = CrossDomainKnowledgeBridge(
            embedding_dim=model_config.get('embedding_dim', 256)
        )
        
        # User and item embeddings (normally loaded from database)
        self.user_embeddings = {}
        self.item_embeddings = {}
        
        # Recommendation cache for performance
        self.recommendation_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance metrics
        self.metrics = {
            'recommendations_served': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'cross_domain_transfers': 0
        }
        
        # Diversity optimization
        self.diversity_tracker = defaultdict(deque)
        
        logger.info(f"Intelligent recommendation engine initialized on {device}")
    
    async def generate_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Generate personalized recommendations using cross-domain knowledge
        """
        start_time = time.perf_counter()
        
        if request.timestamp is None:
            request.timestamp = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.recommendation_cache:
            cache_entry = self.recommendation_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                self.metrics['cache_hits'] += 1
                logger.debug(f"Cache hit for request {request.request_id}")
                return cache_entry['response']
        
        try:
            # Get user embedding
            user_emb = await self._get_user_embedding(request.user_id, request.context)
            
            # Generate candidate recommendations
            candidates = await self._generate_candidates(request, user_emb)
            
            # Apply cross-domain knowledge enhancement
            enhanced_candidates = await self._enhance_with_cross_domain_knowledge(
                candidates, request
            )
            
            # Score and rank recommendations
            scored_recommendations = await self._score_and_rank_recommendations(
                enhanced_candidates, user_emb, request
            )
            
            # Apply diversity and filtering
            final_recommendations = self._apply_diversity_filtering(
                scored_recommendations, request
            )
            
            # Generate explanations
            explanations = self._generate_explanations(final_recommendations, request)
            
            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Create response
            response = RecommendationResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                recommendations=final_recommendations[:request.max_recommendations],
                total_candidates=len(candidates),
                processing_time_ms=processing_time,
                strategy_used=request.strategy.value,
                explanation=explanations,
                metadata={
                    'cross_domain_enhanced': len(enhanced_candidates) > len(candidates),
                    'cache_miss': True,
                    'diversity_applied': request.diversity_threshold > 0,
                    'model_device': self.device
                }
            )
            
            # Update cache
            self.recommendation_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            
            # Update metrics
            self._update_metrics(processing_time)
            
            logger.debug(f"Generated {len(final_recommendations)} recommendations in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error generating recommendations for {request.request_id}: {e}")
            
            # Return empty response on error
            return RecommendationResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                recommendations=[],
                total_candidates=0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                strategy_used="error",
                explanation=[f"Error generating recommendations: {str(e)}"],
                metadata={'error': str(e)}
            )
    
    async def _get_user_embedding(self, user_id: str, context: Dict[str, Any]) -> torch.Tensor:
        """Get or generate user embedding"""
        
        if user_id in self.user_embeddings:
            base_embedding = self.user_embeddings[user_id]
        else:
            # Generate new user embedding (simplified)
            base_embedding = torch.randn(self.model_config.get('user_features', 128), device=self.device)
            self.user_embeddings[user_id] = base_embedding
        
        # Add contextual information
        context_features = self._encode_context(context)
        
        return base_embedding
    
    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """Encode context information into tensor"""
        # Simplified context encoding
        context_dim = self.model_config.get('context_features', 64)
        
        # Extract numerical features from context
        features = []
        
        # Time-based features
        current_time = time.time()
        hour = int((current_time % 86400) / 3600)  # Hour of day
        features.extend([
            hour / 24.0,  # Normalized hour
            (current_time % 604800) / 604800.0,  # Day of week (normalized)
        ])
        
        # Context-specific features
        if 'urgency' in context:
            urgency_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
            features.append(urgency_map.get(context['urgency'], 0.5))
        else:
            features.append(0.5)
        
        # Session features
        if 'session_length' in context:
            features.append(min(context['session_length'] / 3600.0, 1.0))  # Normalized to hours
        else:
            features.append(0.0)
        
        # Pad or truncate to match expected dimension
        while len(features) < context_dim:
            features.append(0.0)
        
        features = features[:context_dim]
        
        return torch.tensor(features, device=self.device, dtype=torch.float32)
    
    async def _generate_candidates(self, request: RecommendationRequest, user_emb: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate candidate recommendations based on strategy"""
        
        candidates = []
        
        if request.strategy == RecommendationStrategy.CONTENT_BASED:
            candidates = await self._content_based_candidates(request, user_emb)
            
        elif request.strategy == RecommendationStrategy.COLLABORATIVE:
            candidates = await self._collaborative_filtering_candidates(request, user_emb)
            
        elif request.strategy == RecommendationStrategy.HYBRID:
            content_candidates = await self._content_based_candidates(request, user_emb)
            collab_candidates = await self._collaborative_filtering_candidates(request, user_emb)
            candidates = content_candidates + collab_candidates
            
        elif request.strategy == RecommendationStrategy.KNOWLEDGE_GRAPH:
            candidates = await self._knowledge_graph_candidates(request, user_emb)
            
        elif request.strategy == RecommendationStrategy.CROSS_DOMAIN:
            candidates = await self._cross_domain_candidates(request, user_emb)
            
        elif request.strategy == RecommendationStrategy.TEMPORAL:
            candidates = await self._temporal_pattern_candidates(request, user_emb)
        
        return candidates
    
    async def _content_based_candidates(self, request: RecommendationRequest, user_emb: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate content-based recommendations"""
        # Simplified content-based recommendation
        candidates = []
        
        # Mock item database
        mock_items = [
            {'id': f'item_{i}', 'title': f'Item {i}', 'features': torch.randn(256), 'category': f'cat_{i%5}'}
            for i in range(50)
        ]
        
        context_emb = self._encode_context(request.context)
        
        for item in mock_items:
            # Compute similarity score using quantized model
            item_emb = item['features'].to(self.device)
            score = self.recommendation_model.quantized_forward(
                user_emb.unsqueeze(0), 
                item_emb.unsqueeze(0), 
                context_emb.unsqueeze(0),
                mode='realtime'
            ).item()
            
            candidates.append({
                'item_id': item['id'],
                'title': item['title'],
                'score': score,
                'category': item['category'],
                'source': 'content_based',
                'features': item['features']
            })
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:20]
    
    async def _collaborative_filtering_candidates(self, request: RecommendationRequest, user_emb: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate collaborative filtering recommendations"""
        # Simplified collaborative filtering
        candidates = []
        
        # Mock similar users and their preferences
        similar_users = [f'user_{i}' for i in range(10)]
        
        for i, similar_user in enumerate(similar_users):
            # Mock items liked by similar users
            liked_items = [
                {'id': f'collab_item_{i}_{j}', 'title': f'Collaborative Item {i}_{j}', 'score': 0.8 - j*0.1}
                for j in range(3)
            ]
            
            for item in liked_items:
                candidates.append({
                    'item_id': item['id'],
                    'title': item['title'],
                    'score': item['score'],
                    'source': 'collaborative',
                    'similar_user': similar_user
                })
        
        return candidates
    
    async def _knowledge_graph_candidates(self, request: RecommendationRequest, user_emb: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate recommendations using knowledge graph traversal"""
        # Simplified knowledge graph recommendations
        candidates = []
        
        # Mock knowledge graph relationships
        user_interests = request.context.get('interests', ['programming', 'ai', 'optimization'])
        
        for interest in user_interests:
            # Find related items in knowledge graph
            related_items = [
                {'id': f'kg_{interest}_{i}', 'title': f'{interest.title()} Resource {i}', 'relation': 'related_to'}
                for i in range(5)
            ]
            
            for item in related_items:
                candidates.append({
                    'item_id': item['id'],
                    'title': item['title'],
                    'score': 0.7 + np.random.random() * 0.3,
                    'source': 'knowledge_graph',
                    'interest': interest,
                    'relation': item['relation']
                })
        
        return candidates
    
    async def _cross_domain_candidates(self, request: RecommendationRequest, user_emb: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate recommendations using cross-domain knowledge transfer"""
        candidates = []
        
        # Get knowledge from multiple domains
        domains = ['code', 'documentation', 'performance', 'security']
        
        for source_domain in domains:
            for target_domain in domains:
                if source_domain != target_domain:
                    # Simulate knowledge transfer
                    similarity = self.knowledge_bridge.compute_domain_similarity(source_domain, target_domain)
                    
                    if similarity > 0.3:  # Threshold for useful transfer
                        # Create cross-domain recommendations
                        transferred_item = {
                            'item_id': f'cross_{source_domain}_{target_domain}',
                            'title': f'{target_domain.title()} insights from {source_domain}',
                            'score': similarity,
                            'source': 'cross_domain',
                            'source_domain': source_domain,
                            'target_domain': target_domain,
                            'transfer_confidence': similarity
                        }
                        candidates.append(transferred_item)
                        
                        # Track cross-domain transfer
                        self.metrics['cross_domain_transfers'] += 1
        
        return candidates
    
    async def _temporal_pattern_candidates(self, request: RecommendationRequest, user_emb: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate recommendations based on temporal patterns"""
        candidates = []
        
        current_hour = int((time.time() % 86400) / 3600)
        
        # Time-based recommendations (simplified)
        time_based_items = {
            'morning': ['productivity_tips', 'planning_tools', 'energy_boosters'],
            'afternoon': ['collaboration_tools', 'problem_solving', 'optimization_guides'],
            'evening': ['learning_resources', 'documentation', 'review_materials'],
            'night': ['automation_tools', 'maintenance_guides', 'system_monitoring']
        }
        
        if 6 <= current_hour < 12:
            time_category = 'morning'
        elif 12 <= current_hour < 18:
            time_category = 'afternoon'  
        elif 18 <= current_hour < 22:
            time_category = 'evening'
        else:
            time_category = 'night'
        
        for i, item_type in enumerate(time_based_items[time_category]):
            candidates.append({
                'item_id': f'temporal_{item_type}_{i}',
                'title': f'{item_type.replace("_", " ").title()}',
                'score': 0.6 + (len(time_based_items[time_category]) - i) * 0.1,
                'source': 'temporal',
                'time_category': time_category,
                'item_type': item_type
            })
        
        return candidates
    
    async def _enhance_with_cross_domain_knowledge(self, candidates: List[Dict[str, Any]], request: RecommendationRequest) -> List[Dict[str, Any]]:
        """Enhance candidates with cross-domain knowledge"""
        enhanced_candidates = candidates.copy()
        
        # Group candidates by source domain
        domain_groups = defaultdict(list)
        for candidate in candidates:
            source = candidate.get('source', 'unknown')
            domain_groups[source].append(candidate)
        
        # Apply cross-domain enhancement
        for source_domain, domain_candidates in domain_groups.items():
            if source_domain in ['content_based', 'collaborative']:
                # Enhance with knowledge graph insights
                for candidate in domain_candidates:
                    # Add related knowledge from other domains
                    related_knowledge = {
                        'performance_insights': f"Optimized for {candidate.get('category', 'general')} use cases",
                        'security_considerations': f"Secure implementation patterns available",
                        'documentation_links': f"Comprehensive guides for {candidate['title']}"
                    }
                    candidate['cross_domain_enhancements'] = related_knowledge
        
        return enhanced_candidates
    
    async def _score_and_rank_recommendations(self, candidates: List[Dict[str, Any]], user_emb: torch.Tensor, request: RecommendationRequest) -> List[Recommendation]:
        """Score and rank all candidates into final recommendations"""
        
        recommendations = []
        context_emb = self._encode_context(request.context)
        
        for candidate in candidates:
            # Get item embedding (mock for candidates without features)
            if 'features' in candidate:
                item_emb = candidate['features'].to(self.device)
            else:
                item_emb = torch.randn(self.model_config.get('item_features', 256), device=self.device)
            
            # Compute final score using quantized model
            base_score = candidate.get('score', 0.5)
            
            # Use neural model for refinement
            if base_score > 0.3:  # Only for promising candidates
                neural_score = self.recommendation_model.quantized_forward(
                    user_emb.unsqueeze(0),
                    item_emb.unsqueeze(0),
                    context_emb.unsqueeze(0),
                    mode='realtime'
                ).item()
                
                # Combine scores
                final_score = 0.7 * base_score + 0.3 * neural_score
            else:
                final_score = base_score
            
            # Calculate diversity score (simplified)
            diversity_score = self._calculate_diversity_score(candidate, recommendations)
            
            # Generate explanation
            explanation = self._generate_item_explanation(candidate, final_score, request)
            
            # Get cross-domain sources
            cross_domain_sources = []
            if 'source_domain' in candidate:
                cross_domain_sources.append(f"{candidate['source_domain']} -> {candidate.get('target_domain', 'unknown')}")
            if 'cross_domain_enhancements' in candidate:
                cross_domain_sources.extend(list(candidate['cross_domain_enhancements'].keys()))
            
            recommendation = Recommendation(
                item_id=candidate['item_id'],
                title=candidate['title'],
                description=candidate.get('description', f"AI-recommended {candidate['title']}"),
                confidence=final_score,
                relevance_score=final_score,
                diversity_score=diversity_score,
                explanation=explanation,
                metadata={
                    'source': candidate.get('source', 'unknown'),
                    'original_score': candidate.get('score', 0.0),
                    'neural_enhancement': final_score != base_score,
                    **{k: v for k, v in candidate.items() if k not in ['features']}
                },
                cross_domain_sources=cross_domain_sources
            )
            
            recommendations.append(recommendation)
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        return recommendations
    
    def _calculate_diversity_score(self, candidate: Dict[str, Any], existing_recommendations: List[Recommendation]) -> float:
        """Calculate diversity score for a candidate"""
        if not existing_recommendations:
            return 1.0
        
        # Simple diversity based on source and category
        candidate_source = candidate.get('source', 'unknown')
        candidate_category = candidate.get('category', 'general')
        
        source_diversity = 1.0
        category_diversity = 1.0
        
        for rec in existing_recommendations:
            if rec.metadata.get('source') == candidate_source:
                source_diversity *= 0.8
            if rec.metadata.get('category') == candidate_category:
                category_diversity *= 0.9
        
        return (source_diversity + category_diversity) / 2.0
    
    def _apply_diversity_filtering(self, recommendations: List[Recommendation], request: RecommendationRequest) -> List[Recommendation]:
        """Apply diversity filtering to avoid redundant recommendations"""
        
        if request.diversity_threshold <= 0:
            return recommendations
        
        filtered_recommendations = []
        
        for rec in recommendations:
            # Check if recommendation is diverse enough
            is_diverse = True
            
            for existing_rec in filtered_recommendations:
                if rec.diversity_score < request.diversity_threshold:
                    # Check similarity with existing recommendations
                    similarity = self._compute_recommendation_similarity(rec, existing_rec)
                    if similarity > (1.0 - request.diversity_threshold):
                        is_diverse = False
                        break
            
            if is_diverse:
                filtered_recommendations.append(rec)
        
        return filtered_recommendations
    
    def _compute_recommendation_similarity(self, rec1: Recommendation, rec2: Recommendation) -> float:
        """Compute similarity between two recommendations"""
        # Simple similarity based on metadata
        similarity = 0.0
        
        if rec1.metadata.get('source') == rec2.metadata.get('source'):
            similarity += 0.3
        
        if rec1.metadata.get('category') == rec2.metadata.get('category'):
            similarity += 0.3
        
        # Title similarity (simplified)
        title1_words = set(rec1.title.lower().split())
        title2_words = set(rec2.title.lower().split())
        
        if title1_words and title2_words:
            word_similarity = len(title1_words & title2_words) / len(title1_words | title2_words)
            similarity += 0.4 * word_similarity
        
        return similarity
    
    def _generate_item_explanation(self, candidate: Dict[str, Any], score: float, request: RecommendationRequest) -> List[str]:
        """Generate explanation for individual recommendation"""
        
        if request.explanation_level == "none":
            return []
        
        explanations = []
        
        # Basic explanation
        source = candidate.get('source', 'unknown')
        explanations.append(f"Recommended based on {source.replace('_', ' ')} analysis")
        
        if score > 0.8:
            explanations.append(f"High relevance score ({score:.2f}) indicates strong match")
        elif score > 0.6:
            explanations.append(f"Good relevance score ({score:.2f}) suggests suitable match")
        
        if request.explanation_level == "detailed":
            # Add detailed explanations
            if 'category' in candidate:
                explanations.append(f"Matches your interest in {candidate['category']}")
            
            if 'cross_domain_enhancements' in candidate:
                explanations.append("Enhanced with cross-domain knowledge insights")
            
            if candidate.get('source') == 'cross_domain':
                source_domain = candidate.get('source_domain', 'unknown')
                target_domain = candidate.get('target_domain', 'unknown')
                explanations.append(f"Knowledge transferred from {source_domain} to {target_domain}")
        
        return explanations
    
    def _generate_explanations(self, recommendations: List[Recommendation], request: RecommendationRequest) -> List[str]:
        """Generate overall explanation for the recommendation set"""
        
        explanations = []
        
        if not recommendations:
            return ["No suitable recommendations found for your criteria"]
        
        # Strategy explanation
        strategy_descriptions = {
            RecommendationStrategy.CONTENT_BASED: "Based on content similarity to your preferences",
            RecommendationStrategy.COLLABORATIVE: "Based on users with similar behavior patterns", 
            RecommendationStrategy.HYBRID: "Combining content-based and collaborative approaches",
            RecommendationStrategy.KNOWLEDGE_GRAPH: "Using knowledge graph relationships",
            RecommendationStrategy.CROSS_DOMAIN: "Leveraging insights across different domains",
            RecommendationStrategy.TEMPORAL: "Based on temporal usage patterns"
        }
        
        explanations.append(strategy_descriptions.get(
            request.strategy,
            f"Using {request.strategy.value} recommendation strategy"
        ))
        
        # Cross-domain explanation
        cross_domain_count = sum(1 for rec in recommendations if rec.cross_domain_sources)
        if cross_domain_count > 0:
            explanations.append(f"{cross_domain_count} recommendations enhanced with cross-domain knowledge")
        
        # Diversity explanation
        if request.diversity_threshold > 0:
            explanations.append(f"Applied diversity filtering (threshold: {request.diversity_threshold:.1f})")
        
        return explanations
    
    def _generate_cache_key(self, request: RecommendationRequest) -> str:
        """Generate cache key for request"""
        key_parts = [
            request.user_id,
            request.recommendation_type.value,
            request.strategy.value,
            str(request.max_recommendations),
            str(hash(frozenset(request.context.items())))
        ]
        return "_".join(key_parts)
    
    def _update_metrics(self, processing_time_ms: float):
        """Update performance metrics"""
        self.metrics['recommendations_served'] += 1
        
        # Update rolling average processing time
        alpha = 0.1
        if self.metrics['avg_processing_time'] == 0:
            self.metrics['avg_processing_time'] = processing_time_ms
        else:
            self.metrics['avg_processing_time'] = (1 - alpha) * self.metrics['avg_processing_time'] + alpha * processing_time_ms
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        cache_hit_rate = self.metrics['cache_hits'] / max(self.metrics['recommendations_served'], 1)
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.recommendation_cache),
            'user_embeddings_cached': len(self.user_embeddings),
            'knowledge_bridge_domains': len(self.knowledge_bridge.domain_embeddings),
            'cross_domain_transfer_rate': self.metrics['cross_domain_transfers'] / max(self.metrics['recommendations_served'], 1)
        }
    
    async def batch_generate_recommendations(self, requests: List[RecommendationRequest]) -> List[RecommendationResponse]:
        """Generate recommendations for multiple requests efficiently"""
        
        # Group requests by strategy for optimized processing
        strategy_groups = defaultdict(list)
        for req in requests:
            strategy_groups[req.strategy].append(req)
        
        responses = []
        
        # Process each strategy group
        for strategy, group_requests in strategy_groups.items():
            if len(group_requests) == 1:
                # Single request - process normally
                response = await self.generate_recommendations(group_requests[0])
                responses.append(response)
            else:
                # Multiple requests - batch process
                batch_responses = await self._batch_process_strategy_group(group_requests)
                responses.extend(batch_responses)
        
        return responses
    
    async def _batch_process_strategy_group(self, requests: List[RecommendationRequest]) -> List[RecommendationResponse]:
        """Process a group of requests with the same strategy"""
        
        # For now, process individually but could be optimized for true batch processing
        responses = []
        
        for request in requests:
            response = await self.generate_recommendations(request)
            responses.append(response)
        
        return responses


# Example usage and testing
async def test_recommendation_engine():
    """Test the recommendation engine"""
    
    config = {
        'user_features': 128,
        'item_features': 256,
        'context_features': 64,
        'hidden_dim': 512,
        'embedding_dim': 256
    }
    
    engine = IntelligentRecommendationEngine(config)
    
    # Test different recommendation strategies
    strategies = [
        RecommendationStrategy.CONTENT_BASED,
        RecommendationStrategy.COLLABORATIVE,
        RecommendationStrategy.CROSS_DOMAIN,
        RecommendationStrategy.TEMPORAL
    ]
    
    for strategy in strategies:
        request = RecommendationRequest(
            user_id="test_user_001",
            request_id=f"test_{strategy.value}",
            context={
                'interests': ['ai', 'programming', 'optimization'],
                'urgency': 'medium',
                'session_length': 1800,  # 30 minutes
                'current_project': 'recommendation_system'
            },
            recommendation_type=RecommendationType.CONTENT,
            strategy=strategy,
            max_recommendations=5,
            diversity_threshold=0.7,
            explanation_level="detailed"
        )
        
        response = await engine.generate_recommendations(request)
        
        print(f"\n=== {strategy.value.upper()} STRATEGY ===")
        print(f"Processing time: {response.processing_time_ms:.2f}ms")
        print(f"Total candidates: {response.total_candidates}")
        print(f"Final recommendations: {len(response.recommendations)}")
        
        for i, rec in enumerate(response.recommendations):
            print(f"{i+1}. {rec.title}")
            print(f"   Confidence: {rec.confidence:.3f}")
            print(f"   Cross-domain: {rec.cross_domain_sources}")
            print(f"   Explanation: {rec.explanation[0] if rec.explanation else 'N/A'}")
    
    # Test batch processing
    batch_requests = [
        RecommendationRequest(
            user_id=f"batch_user_{i}",
            request_id=f"batch_test_{i}",
            context={'interests': ['programming'], 'urgency': 'low'},
            recommendation_type=RecommendationType.ACTION,
            strategy=RecommendationStrategy.HYBRID,
            max_recommendations=3
        ) for i in range(5)
    ]
    
    batch_responses = await engine.batch_generate_recommendations(batch_requests)
    print(f"\n=== BATCH PROCESSING ===")
    print(f"Processed {len(batch_responses)} requests")
    
    avg_batch_time = sum(r.processing_time_ms for r in batch_responses) / len(batch_responses)
    print(f"Average processing time: {avg_batch_time:.2f}ms")
    
    # Show performance metrics
    metrics = engine.get_performance_metrics()
    print(f"\n=== PERFORMANCE METRICS ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_recommendation_engine())
