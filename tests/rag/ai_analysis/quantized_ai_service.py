"""
Quantized AI Analysis Service for KnowledgeHub
Created by Annelies Claes - Expert in Neural Network Quantization & API Design

This service provides efficient AI analysis using quantized neural networks
and integrates seamlessly with the existing KnowledgeHub MCP architecture.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import redis
from contextlib import asynccontextmanager

from .lottery_ticket_pattern_engine import LotteryTicketPatternEngine, PatternMatch

logger = logging.getLogger(__name__)

# Pydantic Models for API
from collections import Counter
from datetime import datetime, timedelta
class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis."""
    content: str = Field(..., min_length=1, max_length=50000, description="Content to analyze")
    content_type: str = Field("text", description="Type of content (text, code, document)")
    analysis_depth: str = Field("standard", description="Analysis depth: fast, standard, deep")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class PatternMatchResponse(BaseModel):
    """Response model for pattern matches."""
    pattern_id: str
    pattern_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    location: Tuple[int, int]
    context: str
    severity: str
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AnalysisResponse(BaseModel):
    """Response model for content analysis."""
    analysis_id: str
    patterns: List[PatternMatchResponse]
    summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float
    processing_time: float

class SemanticSimilarityRequest(BaseModel):
    """Request model for semantic similarity analysis."""
    query_content: str = Field(..., min_length=1, description="Query content")
    target_contents: List[str] = Field(..., min_items=1, max_items=100, description="Target contents to compare")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    use_quantized_model: bool = Field(True, description="Use quantized model for efficiency")

class SimilarityMatch(BaseModel):
    """Model for similarity match results."""
    content_id: str
    content_preview: str
    similarity_score: float
    match_type: str  # 'semantic', 'structural', 'lexical'
    confidence: float

class SimilarityResponse(BaseModel):
    """Response model for similarity analysis."""
    query_id: str
    matches: List[SimilarityMatch]
    processing_time: float
    model_info: Dict[str, Any]

class UserBehaviorAnalysisRequest(BaseModel):
    """Request model for user behavior analysis."""
    user_id: str
    session_data: List[Dict[str, Any]]
    time_window: int = Field(3600, description="Time window in seconds")
    analysis_type: str = Field("pattern_detection", description="Type of analysis")

class BehaviorPattern(BaseModel):
    """Model for detected behavior patterns."""
    pattern_type: str
    description: str
    confidence: float
    frequency: int
    time_range: Tuple[datetime, datetime]
    metadata: Dict[str, Any]

class BehaviorAnalysisResponse(BaseModel):
    """Response model for behavior analysis."""
    user_id: str
    patterns: List[BehaviorPattern]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)

class QuantizedAIService:
    """
    Advanced AI Analysis Service with Neural Network Quantization
    
    Features:
    - Lottery Ticket Hypothesis pattern recognition
    - Quantized neural networks for efficiency
    - Real-time content analysis
    - Semantic similarity beyond RAG
    - User behavior pattern analysis
    - Integration with KnowledgeHub MCP servers
    """
    
    def __init__(
        self,
        knowledgehub_api_base: str = "http://192.168.1.25:3000",
        ai_service_base: str = "http://192.168.1.25:8002",
        redis_url: str = "redis://192.168.1.25:6381",
        enable_caching: bool = True
    ):
        self.knowledgehub_api_base = knowledgehub_api_base
        self.ai_service_base = ai_service_base
        self.redis_url = redis_url
        self.enable_caching = enable_caching
        
        # Initialize core components
        self.pattern_engine: Optional[LotteryTicketPatternEngine] = None
        self.redis_client: Optional[redis.Redis] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Performance metrics
        self.performance_stats = {
            'total_analyses': 0,
            'avg_processing_time': 0.0,
            'cache_hit_rate': 0.0,
            'pattern_detection_accuracy': 0.0
        }
        
        # Model configurations
        self.model_configs = {
            'fast': {'sparsity_target': 0.1, 'quantization_bits': 4},
            'standard': {'sparsity_target': 0.2, 'quantization_bits': 8},
            'deep': {'sparsity_target': 0.3, 'quantization_bits': 16}
        }
        
        logger.info("QuantizedAIService initialized")

    async def initialize(self):
        """Initialize all service components."""
        try:
            # Initialize pattern recognition engine
            self.pattern_engine = LotteryTicketPatternEngine(
                sparsity_target=0.2,  # 20% sparsity for optimal performance
                quantization_bits=8   # 8-bit quantization for efficiency
            )
            await self.pattern_engine.initialize_embedding_model()
            
            # Initialize Redis for caching
            if self.enable_caching:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()  # Test connection
                logger.info("Redis connection established")
            
            # Initialize HTTP client for MCP integration
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            logger.info("QuantizedAIService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize QuantizedAIService: {e}")
            # Graceful fallback - continue without optional components
            if "redis" in str(e).lower():
                logger.warning("Continuing without Redis caching")
                self.enable_caching = False

    async def analyze_content(
        self,
        request: ContentAnalysisRequest
    ) -> AnalysisResponse:
        """
        Perform comprehensive content analysis using quantized neural networks.
        """
        start_time = time.time()
        analysis_id = f"analysis_{int(start_time)}_{hash(request.content) % 10000}"
        
        try:
            # Check cache first
            cached_result = await self._get_cached_analysis(request.content)
            if cached_result and self.enable_caching:
                logger.debug(f"Cache hit for analysis {analysis_id}")
                return cached_result
            
            # Configure model based on analysis depth
            config = self.model_configs.get(request.analysis_depth, self.model_configs['standard'])
            
            # Perform pattern analysis using Lottery Ticket Hypothesis
            pattern_matches = await self.pattern_engine.analyze_content(
                content=request.content,
                content_type=request.content_type,
                context=request.context
            )
            
            # Convert pattern matches to response models
            response_patterns = [
                PatternMatchResponse(
                    pattern_id=match.pattern_id,
                    pattern_name=match.pattern_name,
                    confidence=match.confidence,
                    location=match.location,
                    context=match.context,
                    severity=match.severity,
                    metadata=match.metadata
                )
                for match in pattern_matches
            ]
            
            # Generate analysis summary
            summary = await self._generate_analysis_summary(pattern_matches)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(pattern_matches, request.context)
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            confidence_score = np.mean([p.confidence for p in pattern_matches]) if pattern_matches else 0.0
            
            performance_metrics = {
                'processing_time': processing_time,
                'patterns_detected': len(pattern_matches),
                'sparsity_ratio': config['sparsity_target'],
                'quantization_bits': config['quantization_bits'],
                'model_efficiency': self._calculate_model_efficiency(processing_time, len(request.content))
            }
            
            # Create response
            response = AnalysisResponse(
                analysis_id=analysis_id,
                patterns=response_patterns,
                summary=summary,
                performance_metrics=performance_metrics,
                recommendations=recommendations,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
            # Cache result for future use
            if self.enable_caching:
                await self._cache_analysis_result(request.content, response)
            
            # Update performance statistics
            self._update_performance_stats(processing_time, confidence_score)
            
            # Async: Store analysis for learning (fire and forget)
            asyncio.create_task(self._store_analysis_for_learning(request, response))
            
            return response
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def semantic_similarity_analysis(
        self,
        request: SemanticSimilarityRequest
    ) -> SimilarityResponse:
        """
        Advanced semantic similarity analysis beyond basic RAG.
        
        Uses quantized neural networks for efficient similarity computation
        with multiple similarity measures (semantic, structural, lexical).
        """
        start_time = time.time()
        query_id = f"sim_{int(start_time)}_{hash(request.query_content) % 10000}"
        
        try:
            # Initialize embedding model if not already done
            if not self.pattern_engine.embedding_model:
                await self.pattern_engine.initialize_embedding_model()
            
            # Generate query embedding
            query_embedding = self.pattern_engine.embedding_model.encode(request.query_content)
            
            # Analyze each target content
            matches = []
            for i, content in enumerate(request.target_contents):
                # Generate content embedding
                content_embedding = self.pattern_engine.embedding_model.encode(content)
                
                # Calculate semantic similarity
                semantic_score = float(np.dot(query_embedding, content_embedding) / 
                                     (np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)))
                
                # Calculate structural similarity (using pattern analysis)
                structural_score = await self._calculate_structural_similarity(
                    request.query_content, content
                )
                
                # Calculate lexical similarity
                lexical_score = self._calculate_lexical_similarity(
                    request.query_content, content
                )
                
                # Combined similarity score with weights
                combined_score = (
                    0.5 * semantic_score +
                    0.3 * structural_score +
                    0.2 * lexical_score
                )
                
                if combined_score >= request.similarity_threshold:
                    match_type = self._determine_match_type(
                        semantic_score, structural_score, lexical_score
                    )
                    
                    matches.append(SimilarityMatch(
                        content_id=f"content_{i}",
                        content_preview=content[:200] + "..." if len(content) > 200 else content,
                        similarity_score=combined_score,
                        match_type=match_type,
                        confidence=min(combined_score * 1.2, 1.0)  # Confidence boost for high similarity
                    ))
            
            # Sort matches by similarity score
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            processing_time = time.time() - start_time
            
            # Model information
            model_info = {
                'quantization_enabled': request.use_quantized_model,
                'embedding_model': 'all-MiniLM-L6-v2',
                'similarity_measures': ['semantic', 'structural', 'lexical'],
                'sparsity_optimization': True
            }
            
            return SimilarityResponse(
                query_id=query_id,
                matches=matches,
                processing_time=processing_time,
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Error in similarity analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Similarity analysis failed: {str(e)}")

    async def analyze_user_behavior(
        self,
        request: UserBehaviorAnalysisRequest
    ) -> BehaviorAnalysisResponse:
        """
        Analyze user behavior patterns using sparse neural networks.
        """
        try:
            patterns = []
            anomalies = []
            
            # Analyze session data for patterns
            if request.session_data:
                # Pattern detection using time series analysis
                time_patterns = self._detect_temporal_patterns(request.session_data)
                action_patterns = self._detect_action_patterns(request.session_data)
                interaction_patterns = self._detect_interaction_patterns(request.session_data)
                
                patterns.extend(time_patterns)
                patterns.extend(action_patterns)
                patterns.extend(interaction_patterns)
                
                # Anomaly detection
                anomalies = self._detect_behavioral_anomalies(request.session_data)
            
            # Generate recommendations based on patterns
            recommendations = self._generate_behavior_recommendations(patterns, anomalies)
            
            return BehaviorAnalysisResponse(
                user_id=request.user_id,
                patterns=patterns,
                anomalies=anomalies,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in behavior analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Behavior analysis failed: {str(e)}")

    # Helper methods for analysis

    async def _get_cached_analysis(self, content: str) -> Optional[AnalysisResponse]:
        """Get cached analysis result."""
        if not self.enable_caching or not self.redis_client:
            return None
        
        try:
            cache_key = f"analysis:{hash(content)}"
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return AnalysisResponse.parse_raw(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None

    async def _cache_analysis_result(self, content: str, result: AnalysisResponse):
        """Cache analysis result."""
        if not self.enable_caching or not self.redis_client:
            return
        
        try:
            cache_key = f"analysis:{hash(content)}"
            await self.redis_client.setex(
                cache_key, 
                3600,  # 1 hour TTL
                result.json()
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    async def _generate_analysis_summary(self, patterns: List[PatternMatch]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        if not patterns:
            return {
                'total_patterns': 0,
                'severity_distribution': {},
                'confidence_stats': {'mean': 0.0, 'std': 0.0},
                'pattern_categories': {}
            }
        
        severities = [p.severity for p in patterns]
        confidences = [p.confidence for p in patterns]
        categories = [p.metadata.get('pattern_type', 'unknown') for p in patterns]
        
        return {
            'total_patterns': len(patterns),
            'severity_distribution': dict(Counter(severities)),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'pattern_categories': dict(Counter(categories)),
            'critical_issues': len([p for p in patterns if p.severity == 'critical']),
            'high_issues': len([p for p in patterns if p.severity == 'high'])
        }

    async def _generate_recommendations(
        self, 
        patterns: List[PatternMatch], 
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on detected patterns."""
        recommendations = []
        
        # Security recommendations
        security_patterns = [p for p in patterns if 'security' in p.pattern_name.lower() or p.severity == 'critical']
        if security_patterns:
            recommendations.append("Review and address critical security vulnerabilities immediately")
            recommendations.append("Implement input validation and output encoding")
            recommendations.append("Consider security code review and penetration testing")
        
        # Performance recommendations
        perf_patterns = [p for p in patterns if 'performance' in p.pattern_name.lower() or 'bottleneck' in p.pattern_name.lower()]
        if perf_patterns:
            recommendations.append("Optimize identified performance bottlenecks")
            recommendations.append("Consider code profiling and performance testing")
        
        # Code quality recommendations
        quality_patterns = [p for p in patterns if p.severity in ['medium', 'low']]
        if len(quality_patterns) > 5:
            recommendations.append("Address code quality issues to improve maintainability")
            recommendations.append("Consider refactoring and implementing better design patterns")
        
        # Add context-specific recommendations
        if context and context.get('project_type') == 'web_application':
            recommendations.append("Ensure HTTPS is properly configured")
            recommendations.append("Implement Content Security Policy (CSP) headers")
        
        return recommendations

    def _calculate_model_efficiency(self, processing_time: float, content_length: int) -> float:
        """Calculate model efficiency metric."""
        # Characters processed per second
        chars_per_second = content_length / processing_time if processing_time > 0 else 0
        
        # Normalize to a 0-1 scale (assuming 10000 chars/sec as excellent)
        efficiency = min(chars_per_second / 10000, 1.0)
        return round(efficiency, 3)

    def _update_performance_stats(self, processing_time: float, confidence_score: float):
        """Update running performance statistics."""
        self.performance_stats['total_analyses'] += 1
        
        # Update average processing time
        n = self.performance_stats['total_analyses']
        current_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (current_avg * (n-1) + processing_time) / n
        
        # Update average confidence
        current_acc = self.performance_stats['pattern_detection_accuracy']
        self.performance_stats['pattern_detection_accuracy'] = (current_acc * (n-1) + confidence_score) / n

    async def _store_analysis_for_learning(
        self, 
        request: ContentAnalysisRequest, 
        response: AnalysisResponse
    ):
        """Store analysis results for continuous learning (async)."""
        try:
            # Integration with KnowledgeHub memory system
            if self.http_client:
                learning_data = {
                    'analysis_id': response.analysis_id,
                    'content_type': request.content_type,
                    'patterns_detected': len(response.patterns),
                    'confidence_score': response.confidence_score,
                    'processing_time': response.processing_time,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Store in KnowledgeHub for pattern learning
                await self.http_client.post(
                    f"{self.knowledgehub_api_base}/api/claude-auto/pattern-learning",
                    json=learning_data,
                    timeout=5.0
                )
                
        except Exception as e:
            logger.warning(f"Failed to store analysis for learning: {e}")

    async def _calculate_structural_similarity(self, query: str, content: str) -> float:
        """Calculate structural similarity between query and content."""
        try:
            # Use pattern engine to analyze both contents
            query_patterns = await self.pattern_engine.analyze_content(query)
            content_patterns = await self.pattern_engine.analyze_content(content)
            
            # Compare pattern structures
            query_pattern_types = set(p.metadata.get('pattern_type', '') for p in query_patterns)
            content_pattern_types = set(p.metadata.get('pattern_type', '') for p in content_patterns)
            
            if not query_pattern_types and not content_pattern_types:
                return 0.5  # No patterns in either
            
            if not query_pattern_types or not content_pattern_types:
                return 0.2  # Patterns in only one
            
            # Jaccard similarity of pattern types
            intersection = len(query_pattern_types.intersection(content_pattern_types))
            union = len(query_pattern_types.union(content_pattern_types))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Structural similarity calculation failed: {e}")
            return 0.0

    def _calculate_lexical_similarity(self, query: str, content: str) -> float:
        """Calculate lexical similarity using n-grams."""
        try:
            # Simple word-based similarity
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words and not content_words:
                return 1.0
            
            if not query_words or not content_words:
                return 0.0
            
            intersection = len(query_words.intersection(content_words))
            union = len(query_words.union(content_words))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0

    def _determine_match_type(
        self, 
        semantic_score: float, 
        structural_score: float, 
        lexical_score: float
    ) -> str:
        """Determine the primary type of match."""
        scores = {
            'semantic': semantic_score,
            'structural': structural_score,
            'lexical': lexical_score
        }
        
        return max(scores, key=scores.get)

    def _detect_temporal_patterns(self, session_data: List[Dict[str, Any]]) -> List[BehaviorPattern]:
        """Detect temporal patterns in user behavior."""
        patterns = []
        
        if len(session_data) < 2:
            return patterns
        
        # Analyze time intervals between actions
        timestamps = []
        for session in session_data:
            if 'timestamp' in session:
                try:
                    timestamps.append(datetime.fromisoformat(session['timestamp']))
                except:
                    continue
        
        if len(timestamps) >= 2:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Detect regular intervals (routine behavior)
            if std_interval < avg_interval * 0.3:  # Low variance
                patterns.append(BehaviorPattern(
                    pattern_type="regular_intervals",
                    description=f"User shows regular activity intervals (avg: {avg_interval:.1f}s)",
                    confidence=0.8,
                    frequency=len(intervals),
                    time_range=(timestamps[0], timestamps[-1]),
                    metadata={'avg_interval': avg_interval, 'std_interval': std_interval}
                ))
        
        return patterns

    def _detect_action_patterns(self, session_data: List[Dict[str, Any]]) -> List[BehaviorPattern]:
        """Detect action patterns in user behavior."""
        patterns = []
        
        # Extract actions
        actions = []
        for session in session_data:
            if 'action' in session:
                actions.append(session['action'])
        
        if not actions:
            return patterns
        
        # Find repeated sequences
        action_counts = Counter(actions)
        most_common = action_counts.most_common(3)
        
        for action, count in most_common:
            if count >= 3:  # Action repeated at least 3 times
                patterns.append(BehaviorPattern(
                    pattern_type="repeated_action",
                    description=f"User frequently performs action: {action}",
                    confidence=min(0.6 + (count / len(actions)), 1.0),
                    frequency=count,
                    time_range=(datetime.utcnow() - timedelta(hours=1), datetime.utcnow()),
                    metadata={'action': action, 'total_actions': len(actions)}
                ))
        
        return patterns

    def _detect_interaction_patterns(self, session_data: List[Dict[str, Any]]) -> List[BehaviorPattern]:
        """Detect interaction patterns in user behavior."""
        patterns = []
        
        # Analyze session duration and interaction depth
        if session_data:
            total_interactions = len(session_data)
            
            # Check for session characteristics
            unique_pages = set()
            for session in session_data:
                if 'page' in session:
                    unique_pages.add(session['page'])
            
            if len(unique_pages) > 1:
                patterns.append(BehaviorPattern(
                    pattern_type="multi_page_session",
                    description=f"User explored multiple pages ({len(unique_pages)} unique pages)",
                    confidence=0.7,
                    frequency=len(unique_pages),
                    time_range=(datetime.utcnow() - timedelta(hours=1), datetime.utcnow()),
                    metadata={'unique_pages': len(unique_pages), 'total_interactions': total_interactions}
                ))
        
        return patterns

    def _detect_behavioral_anomalies(self, session_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalous behavior patterns."""
        anomalies = []
        
        if not session_data:
            return anomalies
        
        # Check for unusually high activity
        if len(session_data) > 100:  # More than 100 actions in the time window
            anomalies.append({
                'type': 'high_activity',
                'description': 'Unusually high number of interactions',
                'severity': 'medium',
                'count': len(session_data),
                'threshold': 100
            })
        
        # Check for rapid-fire actions (potential bot behavior)
        timestamps = []
        for session in session_data:
            if 'timestamp' in session:
                try:
                    timestamps.append(datetime.fromisoformat(session['timestamp']))
                except:
                    continue
        
        if len(timestamps) >= 10:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            
            rapid_actions = sum(1 for interval in intervals if interval < 0.5)  # Less than 0.5 seconds
            
            if rapid_actions > len(intervals) * 0.3:  # More than 30% are rapid
                anomalies.append({
                    'type': 'rapid_fire_actions',
                    'description': 'Detected rapid-fire actions (possible automation)',
                    'severity': 'high',
                    'rapid_action_ratio': rapid_actions / len(intervals),
                    'threshold': 0.3
                })
        
        return anomalies

    def _generate_behavior_recommendations(
        self, 
        patterns: List[BehaviorPattern], 
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on behavior analysis."""
        recommendations = []
        
        # Recommendations based on patterns
        for pattern in patterns:
            if pattern.pattern_type == "regular_intervals":
                recommendations.append("User shows consistent engagement patterns - good for retention")
            elif pattern.pattern_type == "repeated_action":
                recommendations.append(f"Consider optimizing the '{pattern.metadata.get('action')}' workflow")
            elif pattern.pattern_type == "multi_page_session":
                recommendations.append("User is actively exploring - consider personalized content recommendations")
        
        # Recommendations based on anomalies
        for anomaly in anomalies:
            if anomaly['type'] == 'high_activity':
                recommendations.append("Monitor for potential bot activity or unusual usage patterns")
            elif anomaly['type'] == 'rapid_fire_actions':
                recommendations.append("Implement rate limiting and CAPTCHA to prevent automated abuse")
        
        if not patterns and not anomalies:
            recommendations.append("Insufficient data for pattern analysis - consider longer observation period")
        
        return recommendations

    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health information."""
        health_info = {
            'service': 'QuantizedAIService',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {},
            'performance_stats': self.performance_stats.copy(),
            'model_info': {}
        }
        
        # Check pattern engine
        if self.pattern_engine:
            engine_stats = await self.pattern_engine.get_pattern_statistics()
            health_info['components']['pattern_engine'] = {
                'status': 'operational',
                'statistics': engine_stats
            }
            health_info['model_info'].update(engine_stats)
        else:
            health_info['components']['pattern_engine'] = {'status': 'not_initialized'}
            health_info['status'] = 'degraded'
        
        # Check Redis
        if self.enable_caching and self.redis_client:
            try:
                await self.redis_client.ping()
                health_info['components']['redis'] = {'status': 'operational'}
            except:
                health_info['components']['redis'] = {'status': 'error'}
                health_info['status'] = 'degraded'
        else:
            health_info['components']['redis'] = {'status': 'disabled'}
        
        # Check HTTP client
        health_info['components']['http_client'] = {
            'status': 'operational' if self.http_client else 'not_initialized'
        }
        
        return health_info

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    app.state.ai_service = QuantizedAIService()
    await app.state.ai_service.initialize()
    
    yield
    
    # Shutdown
    if hasattr(app.state.ai_service, 'http_client') and app.state.ai_service.http_client:
        await app.state.ai_service.http_client.aclose()

# Create FastAPI app
app = FastAPI(
    title="KnowledgeHub Quantized AI Service",
    description="Advanced AI Analysis with Neural Network Quantization and Lottery Ticket Hypothesis",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get AI service
def get_ai_service() -> QuantizedAIService:
    return app.state.ai_service

# API Endpoints
@app.get("/health")
async def health_check(service: QuantizedAIService = Depends(get_ai_service)):
    """Health check endpoint with comprehensive service status."""
    return await service.get_service_health()

@app.post("/api/ai/analyze-content", response_model=AnalysisResponse)
async def analyze_content_endpoint(
    request: ContentAnalysisRequest,
    service: QuantizedAIService = Depends(get_ai_service)
):
    """Analyze content using quantized AI models."""
    return await service.analyze_content(request)

@app.post("/api/ai/semantic-similarity", response_model=SimilarityResponse)  
async def semantic_similarity_endpoint(
    request: SemanticSimilarityRequest,
    service: QuantizedAIService = Depends(get_ai_service)
):
    """Advanced semantic similarity analysis beyond RAG."""
    return await service.semantic_similarity_analysis(request)

@app.post("/api/ai/analyze-behavior", response_model=BehaviorAnalysisResponse)
async def analyze_behavior_endpoint(
    request: UserBehaviorAnalysisRequest,
    service: QuantizedAIService = Depends(get_ai_service)
):
    """Analyze user behavior patterns."""
    return await service.analyze_user_behavior(request)

@app.get("/api/ai/statistics")
async def get_statistics_endpoint(service: QuantizedAIService = Depends(get_ai_service)):
    """Get comprehensive AI service statistics."""
    health_info = await service.get_service_health()
    return {
        'performance_stats': health_info['performance_stats'],
        'model_info': health_info['model_info'],
        'component_status': health_info['components']
    }

@app.post("/api/ai/learn-pattern")
async def learn_pattern_endpoint(
    pattern_data: Dict[str, Any],
    service: QuantizedAIService = Depends(get_ai_service)
):
    """Teach the AI service a new pattern."""
    try:
        await service.pattern_engine.learn_new_pattern(
            content=pattern_data.get('content', ''),
            pattern_type=pattern_data.get('pattern_type', 'custom'),
            pattern_name=pattern_data.get('pattern_name', 'User Pattern'),
            user_feedback=pattern_data.get('feedback', {})
        )
        
        return {
            'success': True,
            'message': f"Learned new pattern: {pattern_data.get('pattern_name')}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern learning failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "quantized_ai_service:app",
        host="0.0.0.0", 
        port=8003,
        reload=True,
        log_level="info"
    )

