"""
Real-Time AI Decision Making & Recommendations Router
Author: Pol Verbruggen - Adaptive Quantization & Model Pruning Expert

FastAPI router providing endpoints for real-time decision making, 
AI recommendations, and temporal analytics integration.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import time
import torch
import numpy as np
import json

# Import our services
try:
    from ..services.realtime_decision.adaptive_quantization_engine import (
        RealTimeDecisionEngine, DecisionRequest, DecisionResponse, DecisionUrgency, 
        AdaptiveQuantizer, QuantizationStrategy
    )
    from ..services.realtime_decision.model_pruning_optimizer import (
        ModelPruningOptimizer, PruningStrategy, ImportanceCriteria, PruningConfig, PruningResult
    )
    from ..services.realtime_decision.recommendation_engine import (
        IntelligentRecommendationEngine, RecommendationRequest, RecommendationResponse,
        RecommendationType, RecommendationStrategy, Recommendation
    )
    from ..services.realtime_decision.temporal_analytics_service import (
        TimescaleAnalyticsService, TemporalMetric, MetricType, TemporalPattern, PatternAnalysis
    )
    SERVICES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Real-time decision services not available: {e}")
    SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/realtime-decision", tags=["realtime-decision"])

# Pydantic models for API
class DecisionRequestModel(BaseModel):
    decision_id: str
    urgency: str = Field(..., description="Decision urgency: critical, high, medium, low")
    context: Dict[str, Any] = Field(default_factory=dict)
    features: List[float] = Field(..., description="Input features for decision")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DecisionResponseModel(BaseModel):
    decision_id: str
    decision: str
    confidence: float
    latency_ms: float
    quantization_level: int
    reasoning: List[str]
    alternatives: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class RecommendationRequestModel(BaseModel):
    user_id: str
    request_id: str
    context: Dict[str, Any] = Field(default_factory=dict)
    recommendation_type: str = Field(..., description="Type: content, action, workflow, resource, optimization, learning")
    strategy: str = Field(..., description="Strategy: collaborative, content_based, hybrid, knowledge_graph, cross_domain, temporal")
    max_recommendations: int = Field(10, ge=1, le=50)
    diversity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    explanation_level: str = Field("basic", description="Explanation level: none, basic, detailed")

class RecommendationResponseModel(BaseModel):
    request_id: str
    user_id: str
    recommendations: List[Dict[str, Any]]
    total_candidates: int
    processing_time_ms: float
    strategy_used: str
    explanation: List[str]
    metadata: Dict[str, Any]

class PruningRequestModel(BaseModel):
    strategy: str = Field(..., description="Pruning strategy: magnitude, gradient, structured, unstructured, lottery_ticket, gradual")
    importance_criteria: str = Field("l2_norm", description="Importance criteria: l1_norm, l2_norm, grad_mag, taylor, fisher")
    sparsity_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of weights to remove")
    structured_granularity: str = Field("neuron", description="For structured pruning: neuron, channel, layer")
    gradual_steps: int = Field(10, ge=1, le=50)

class TemporalAnalysisRequestModel(BaseModel):
    metric_type: str = Field(..., description="Metric type: decision_latency, decision_confidence, recommendation_relevance, user_satisfaction, system_throughput, error_rate, resource_utilization")
    pattern_type: str = Field(..., description="Pattern type: hourly, daily, weekly, monthly, seasonal, custom")
    time_range_hours: Optional[int] = Field(None, ge=1, le=8760, description="Time range in hours (default: pattern-specific)")
    user_id: Optional[str] = Field(None, description="Filter by specific user")

class MetricRecordModel(BaseModel):
    metric_type: str
    value: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Global service instances (initialized on startup)
decision_engine: Optional[RealTimeDecisionEngine] = None
recommendation_engine: Optional[IntelligentRecommendationEngine] = None
temporal_analytics: Optional[TimescaleAnalyticsService] = None
pruning_optimizers: Dict[str, ModelPruningOptimizer] = {}

# Configuration
DEFAULT_MODEL_CONFIG = {
    'input_size': 256,
    'hidden_sizes': [512, 256, 128],
    'output_size': 64,
    'user_features': 128,
    'item_features': 256,
    'context_features': 64,
    'hidden_dim': 512,
    'embedding_dim': 256
}

TIMESCALE_CONFIG = {
    'host': 'localhost',
    'port': 5434,
    'database': 'knowledgehub_analytics',
    'user': 'postgres',
    'password': 'postgres'
}

async def get_decision_engine() -> RealTimeDecisionEngine:
    """Dependency to get decision engine instance"""
    global decision_engine
    if decision_engine is None:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Real-time decision services not available")
        decision_engine = RealTimeDecisionEngine(DEFAULT_MODEL_CONFIG)
    return decision_engine

async def get_recommendation_engine() -> IntelligentRecommendationEngine:
    """Dependency to get recommendation engine instance"""
    global recommendation_engine
    if recommendation_engine is None:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Recommendation services not available")
        recommendation_engine = IntelligentRecommendationEngine(DEFAULT_MODEL_CONFIG)
    return recommendation_engine

async def get_temporal_analytics() -> TimescaleAnalyticsService:
    """Dependency to get temporal analytics service"""
    global temporal_analytics
    if temporal_analytics is None:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Temporal analytics services not available")
        temporal_analytics = TimescaleAnalyticsService(TIMESCALE_CONFIG)
        try:
            await temporal_analytics.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize TimescaleDB: {e}")
    return temporal_analytics

# Decision Making Endpoints

@router.post("/decisions/make", response_model=DecisionResponseModel)
async def make_decision(
    request: DecisionRequestModel,
    engine: RealTimeDecisionEngine = Depends(get_decision_engine)
):
    """
    Make a real-time AI decision using adaptive quantization
    """
    try:
        # Convert request to internal format
        urgency_map = {
            'critical': DecisionUrgency.CRITICAL,
            'high': DecisionUrgency.HIGH,
            'medium': DecisionUrgency.MEDIUM,
            'low': DecisionUrgency.LOW
        }
        
        urgency = urgency_map.get(request.urgency.lower())
        if urgency is None:
            raise HTTPException(status_code=400, detail=f"Invalid urgency level: {request.urgency}")
        
        # Create decision request
        decision_request = DecisionRequest(
            decision_id=request.decision_id,
            urgency=urgency,
            context=request.context,
            features=torch.tensor(request.features, dtype=torch.float32),
            metadata=request.metadata,
            timestamp=time.time()
        )
        
        # Process decision
        response = await engine.process_decision_request(decision_request)
        
        # Convert to API response
        return DecisionResponseModel(
            decision_id=response.decision_id,
            decision=response.decision,
            confidence=response.confidence,
            latency_ms=response.latency_ms,
            quantization_level=response.quantization_level,
            reasoning=response.reasoning,
            alternatives=response.alternatives,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"Error making decision: {e}")
        raise HTTPException(status_code=500, detail=f"Decision making failed: {str(e)}")

@router.post("/decisions/batch")
async def make_batch_decisions(
    requests: List[DecisionRequestModel],
    engine: RealTimeDecisionEngine = Depends(get_decision_engine)
):
    """
    Make multiple decisions in batch for efficiency
    """
    try:
        if len(requests) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 decisions per batch")
        
        # Convert requests
        decision_requests = []
        urgency_map = {
            'critical': DecisionUrgency.CRITICAL,
            'high': DecisionUrgency.HIGH,
            'medium': DecisionUrgency.MEDIUM,
            'low': DecisionUrgency.LOW
        }
        
        for req in requests:
            urgency = urgency_map.get(req.urgency.lower())
            if urgency is None:
                raise HTTPException(status_code=400, detail=f"Invalid urgency level: {req.urgency}")
            
            decision_request = DecisionRequest(
                decision_id=req.decision_id,
                urgency=urgency,
                context=req.context,
                features=torch.tensor(req.features, dtype=torch.float32),
                metadata=req.metadata,
                timestamp=time.time()
            )
            decision_requests.append(decision_request)
        
        # Process batch
        responses = await engine.batch_process_decisions(decision_requests)
        
        # Convert responses
        api_responses = []
        for response in responses:
            api_responses.append({
                'decision_id': response.decision_id,
                'decision': response.decision,
                'confidence': response.confidence,
                'latency_ms': response.latency_ms,
                'quantization_level': response.quantization_level,
                'reasoning': response.reasoning,
                'alternatives': response.alternatives,
                'metadata': response.metadata
            })
        
        return {
            'decisions': api_responses,
            'batch_size': len(responses),
            'total_processing_time': sum(r.latency_ms for r in responses)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch decisions: {e}")
        raise HTTPException(status_code=500, detail=f"Batch decision making failed: {str(e)}")

# Recommendation Endpoints

@router.post("/recommendations/generate", response_model=RecommendationResponseModel)
async def generate_recommendations(
    request: RecommendationRequestModel,
    engine: IntelligentRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Generate AI-powered recommendations with cross-domain knowledge synthesis
    """
    try:
        # Convert request to internal format
        type_map = {
            'content': RecommendationType.CONTENT,
            'action': RecommendationType.ACTION,
            'workflow': RecommendationType.WORKFLOW,
            'resource': RecommendationType.RESOURCE,
            'optimization': RecommendationType.OPTIMIZATION,
            'learning': RecommendationType.LEARNING
        }
        
        strategy_map = {
            'collaborative': RecommendationStrategy.COLLABORATIVE,
            'content_based': RecommendationStrategy.CONTENT_BASED,
            'hybrid': RecommendationStrategy.HYBRID,
            'knowledge_graph': RecommendationStrategy.KNOWLEDGE_GRAPH,
            'cross_domain': RecommendationStrategy.CROSS_DOMAIN,
            'temporal': RecommendationStrategy.TEMPORAL
        }
        
        rec_type = type_map.get(request.recommendation_type.lower())
        strategy = strategy_map.get(request.strategy.lower())
        
        if rec_type is None:
            raise HTTPException(status_code=400, detail=f"Invalid recommendation type: {request.recommendation_type}")
        if strategy is None:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")
        
        # Create recommendation request
        rec_request = RecommendationRequest(
            user_id=request.user_id,
            request_id=request.request_id,
            context=request.context,
            recommendation_type=rec_type,
            strategy=strategy,
            max_recommendations=request.max_recommendations,
            diversity_threshold=request.diversity_threshold,
            explanation_level=request.explanation_level,
            timestamp=time.time()
        )
        
        # Generate recommendations
        response = await engine.generate_recommendations(rec_request)
        
        # Convert recommendations to dict format
        recommendations = []
        for rec in response.recommendations:
            recommendations.append({
                'item_id': rec.item_id,
                'title': rec.title,
                'description': rec.description,
                'confidence': rec.confidence,
                'relevance_score': rec.relevance_score,
                'diversity_score': rec.diversity_score,
                'explanation': rec.explanation,
                'metadata': rec.metadata,
                'cross_domain_sources': rec.cross_domain_sources
            })
        
        return RecommendationResponseModel(
            request_id=response.request_id,
            user_id=response.user_id,
            recommendations=recommendations,
            total_candidates=response.total_candidates,
            processing_time_ms=response.processing_time_ms,
            strategy_used=response.strategy_used,
            explanation=response.explanation,
            metadata=response.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@router.post("/recommendations/batch")
async def generate_batch_recommendations(
    requests: List[RecommendationRequestModel],
    engine: IntelligentRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Generate recommendations for multiple users efficiently
    """
    try:
        if len(requests) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 recommendation requests per batch")
        
        # Convert requests (reusing logic from single endpoint)
        rec_requests = []
        
        type_map = {
            'content': RecommendationType.CONTENT,
            'action': RecommendationType.ACTION,
            'workflow': RecommendationType.WORKFLOW,
            'resource': RecommendationType.RESOURCE,
            'optimization': RecommendationType.OPTIMIZATION,
            'learning': RecommendationType.LEARNING
        }
        
        strategy_map = {
            'collaborative': RecommendationStrategy.COLLABORATIVE,
            'content_based': RecommendationStrategy.CONTENT_BASED,
            'hybrid': RecommendationStrategy.HYBRID,
            'knowledge_graph': RecommendationStrategy.KNOWLEDGE_GRAPH,
            'cross_domain': RecommendationStrategy.CROSS_DOMAIN,
            'temporal': RecommendationStrategy.TEMPORAL
        }
        
        for req in requests:
            rec_type = type_map.get(req.recommendation_type.lower())
            strategy = strategy_map.get(req.strategy.lower())
            
            if rec_type is None or strategy is None:
                raise HTTPException(status_code=400, detail="Invalid recommendation type or strategy")
            
            rec_request = RecommendationRequest(
                user_id=req.user_id,
                request_id=req.request_id,
                context=req.context,
                recommendation_type=rec_type,
                strategy=strategy,
                max_recommendations=req.max_recommendations,
                diversity_threshold=req.diversity_threshold,
                explanation_level=req.explanation_level,
                timestamp=time.time()
            )
            rec_requests.append(rec_request)
        
        # Process batch
        responses = await engine.batch_generate_recommendations(rec_requests)
        
        # Convert responses
        api_responses = []
        for response in responses:
            recommendations = []
            for rec in response.recommendations:
                recommendations.append({
                    'item_id': rec.item_id,
                    'title': rec.title,
                    'description': rec.description,
                    'confidence': rec.confidence,
                    'relevance_score': rec.relevance_score,
                    'diversity_score': rec.diversity_score,
                    'explanation': rec.explanation,
                    'metadata': rec.metadata,
                    'cross_domain_sources': rec.cross_domain_sources
                })
            
            api_responses.append({
                'request_id': response.request_id,
                'user_id': response.user_id,
                'recommendations': recommendations,
                'total_candidates': response.total_candidates,
                'processing_time_ms': response.processing_time_ms,
                'strategy_used': response.strategy_used,
                'explanation': response.explanation,
                'metadata': response.metadata
            })
        
        return {
            'recommendations': api_responses,
            'batch_size': len(responses),
            'total_processing_time': sum(r.processing_time_ms for r in responses)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Batch recommendations failed: {str(e)}")

# Model Optimization Endpoints

@router.post("/models/prune")
async def prune_model(request: PruningRequestModel):
    """
    Apply model pruning optimization for faster inference
    """
    try:
        if not SERVICES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Model pruning services not available")
        
        # Create a test model for pruning (in practice, this would be an existing model)
        import torch.nn as nn
        test_model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        # Initialize pruning optimizer
        pruning_id = f"pruning_{int(time.time())}"
        pruning_optimizer = ModelPruningOptimizer(test_model)
        pruning_optimizers[pruning_id] = pruning_optimizer
        
        # Convert request parameters
        strategy_map = {
            'magnitude': PruningStrategy.MAGNITUDE,
            'gradient': PruningStrategy.GRADIENT,
            'structured': PruningStrategy.STRUCTURED,
            'unstructured': PruningStrategy.UNSTRUCTURED,
            'lottery_ticket': PruningStrategy.LOTTERY_TICKET,
            'gradual': PruningStrategy.GRADUAL
        }
        
        criteria_map = {
            'l1_norm': ImportanceCriteria.L1_NORM,
            'l2_norm': ImportanceCriteria.L2_NORM,
            'grad_mag': ImportanceCriteria.GRADIENT_MAGNITUDE,
            'taylor': ImportanceCriteria.TAYLOR_EXPANSION,
            'fisher': ImportanceCriteria.FISHER_INFORMATION
        }
        
        strategy = strategy_map.get(request.strategy.lower())
        criteria = criteria_map.get(request.importance_criteria.lower())
        
        if strategy is None or criteria is None:
            raise HTTPException(status_code=400, detail="Invalid pruning strategy or importance criteria")
        
        # Apply pruning
        if strategy == PruningStrategy.STRUCTURED:
            result = pruning_optimizer.apply_structured_pruning(
                request.sparsity_ratio,
                request.structured_granularity
            )
        elif strategy == PruningStrategy.LOTTERY_TICKET:
            result = pruning_optimizer.find_lottery_ticket_subnetwork(request.sparsity_ratio)
        elif strategy == PruningStrategy.GRADUAL:
            config = PruningConfig(
                strategy=strategy,
                importance_criteria=criteria,
                sparsity_ratio=request.sparsity_ratio,
                structured_granularity=request.structured_granularity,
                gradual_steps=request.gradual_steps
            )
            results = pruning_optimizer.apply_gradual_pruning(config)
            result = results[-1]  # Return final result
        else:
            result = pruning_optimizer.apply_unstructured_pruning(request.sparsity_ratio)
        
        # Benchmark inference speed
        speed_metrics = pruning_optimizer.benchmark_inference_speed(input_size=(128,))
        
        return {
            'pruning_id': pruning_id,
            'original_parameters': result.original_params,
            'pruned_parameters': result.pruned_params,
            'compression_ratio': result.compression_ratio,
            'inference_speedup': speed_metrics['inference_speedup'],
            'original_inference_time_ms': speed_metrics['original_inference_time_ms'],
            'pruned_inference_time_ms': speed_metrics['pruned_inference_time_ms'],
            'performance_metrics': result.performance_metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in model pruning: {e}")
        raise HTTPException(status_code=500, detail=f"Model pruning failed: {str(e)}")

@router.get("/models/pruning/{pruning_id}/summary")
async def get_pruning_summary(pruning_id: str):
    """
    Get summary of pruning operation
    """
    if pruning_id not in pruning_optimizers:
        raise HTTPException(status_code=404, detail="Pruning operation not found")
    
    try:
        optimizer = pruning_optimizers[pruning_id]
        summary = optimizer.get_pruning_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting pruning summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pruning summary: {str(e)}")

# Temporal Analytics Endpoints

@router.post("/analytics/record-metric")
async def record_metric(
    metric: MetricRecordModel,
    analytics: TimescaleAnalyticsService = Depends(get_temporal_analytics)
):
    """
    Record a temporal metric for analysis
    """
    try:
        # Convert to internal format
        metric_type_map = {
            'decision_latency': MetricType.DECISION_LATENCY,
            'decision_confidence': MetricType.DECISION_CONFIDENCE,
            'recommendation_relevance': MetricType.RECOMMENDATION_RELEVANCE,
            'user_satisfaction': MetricType.USER_SATISFACTION,
            'system_throughput': MetricType.SYSTEM_THROUGHPUT,
            'error_rate': MetricType.ERROR_RATE,
            'resource_utilization': MetricType.RESOURCE_UTILIZATION
        }
        
        metric_type = metric_type_map.get(metric.metric_type.lower())
        if metric_type is None:
            raise HTTPException(status_code=400, detail=f"Invalid metric type: {metric.metric_type}")
        
        temporal_metric = TemporalMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=metric.value,
            user_id=metric.user_id,
            session_id=metric.session_id,
            context=metric.context,
            metadata=metric.metadata
        )
        
        await analytics.record_metric(temporal_metric)
        
        return {'status': 'recorded', 'timestamp': temporal_metric.timestamp}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording metric: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")

@router.post("/analytics/analyze-patterns")
async def analyze_temporal_patterns(
    request: TemporalAnalysisRequestModel,
    analytics: TimescaleAnalyticsService = Depends(get_temporal_analytics)
):
    """
    Analyze temporal patterns in decision and recommendation metrics
    """
    try:
        # Convert parameters
        metric_type_map = {
            'decision_latency': MetricType.DECISION_LATENCY,
            'decision_confidence': MetricType.DECISION_CONFIDENCE,
            'recommendation_relevance': MetricType.RECOMMENDATION_RELEVANCE,
            'user_satisfaction': MetricType.USER_SATISFACTION,
            'system_throughput': MetricType.SYSTEM_THROUGHPUT,
            'error_rate': MetricType.ERROR_RATE,
            'resource_utilization': MetricType.RESOURCE_UTILIZATION
        }
        
        pattern_type_map = {
            'hourly': TemporalPattern.HOURLY,
            'daily': TemporalPattern.DAILY,
            'weekly': TemporalPattern.WEEKLY,
            'monthly': TemporalPattern.MONTHLY,
            'seasonal': TemporalPattern.SEASONAL,
            'custom': TemporalPattern.CUSTOM
        }
        
        metric_type = metric_type_map.get(request.metric_type.lower())
        pattern_type = pattern_type_map.get(request.pattern_type.lower())
        
        if metric_type is None or pattern_type is None:
            raise HTTPException(status_code=400, detail="Invalid metric type or pattern type")
        
        # Set time range if specified
        time_range = None
        if request.time_range_hours:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=request.time_range_hours)
            time_range = (start_time, end_time)
        
        # Perform analysis
        analysis = await analytics.analyze_temporal_patterns(
            metric_type=metric_type,
            pattern_type=pattern_type,
            time_range=time_range,
            user_id=request.user_id
        )
        
        # Convert analysis to API response
        return {
            'pattern_type': analysis.pattern_type.value,
            'metric_type': analysis.metric_type.value,
            'time_range': {
                'start': analysis.time_range[0].isoformat(),
                'end': analysis.time_range[1].isoformat()
            },
            'statistics': analysis.statistics,
            'trends': analysis.trends,
            'anomalies': [
                {
                    **anomaly,
                    'timestamp': anomaly['timestamp'].isoformat() if isinstance(anomaly.get('timestamp'), datetime) else anomaly.get('timestamp')
                }
                for anomaly in analysis.anomalies
            ],
            'predictions': [
                {
                    **pred,
                    'timestamp': pred['timestamp'].isoformat() if isinstance(pred.get('timestamp'), datetime) else pred.get('timestamp')
                }
                for pred in analysis.predictions
            ],
            'insights': analysis.insights
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")

@router.get("/analytics/dashboard/{time_window_minutes}")
async def get_dashboard_data(
    time_window_minutes: int,
    analytics: TimescaleAnalyticsService = Depends(get_temporal_analytics)
):
    """
    Get real-time dashboard data for all metrics
    """
    try:
        if time_window_minutes < 1 or time_window_minutes > 1440:  # Max 24 hours
            raise HTTPException(status_code=400, detail="Time window must be between 1 and 1440 minutes")
        
        # Get all metric types
        metric_types = [
            MetricType.DECISION_LATENCY,
            MetricType.DECISION_CONFIDENCE,
            MetricType.RECOMMENDATION_RELEVANCE,
            MetricType.SYSTEM_THROUGHPUT
        ]
        
        dashboard_data = await analytics.get_realtime_dashboard_data(
            metric_types, time_window_minutes
        )
        
        # Convert timestamps to ISO format
        for metric_name, data_points in dashboard_data.items():
            for point in data_points:
                point['timestamp'] = point['timestamp'].isoformat()
        
        return {
            'time_window_minutes': time_window_minutes,
            'data': dashboard_data,
            'retrieved_at': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data retrieval failed: {str(e)}")

# Performance and Status Endpoints

@router.get("/status")
async def get_system_status():
    """
    Get overall system status and performance metrics
    """
    try:
        status = {
            'services_available': SERVICES_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        }
        
        if SERVICES_AVAILABLE:
            # Get performance metrics from all services
            if decision_engine:
                status['decision_engine'] = decision_engine.get_performance_metrics()
            
            if recommendation_engine:
                status['recommendation_engine'] = recommendation_engine.get_performance_metrics()
            
            if temporal_analytics:
                status['temporal_analytics'] = await temporal_analytics.get_performance_metrics()
            
            status['model_optimizers'] = len(pruning_optimizers)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            'services_available': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        'status': 'healthy',
        'services_available': SERVICES_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    }

# Background task for metric recording
async def record_system_metrics():
    """Background task to record system performance metrics"""
    if not temporal_analytics:
        return
    
    try:
        # Record decision engine metrics
        if decision_engine:
            metrics = decision_engine.get_performance_metrics()
            
            await temporal_analytics.record_metric(TemporalMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.SYSTEM_THROUGHPUT,
                value=metrics.get('throughput', 0.0),
                context={'component': 'decision_engine'},
                metadata=metrics
            ))
        
        # Record recommendation engine metrics
        if recommendation_engine:
            metrics = recommendation_engine.get_performance_metrics()
            
            await temporal_analytics.record_metric(TemporalMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.SYSTEM_THROUGHPUT,
                value=metrics.get('recommendations_served', 0) / 60.0,  # Per minute
                context={'component': 'recommendation_engine'},
                metadata=metrics
            ))
            
    except Exception as e:
        logger.error(f"Error recording system metrics: {e}")

# Startup event handler
@router.on_event("startup")
async def startup_realtime_services():
    """Initialize real-time decision services on startup"""
    if SERVICES_AVAILABLE:
        logger.info("Initializing real-time decision making services...")
        
        # Initialize services will be done lazily through dependencies
        # Start background metric recording
        if temporal_analytics:
            asyncio.create_task(periodic_metric_recording())

async def periodic_metric_recording():
    """Periodic background task for metric recording"""
    while True:
        try:
            await record_system_metrics()
            await asyncio.sleep(60)  # Record every minute
        except Exception as e:
            logger.error(f"Error in periodic metric recording: {e}")
            await asyncio.sleep(60)

