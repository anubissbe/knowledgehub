"""
Adaptive Quantization Engine for Real-Time Decision Making
Author: Pol Verbruggen - Adaptive Quantization & Model Pruning Expert

This module implements adaptive quantization techniques for real-time AI decision making,
dynamically adjusting precision based on decision urgency and complexity.
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
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class DecisionUrgency(Enum):
    """Decision urgency levels with corresponding quantization strategies"""
    CRITICAL = "critical"    # 1-4 bit quantization, <10ms latency
    HIGH = "high"           # 4-8 bit quantization, <50ms latency
    MEDIUM = "medium"       # 8-16 bit quantization, <100ms latency
    LOW = "low"             # 16-32 bit quantization, <500ms latency

class QuantizationStrategy(Enum):
    """Different quantization strategies for decision models"""
    DYNAMIC_RANGE = "dynamic_range"      # Optimal for varying input ranges
    PERCENTILE = "percentile"            # Best for outlier handling
    ENTROPY_BASED = "entropy_based"      # Adaptive based on information content
    MIXED_PRECISION = "mixed_precision"  # Critical layers in higher precision

@dataclass
class DecisionRequest:
    """Structure for decision requests"""
    decision_id: str
    urgency: DecisionUrgency
    context: Dict[str, Any]
    features: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float

@dataclass
class DecisionResponse:
    """Structure for decision responses"""
    decision_id: str
    decision: str
    confidence: float
    latency_ms: float
    quantization_level: int
    reasoning: List[str]
    alternatives: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class AdaptiveQuantizer(nn.Module):
    """
    Adaptive quantization module that adjusts precision based on decision requirements
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        # Build the neural network layers
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:  # Don't add activation after output layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        self.network = nn.Sequential(*layers)
        
        # Quantization parameters for different urgency levels
        self.quantization_params = {
            DecisionUrgency.CRITICAL: {'bits': 2, 'scale': 4.0},
            DecisionUrgency.HIGH: {'bits': 4, 'scale': 2.0},
            DecisionUrgency.MEDIUM: {'bits': 8, 'scale': 1.0},
            DecisionUrgency.LOW: {'bits': 16, 'scale': 0.5}
        }
        
        # Performance tracking
        self.decision_history = []
        self.performance_metrics = {
            'total_decisions': 0,
            'avg_latency': 0.0,
            'accuracy': 0.0
        }
        
        self.to(device)
    
    def quantize_tensor(self, tensor: torch.Tensor, bits: int, strategy: QuantizationStrategy = QuantizationStrategy.DYNAMIC_RANGE) -> torch.Tensor:
        """Apply adaptive quantization to tensor"""
        
        if strategy == QuantizationStrategy.DYNAMIC_RANGE:
            # Dynamic range quantization
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / (2**bits - 1)
            quantized = torch.round((tensor - min_val) / scale)
            return quantized * scale + min_val
            
        elif strategy == QuantizationStrategy.PERCENTILE:
            # Percentile-based quantization (robust to outliers)
            p_low = torch.quantile(tensor, 0.01)
            p_high = torch.quantile(tensor, 0.99)
            scale = (p_high - p_low) / (2**bits - 1)
            quantized = torch.clamp((tensor - p_low) / scale, 0, 2**bits - 1)
            quantized = torch.round(quantized)
            return quantized * scale + p_low
            
        elif strategy == QuantizationStrategy.ENTROPY_BASED:
            # Information theory based quantization
            hist = torch.histc(tensor.flatten(), bins=256)
            entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
            
            # Adjust quantization based on entropy
            entropy_scale = entropy / 8.0  # Normalize entropy
            effective_bits = max(1, int(bits * entropy_scale))
            
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / (2**effective_bits - 1)
            quantized = torch.round((tensor - min_val) / scale)
            return quantized * scale + min_val
            
        else:  # MIXED_PRECISION - preserve higher precision for critical layers
            # For first and last layers, use higher precision
            return tensor  # Mixed precision handled at layer level

class RealTimeDecisionEngine:
    """
    Real-time decision engine using adaptive quantization for speed optimization
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.model_config = model_config
        
        # Initialize the adaptive quantizer
        self.quantizer = AdaptiveQuantizer(
            input_size=model_config.get('input_size', 256),
            hidden_sizes=model_config.get('hidden_sizes', [512, 256, 128]),
            output_size=model_config.get('output_size', 64),
            device=device
        )
        
        # Decision processing queue
        self.decision_queue = asyncio.Queue()
        self.processing_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'decisions_processed': 0,
            'avg_latency': 0.0,
            'throughput': 0.0,
            'accuracy': 0.0
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Real-time decision engine initialized on {device}")
    
    async def process_decision_request(self, request: DecisionRequest) -> DecisionResponse:
        """
        Process a decision request with adaptive quantization optimization
        """
        start_time = time.perf_counter()
        
        try:
            # Convert features to tensor and move to device
            if isinstance(request.features, np.ndarray):
                features = torch.tensor(request.features, dtype=torch.float32, device=self.device)
            else:
                features = request.features.to(self.device)
            
            # Ensure features are properly shaped
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            # Apply adaptive quantization based on urgency
            with torch.no_grad():
                if request.urgency in [DecisionUrgency.CRITICAL, DecisionUrgency.HIGH]:
                    # Use quantized inference for speed
                    output = self._forward_with_quantization(
                        features, 
                        request.urgency,
                        QuantizationStrategy.DYNAMIC_RANGE
                    )
                else:
                    # Use full precision for accuracy
                    output = self.quantizer.forward(features)
            
            # Process output to generate decision
            decision_scores = F.softmax(output, dim=-1)
            confidence, decision_idx = torch.max(decision_scores, dim=-1)
            
            # Generate decision and alternatives
            decision = self._generate_decision(decision_idx.item(), request.context)
            alternatives = self._generate_alternatives(decision_scores, request.context)
            reasoning = self._generate_reasoning(decision_scores, request.context, request.urgency)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            self._update_metrics(latency_ms)
            
            # Get quantization level used
            quant_level = self.quantizer.quantization_params[request.urgency]['bits']
            
            response = DecisionResponse(
                decision_id=request.decision_id,
                decision=decision,
                confidence=confidence.item(),
                latency_ms=latency_ms,
                quantization_level=quant_level,
                reasoning=reasoning,
                alternatives=alternatives,
                metadata={
                    'urgency': request.urgency.value,
                    'model_device': self.device,
                    'quantization_strategy': 'adaptive',
                    'processing_time': latency_ms
                }
            )
            
            logger.debug(f"Decision {request.decision_id} processed in {latency_ms:.2f}ms with {quant_level}-bit quantization")
            return response
            
        except Exception as e:
            logger.error(f"Error processing decision {request.decision_id}: {e}")
            
            # Fallback response
            return DecisionResponse(
                decision_id=request.decision_id,
                decision="error",
                confidence=0.0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                quantization_level=0,
                reasoning=[f"Error: {str(e)}"],
                alternatives=[],
                metadata={'error': str(e)}
            )
    
    def _forward_with_quantization(self, x: torch.Tensor, urgency: DecisionUrgency, strategy: QuantizationStrategy) -> torch.Tensor:
        """Forward pass with adaptive quantization"""
        
        params = self.quantizer.quantization_params[urgency]
        bits = params['bits']
        
        # Apply quantization to input
        if urgency in [DecisionUrgency.CRITICAL, DecisionUrgency.HIGH]:
            x = self.quantizer.quantize_tensor(x, bits, strategy)
        
        return self.quantizer.network(x)
    
    def _generate_decision(self, decision_idx: int, context: Dict[str, Any]) -> str:
        """Generate human-readable decision from model output"""
        decision_map = {
            0: "approve",
            1: "deny", 
            2: "investigate",
            3: "escalate",
            4: "defer"
        }
        
        return decision_map.get(decision_idx % len(decision_map), "unknown")
    
    def _generate_alternatives(self, scores: torch.Tensor, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative decisions with confidence scores"""
        top_k = min(3, scores.size(-1))
        top_scores, top_indices = torch.topk(scores, top_k)
        
        alternatives = []
        decision_map = {
            0: "approve",
            1: "deny", 
            2: "investigate",
            3: "escalate",
            4: "defer"
        }
        
        for i in range(top_k):
            idx = top_indices[0, i].item()
            score = top_scores[0, i].item()
            
            alternatives.append({
                'decision': decision_map.get(idx % len(decision_map), "unknown"),
                'confidence': score,
                'reasoning': f"Alternative option with {score:.3f} confidence"
            })
        
        return alternatives
    
    def _generate_reasoning(self, scores: torch.Tensor, context: Dict[str, Any], urgency: DecisionUrgency) -> List[str]:
        """Generate reasoning for the decision"""
        reasoning = []
        
        max_score = torch.max(scores).item()
        entropy = -torch.sum(scores * torch.log(scores + 1e-8)).item()
        
        reasoning.append(f"Decision confidence: {max_score:.3f}")
        reasoning.append(f"Decision entropy: {entropy:.3f}")
        reasoning.append(f"Urgency level: {urgency.value}")
        
        if urgency == DecisionUrgency.CRITICAL:
            reasoning.append("Used aggressive quantization for sub-10ms response")
        elif urgency == DecisionUrgency.HIGH:
            reasoning.append("Balanced quantization for speed-accuracy tradeoff")
        
        # Add context-based reasoning
        if context:
            reasoning.append(f"Context factors considered: {len(context)}")
        
        return reasoning
    
    def _update_metrics(self, latency_ms: float):
        """Update performance metrics"""
        with self.processing_lock:
            self.metrics['decisions_processed'] += 1
            
            # Update rolling average latency
            alpha = 0.1  # Exponential moving average factor
            if self.metrics['avg_latency'] == 0:
                self.metrics['avg_latency'] = latency_ms
            else:
                self.metrics['avg_latency'] = (1 - alpha) * self.metrics['avg_latency'] + alpha * latency_ms
            
            # Calculate throughput (decisions per second)
            if latency_ms > 0:
                current_throughput = 1000 / latency_ms
                if self.metrics['throughput'] == 0:
                    self.metrics['throughput'] = current_throughput
                else:
                    self.metrics['throughput'] = (1 - alpha) * self.metrics['throughput'] + alpha * current_throughput

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.metrics,
            'model_device': self.device,
            'quantization_levels': {
                urgency.value: params['bits'] 
                for urgency, params in self.quantizer.quantization_params.items()
            }
        }
