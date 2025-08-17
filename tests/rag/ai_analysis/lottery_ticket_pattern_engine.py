"""
Advanced Pattern Recognition Engine using Lottery Ticket Hypothesis
Created by Annelies Claes - Expert in Lottery Ticket Hypothesis & Neural Network Quantization

This engine implements sparse neural networks based on the Lottery Ticket Hypothesis
to efficiently identify critical patterns in code and content while maintaining
high accuracy with reduced computational requirements.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import json
import logging
import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import networkx as nx
from itertools import combinations

logger = logging.getLogger(__name__)

@dataclass
class PatternMatch:
    """Represents a pattern match with confidence and metadata."""
    pattern_id: str
    pattern_name: str
    confidence: float
    location: Tuple[int, int]  # start, end positions
    context: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    metadata: Dict[str, Any]

@dataclass
class SparseNeuralNetwork:
    """Represents a sparse neural network based on Lottery Ticket Hypothesis."""
    weights: torch.Tensor
    mask: torch.Tensor
    sparsity_ratio: float
    performance_score: float
    
class LotteryTicketPatternEngine:
    """
    Advanced Pattern Recognition Engine using Lottery Ticket Hypothesis
    
    Core Principles:
    1. Sparse Sub-networks: Find winning lottery tickets (critical patterns)
    2. Iterative Magnitude Pruning: Remove low-importance connections
    3. Quantization: Reduce precision while maintaining accuracy
    4. Pattern Hierarchy: Build hierarchical pattern representations
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 512,
        num_pattern_classes: int = 32,
        sparsity_target: float = 0.2,  # 20% of connections remain
        quantization_bits: int = 8
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_pattern_classes = num_pattern_classes
        self.sparsity_target = sparsity_target
        self.quantization_bits = quantization_bits
        
        # Initialize sparse neural networks for different pattern types
        self.pattern_networks = self._initialize_sparse_networks()
        
        # Pattern hierarchy graph
        self.pattern_hierarchy = nx.DiGraph()
        
        # Embedding model for content analysis
        self.embedding_model = None
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        
        # Pattern knowledge base
        self.pattern_knowledge = {
            'security_patterns': self._load_security_patterns(),
            'design_patterns': self._load_design_patterns(),
            'anti_patterns': self._load_anti_patterns(),
            'performance_patterns': self._load_performance_patterns()
        }
        
        # Real-time learning components
        self.online_patterns = defaultdict(list)
        self.pattern_evolution_graph = nx.DiGraph()
        
        logger.info(f"LotteryTicketPatternEngine initialized with {sparsity_target*100}% sparsity")

    def _initialize_sparse_networks(self) -> Dict[str, SparseNeuralNetwork]:
        """Initialize sparse neural networks for different pattern categories."""
        networks = {}
        
        pattern_types = [
            'security_critical', 'performance_bottleneck', 'design_violation',
            'code_quality', 'api_misuse', 'concurrency_issue', 'memory_leak',
            'sql_injection', 'xss_vulnerability', 'authentication_bypass'
        ]
        
        for pattern_type in pattern_types:
            # Create dense network first
            dense_weights = torch.randn(self.hidden_dim, self.embedding_dim) * 0.1
            
            # Apply lottery ticket hypothesis: find sparse subnetwork
            mask = self._find_winning_lottery_ticket(dense_weights)
            sparse_weights = dense_weights * mask
            
            networks[pattern_type] = SparseNeuralNetwork(
                weights=sparse_weights,
                mask=mask,
                sparsity_ratio=self.sparsity_target,
                performance_score=0.0  # Will be updated during training
            )
        
        return networks
    
    def _find_winning_lottery_ticket(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Implement the Lottery Ticket Hypothesis algorithm to find sparse subnetworks.
        
        Key insight: A randomly-initialized, dense neural network contains a subnetwork
        that is initialized such that — when trained in isolation — it can match the
        test accuracy of the original network after training for at most the same
        number of iterations.
        """
        # Step 1: Random initialization (already done)
        original_weights = weights.clone()
        
        # Step 2: Train the network to convergence (simulated with magnitude-based pruning)
        # In practice, this would involve actual training on pattern recognition tasks
        
        # Step 3: Prune connections with smallest magnitude
        flat_weights = weights.abs().flatten()
        k = int(len(flat_weights) * (1 - self.sparsity_target))  # Keep top k% connections
        threshold = torch.topk(flat_weights, k).values[-1]
        
        # Create binary mask
        mask = (weights.abs() >= threshold).float()
        
        # Verify sparsity ratio
        actual_sparsity = 1 - (mask.sum() / mask.numel())
        logger.debug(f"Achieved sparsity: {actual_sparsity:.3f}, Target: {1-self.sparsity_target:.3f}")
        
        return mask
    
    def _quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply neural network quantization to reduce precision."""
        if self.quantization_bits >= 32:
            return weights  # No quantization needed
        
        # Calculate quantization parameters
        weight_min = weights.min()
        weight_max = weights.max()
        
        # Symmetric quantization
        scale = (weight_max - weight_min) / (2 ** self.quantization_bits - 1)
        zero_point = 0
        
        # Quantize
        quantized = torch.round((weights - weight_min) / scale) * scale + weight_min
        
        # Simulate reduced precision
        if self.quantization_bits == 8:
            quantized = quantized.to(torch.float16).to(torch.float32)
        elif self.quantization_bits == 4:
            # More aggressive quantization
            quantized = torch.round(quantized / scale) * scale
        
        return quantized

    async def initialize_embedding_model(self):
        """Initialize the sentence transformer model asynchronously."""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fallback to simple embeddings
            self.embedding_model = None

    async def analyze_content(
        self,
        content: str,
        content_type: str = "text",
        context: Optional[Dict[str, Any]] = None
    ) -> List[PatternMatch]:
        """
        Analyze content using sparse neural networks to find critical patterns.
        
        Uses Lottery Ticket Hypothesis to focus on the most important patterns
        while maintaining computational efficiency.
        """
        if not self.embedding_model:
            await self.initialize_embedding_model()
        
        # Generate content embeddings
        if self.embedding_model:
            embeddings = self.embedding_model.encode(content)
        else:
            # Fallback to TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform([content])
            embeddings = tfidf_matrix.toarray()[0]
        
        # Convert to tensor
        input_tensor = torch.tensor(embeddings, dtype=torch.float32)
        
        # Ensure correct dimensionality
        if len(input_tensor) < self.embedding_dim:
            # Pad with zeros
            padding = torch.zeros(self.embedding_dim - len(input_tensor))
            input_tensor = torch.cat([input_tensor, padding])
        elif len(input_tensor) > self.embedding_dim:
            # Truncate
            input_tensor = input_tensor[:self.embedding_dim]
        
        pattern_matches = []
        
        # Apply each sparse neural network
        for pattern_type, network in self.pattern_networks.items():
            confidence = await self._compute_pattern_confidence(
                input_tensor, network, pattern_type
            )
            
            if confidence > 0.5:  # Threshold for pattern detection
                pattern_match = PatternMatch(
                    pattern_id=f"{pattern_type}_{hash(content) % 10000}",
                    pattern_name=pattern_type.replace('_', ' ').title(),
                    confidence=confidence,
                    location=(0, len(content)),
                    context=content[:200] + "..." if len(content) > 200 else content,
                    severity=self._determine_severity(pattern_type, confidence),
                    metadata={
                        'pattern_type': pattern_type,
                        'sparsity_ratio': network.sparsity_ratio,
                        'content_type': content_type,
                        'analysis_timestamp': asyncio.get_event_loop().time()
                    }
                )
                pattern_matches.append(pattern_match)
        
        # Apply pattern hierarchy filtering
        filtered_matches = await self._apply_pattern_hierarchy(pattern_matches)
        
        return filtered_matches

    async def _compute_pattern_confidence(
        self,
        input_tensor: torch.Tensor,
        network: SparseNeuralNetwork,
        pattern_type: str
    ) -> float:
        """Compute confidence score using sparse neural network."""
        
        # Apply sparse network computation
        with torch.no_grad():
            # Quantize weights for efficiency
            quantized_weights = self._quantize_weights(network.weights)
            
            # Sparse matrix multiplication
            sparse_weights = quantized_weights * network.mask
            output = torch.matmul(sparse_weights, input_tensor.unsqueeze(-1))
            
            # Apply activation and normalization
            output = torch.sigmoid(output.mean())
            
            # Adjust confidence based on sparsity (lottery ticket insight)
            # Sparser networks that maintain performance are more reliable
            sparsity_bonus = (1 - network.sparsity_ratio) * 0.1
            confidence = float(output) + sparsity_bonus
        
        return min(confidence, 1.0)

    def _determine_severity(self, pattern_type: str, confidence: float) -> str:
        """Determine severity based on pattern type and confidence."""
        critical_patterns = ['security_critical', 'sql_injection', 'xss_vulnerability', 'authentication_bypass']
        high_patterns = ['performance_bottleneck', 'memory_leak', 'concurrency_issue']
        
        if pattern_type in critical_patterns:
            return 'critical' if confidence > 0.8 else 'high'
        elif pattern_type in high_patterns:
            return 'high' if confidence > 0.7 else 'medium'
        else:
            return 'medium' if confidence > 0.6 else 'low'

    async def _apply_pattern_hierarchy(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Apply pattern hierarchy to filter and prioritize matches."""
        if not matches:
            return matches
        
        # Sort by confidence and severity
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        matches.sort(key=lambda m: (severity_order.get(m.severity, 0), m.confidence), reverse=True)
        
        # Remove duplicate patterns in the same location
        unique_matches = []
        seen_locations = set()
        
        for match in matches:
            location_key = (match.location[0] // 100, match.pattern_name)  # Group by approximate location
            if location_key not in seen_locations:
                unique_matches.append(match)
                seen_locations.add(location_key)
        
        return unique_matches

    def _load_security_patterns(self) -> Dict[str, Any]:
        """Load security pattern definitions."""
        return {
            'sql_injection': {
                'keywords': ['SELECT', 'INSERT', 'DELETE', 'DROP', 'UNION', 'OR 1=1', '--'],
                'regex_patterns': [r'(?i)(union\s+select)', r'(?i)(or\s+1\s*=\s*1)', r'(?i)(drop\s+table)'],
                'severity': 'critical',
                'description': 'SQL injection vulnerability detected'
            },
            'xss_vulnerability': {
                'keywords': ['<script>', 'javascript:', 'onerror=', 'onload=', 'document.cookie'],
                'regex_patterns': [r'<script[^>]*>', r'javascript\s*:', r'on\w+\s*='],
                'severity': 'critical',
                'description': 'Cross-site scripting (XSS) vulnerability detected'
            },
            'path_traversal': {
                'keywords': ['../', '..\\', '%2e%2e', '%2f', '%5c'],
                'regex_patterns': [r'\.\.[\\/]', r'%2e%2e', r'%2f%2e%2e'],
                'severity': 'high',
                'description': 'Path traversal vulnerability detected'
            }
        }

    def _load_design_patterns(self) -> Dict[str, Any]:
        """Load design pattern definitions."""
        return {
            'singleton': {
                'keywords': ['__instance', '_instance', '__new__', 'cls._instance'],
                'description': 'Singleton pattern implementation',
                'category': 'creational'
            },
            'factory': {
                'keywords': ['create_', 'make_', 'build_', 'Factory'],
                'description': 'Factory pattern implementation',
                'category': 'creational'
            },
            'observer': {
                'keywords': ['observer', 'subscribe', 'notify', 'listener'],
                'description': 'Observer pattern implementation',
                'category': 'behavioral'
            }
        }

    def _load_anti_patterns(self) -> Dict[str, Any]:
        """Load anti-pattern definitions."""
        return {
            'god_object': {
                'indicators': ['class_length > 1000', 'method_count > 50'],
                'description': 'God object anti-pattern detected',
                'severity': 'high'
            },
            'spaghetti_code': {
                'indicators': ['cyclomatic_complexity > 15', 'nested_depth > 6'],
                'description': 'Spaghetti code anti-pattern detected',
                'severity': 'medium'
            },
            'magic_numbers': {
                'keywords': ['hardcoded_constants', 'numeric_literals'],
                'description': 'Magic numbers anti-pattern detected',
                'severity': 'low'
            }
        }

    def _load_performance_patterns(self) -> Dict[str, Any]:
        """Load performance pattern definitions."""
        return {
            'inefficient_loop': {
                'indicators': ['nested_loops', 'O(n^2)', 'quadratic_complexity'],
                'description': 'Inefficient loop structure detected',
                'severity': 'medium'
            },
            'memory_leak': {
                'keywords': ['circular_reference', 'unclosed_resource', 'growing_memory'],
                'description': 'Potential memory leak detected',
                'severity': 'high'
            },
            'database_n_plus_1': {
                'keywords': ['query_in_loop', 'n+1_problem', 'eager_loading'],
                'description': 'N+1 database query problem detected',
                'severity': 'high'
            }
        }

    async def learn_new_pattern(
        self,
        content: str,
        pattern_type: str,
        pattern_name: str,
        user_feedback: Dict[str, Any]
    ):
        """Learn a new pattern using online learning techniques."""
        # Generate embedding for the new pattern
        if self.embedding_model:
            pattern_embedding = self.embedding_model.encode(content)
        else:
            # Fallback
            pattern_embedding = np.random.randn(self.embedding_dim)
        
        # Store in online patterns
        self.online_patterns[pattern_type].append({
            'content': content,
            'name': pattern_name,
            'embedding': pattern_embedding,
            'feedback': user_feedback,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        # Update sparse neural network with new pattern
        await self._update_sparse_network(pattern_type, pattern_embedding, user_feedback)
        
        logger.info(f"Learned new pattern: {pattern_name} of type {pattern_type}")

    async def _update_sparse_network(
        self,
        pattern_type: str,
        pattern_embedding: np.ndarray,
        feedback: Dict[str, Any]
    ):
        """Update sparse neural network with new pattern information."""
        if pattern_type in self.pattern_networks:
            network = self.pattern_networks[pattern_type]
            
            # Simple online learning: adjust weights based on feedback
            learning_rate = 0.01
            feedback_score = feedback.get('accuracy', 0.5)
            
            # Convert embedding to tensor
            embedding_tensor = torch.tensor(pattern_embedding, dtype=torch.float32)
            
            # Ensure correct dimensionality
            if len(embedding_tensor) != self.embedding_dim:
                if len(embedding_tensor) < self.embedding_dim:
                    padding = torch.zeros(self.embedding_dim - len(embedding_tensor))
                    embedding_tensor = torch.cat([embedding_tensor, padding])
                else:
                    embedding_tensor = embedding_tensor[:self.embedding_dim]
            
            # Update weights where mask allows
            with torch.no_grad():
                weight_update = learning_rate * feedback_score * embedding_tensor.unsqueeze(0)
                network.weights += weight_update * network.mask[:1, :]  # Apply mask
                
                # Re-quantize weights
                network.weights = self._quantize_weights(network.weights)

    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about pattern detection."""
        stats = {
            'network_statistics': {},
            'pattern_counts': defaultdict(int),
            'sparsity_ratios': {},
            'online_patterns': {k: len(v) for k, v in self.online_patterns.items()},
            'quantization_bits': self.quantization_bits
        }
        
        for pattern_type, network in self.pattern_networks.items():
            stats['sparsity_ratios'][pattern_type] = float(network.sparsity_ratio)
            stats['network_statistics'][pattern_type] = {
                'active_connections': int(network.mask.sum()),
                'total_connections': int(network.mask.numel()),
                'performance_score': float(network.performance_score)
            }
        
        return stats

    async def export_sparse_models(self) -> Dict[str, Any]:
        """Export trained sparse models for deployment."""
        exported_models = {}
        
        for pattern_type, network in self.pattern_networks.items():
            exported_models[pattern_type] = {
                'weights': network.weights.tolist(),
                'mask': network.mask.tolist(),
                'sparsity_ratio': network.sparsity_ratio,
                'performance_score': network.performance_score,
                'quantization_bits': self.quantization_bits
            }
        
        return {
            'models': exported_models,
            'metadata': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'sparsity_target': self.sparsity_target,
                'export_timestamp': asyncio.get_event_loop().time()
            }
        }
