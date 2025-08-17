"""
Gradual Domain Integration Service - Phase 2.4
Created by Yves Vandenberge - Expert in Low-Rank Factorization & Gradual Pruning

This service implements progressive domain integration using gradual pruning techniques.
It applies importance-based pruning to identify and preserve the most valuable 
cross-domain connections while removing redundant information.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class PruningConfig:
    """Configuration for gradual pruning operations."""
    initial_pruning_rate: float = 0.05  # Start conservative
    max_pruning_rate: float = 0.3       # Maximum pruning per iteration
    importance_threshold: float = 0.01  # Minimum importance to retain
    max_iterations: int = 20            # Maximum pruning iterations
    convergence_threshold: float = 0.001 # Stop when changes are minimal
    
    # Pruning strategies
    magnitude_weight: float = 0.4       # Weight for magnitude-based importance
    gradient_weight: float = 0.3        # Weight for gradient-based importance  
    connection_weight: float = 0.3      # Weight for connection-based importance

class GradualDomainIntegrator:
    """
    Progressive domain integration using advanced pruning techniques.
    
    Implements multiple pruning strategies:
    - Magnitude-based: Remove low-magnitude connections
    - Gradient-based: Remove connections with low gradients  
    - Structured: Remove entire groups of related connections
    - Connection-based: Remove weakly connected components
    """
    
    def __init__(self, config: PruningConfig = None, device: str = "cuda"):
        self.config = config or PruningConfig()
        self.device = device
        self.pruning_history = []
        self.integration_metrics = {
            "pruning_ratios": [],
            "importance_distributions": [],
            "convergence_scores": [],
            "integration_times": []
        }
        logger.info(f"GradualDomainIntegrator initialized on {device}")
    
    def compute_magnitude_importance(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute importance based on magnitude (L2 norm)."""
        return torch.norm(tensor, p=2, dim=-1)
    
    def compute_gradient_importance(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute importance based on gradient information."""
        if tensor.grad is not None:
            return torch.abs(tensor * tensor.grad).sum(dim=-1)
        else:
            # Fallback to magnitude if no gradients
            return self.compute_magnitude_importance(tensor)
    
    def compute_connection_importance(self, tensor: torch.Tensor, adjacency_matrix: torch.Tensor = None) -> torch.Tensor:
        """Compute importance based on connection strength."""
        if adjacency_matrix is not None:
            # Use actual adjacency matrix
            connection_strength = torch.mm(adjacency_matrix, torch.ones_like(tensor).sum(dim=-1, keepdim=True))
            return connection_strength.squeeze()
        else:
            # Compute self-similarity as proxy for connection strength
            normalized = F.normalize(tensor, p=2, dim=-1)
            similarity = torch.mm(normalized, normalized.t())
            return similarity.sum(dim=-1) - 1  # Subtract self-similarity
    
    def compute_composite_importance(
        self, 
        tensor: torch.Tensor,
        adjacency_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute composite importance score using multiple criteria."""
        
        magnitude_importance = self.compute_magnitude_importance(tensor)
        gradient_importance = self.compute_gradient_importance(tensor)
        connection_importance = self.compute_connection_importance(tensor, adjacency_matrix)
        
        # Normalize each importance score
        magnitude_importance = magnitude_importance / (magnitude_importance.max() + 1e-8)
        gradient_importance = gradient_importance / (gradient_importance.max() + 1e-8)
        connection_importance = connection_importance / (connection_importance.max() + 1e-8)
        
        # Weighted combination
        composite_importance = (
            self.config.magnitude_weight * magnitude_importance +
            self.config.gradient_weight * gradient_importance +
            self.config.connection_weight * connection_importance
        )
        
        return composite_importance
    
    async def apply_gradual_pruning(
        self, 
        domain_tensors: Dict[str, torch.Tensor],
        adjacency_matrices: Dict[str, torch.Tensor] = None,
        target_compression: float = 0.5
    ) -> Dict[str, Any]:
        """
        Apply gradual pruning to integrate multiple domains.
        
        Args:
            domain_tensors: Dictionary of domain tensors to prune
            adjacency_matrices: Optional adjacency matrices for each domain
            target_compression: Target compression ratio (0.5 = 50% reduction)
            
        Returns:
            Results of gradual pruning integration
        """
        start_time = datetime.utcnow()
        
        try:
            pruned_domains = {}
            pruning_stats = {}
            adjacency_matrices = adjacency_matrices or {}
            
            # Initialize pruning for each domain
            for domain_id, tensor in domain_tensors.items():
                logger.info(f"Starting gradual pruning for domain: {domain_id}")
                
                current_tensor = tensor.clone()
                adjacency = adjacency_matrices.get(domain_id)
                pruning_iterations = []
                
                # Gradual pruning loop
                for iteration in range(self.config.max_iterations):
                    # Compute importance scores
                    importance_scores = self.compute_composite_importance(
                        current_tensor, adjacency
                    )
                    
                    # Calculate adaptive pruning rate
                    current_compression = 1.0 - (current_tensor.shape[0] / tensor.shape[0])
                    remaining_compression = target_compression - current_compression
                    
                    if remaining_compression <= 0:
                        logger.info(f"Target compression reached for {domain_id}")
                        break
                    
                    # Adaptive pruning rate (more aggressive early on)
                    adaptive_rate = min(
                        self.config.max_pruning_rate,
                        max(
                            self.config.initial_pruning_rate,
                            remaining_compression * 0.3  # 30% of remaining compression
                        )
                    )
                    
                    # Determine elements to prune
                    num_elements = current_tensor.shape[0]
                    num_to_prune = max(1, int(num_elements * adaptive_rate))
                    
                    # Find lowest importance elements
                    _, prune_indices = torch.topk(
                        importance_scores, num_to_prune, largest=False
                    )
                    
                    # Create pruning mask
                    keep_mask = torch.ones(num_elements, dtype=torch.bool, device=current_tensor.device)
                    keep_mask[prune_indices] = False
                    
                    # Apply pruning
                    pruned_tensor = current_tensor[keep_mask]
                    
                    # Update adjacency matrix if provided
                    if adjacency is not None:
                        adjacency = adjacency[keep_mask][:, keep_mask]
                    
                    # Calculate iteration metrics
                    iteration_stats = {
                        "iteration": iteration,
                        "elements_before": current_tensor.shape[0],
                        "elements_after": pruned_tensor.shape[0],
                        "elements_pruned": num_to_prune,
                        "pruning_rate": adaptive_rate,
                        "compression_ratio": 1.0 - (pruned_tensor.shape[0] / tensor.shape[0]),
                        "avg_importance_pruned": importance_scores[prune_indices].mean().item(),
                        "avg_importance_kept": importance_scores[keep_mask].mean().item(),
                        "importance_variance": importance_scores.var().item()
                    }
                    pruning_iterations.append(iteration_stats)
                    
                    # Check convergence
                    if len(pruning_iterations) >= 2:
                        prev_compression = pruning_iterations[-2]["compression_ratio"]
                        curr_compression = iteration_stats["compression_ratio"]
                        compression_change = abs(curr_compression - prev_compression)
                        
                        if compression_change < self.config.convergence_threshold:
                            logger.info(f"Convergence reached for {domain_id} at iteration {iteration}")
                            break
                    
                    current_tensor = pruned_tensor
                    
                    logger.info(f"Domain {domain_id}, iteration {iteration}: "
                               f"{num_elements} → {pruned_tensor.shape[0]} elements "
                               f"({iteration_stats['compression_ratio']:.3f} compression)")
                
                # Store final pruned domain
                pruned_domains[domain_id] = current_tensor
                pruning_stats[domain_id] = {
                    "original_size": tensor.shape[0],
                    "final_size": current_tensor.shape[0],
                    "final_compression_ratio": 1.0 - (current_tensor.shape[0] / tensor.shape[0]),
                    "iterations_performed": len(pruning_iterations),
                    "iteration_details": pruning_iterations
                }
            
            # Calculate cross-domain integration metrics
            total_original_elements = sum(t.shape[0] for t in domain_tensors.values())
            total_pruned_elements = sum(t.shape[0] for t in pruned_domains.values())
            overall_compression = 1.0 - (total_pruned_elements / total_original_elements)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            self.integration_metrics["pruning_ratios"].append(overall_compression)
            self.integration_metrics["integration_times"].append(processing_time)
            
            result = {
                "pruned_domains": pruned_domains,
                "pruning_statistics": pruning_stats,
                "integration_summary": {
                    "total_domains_processed": len(domain_tensors),
                    "overall_compression_ratio": overall_compression,
                    "total_elements_original": total_original_elements,
                    "total_elements_final": total_pruned_elements,
                    "processing_time_seconds": processing_time,
                    "target_compression_achieved": overall_compression >= target_compression
                },
                "performance_metrics": {
                    "avg_compression_per_domain": np.mean([
                        stats["final_compression_ratio"] for stats in pruning_stats.values()
                    ]),
                    "compression_variance": np.var([
                        stats["final_compression_ratio"] for stats in pruning_stats.values()
                    ]),
                    "total_pruning_iterations": sum(
                        stats["iterations_performed"] for stats in pruning_stats.values()
                    )
                }
            }
            
            self.pruning_history.append(result)
            
            logger.info(f"Gradual domain integration completed: "
                       f"{overall_compression:.3f} compression in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Gradual domain integration failed: {e}")
            return {"error": str(e)}
    
    def get_integration_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics on integration performance."""
        if not self.integration_metrics["integration_times"]:
            return {"status": "No integration operations performed yet"}
        
        return {
            "performance_summary": {
                "total_integrations": len(self.integration_metrics["integration_times"]),
                "average_compression_ratio": np.mean(self.integration_metrics["pruning_ratios"]),
                "best_compression": np.max(self.integration_metrics["pruning_ratios"]),
                "worst_compression": np.min(self.integration_metrics["pruning_ratios"]),
                "compression_variance": np.var(self.integration_metrics["pruning_ratios"]),
                "average_processing_time": np.mean(self.integration_metrics["integration_times"]),
                "total_processing_time": np.sum(self.integration_metrics["integration_times"])
            },
            "configuration": {
                "initial_pruning_rate": self.config.initial_pruning_rate,
                "max_pruning_rate": self.config.max_pruning_rate,
                "importance_threshold": self.config.importance_threshold,
                "max_iterations": self.config.max_iterations,
                "pruning_strategy_weights": {
                    "magnitude": self.config.magnitude_weight,
                    "gradient": self.config.gradient_weight,
                    "connection": self.config.connection_weight
                }
            },
            "historical_performance": {
                "compression_trend": self.integration_metrics["pruning_ratios"][-10:],
                "time_trend": self.integration_metrics["integration_times"][-10:],
                "operations_performed": len(self.pruning_history)
            }
        }

def create_gradual_domain_integrator(config: PruningConfig = None):
    """Factory function for creating gradual domain integrator."""
    return GradualDomainIntegrator(config=config)
