"""
Model Pruning Optimizer for Real-Time Decision Making
Author: Pol Verbruggen - Adaptive Quantization & Model Pruning Expert

This module implements structured and unstructured pruning techniques
to maximize decision model inference speed while maintaining accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import copy
from collections import defaultdict

logger = logging.getLogger(__name__)

class PruningStrategy(Enum):
    """Different pruning strategies for neural networks"""
    MAGNITUDE = "magnitude"           # Remove weights with smallest magnitude
    GRADIENT = "gradient"            # Remove weights with smallest gradients
    STRUCTURED = "structured"        # Remove entire neurons/channels
    UNSTRUCTURED = "unstructured"    # Remove individual weights
    LOTTERY_TICKET = "lottery_ticket" # Find winning lottery ticket subnetworks
    GRADUAL = "gradual"              # Iteratively prune during training

class ImportanceCriteria(Enum):
    """Criteria for determining weight importance"""
    L1_NORM = "l1_norm"              # L1 magnitude
    L2_NORM = "l2_norm"              # L2 magnitude
    GRADIENT_MAGNITUDE = "grad_mag"  # Gradient-based importance
    TAYLOR_EXPANSION = "taylor"      # Taylor expansion approximation
    FISHER_INFORMATION = "fisher"    # Fisher information matrix

@dataclass
class PruningConfig:
    """Configuration for model pruning"""
    strategy: PruningStrategy
    importance_criteria: ImportanceCriteria
    sparsity_ratio: float  # Fraction of weights to remove (0.0 - 1.0)
    structured_granularity: str = "neuron"  # "neuron", "channel", "layer"
    gradual_steps: int = 10
    recovery_epochs: int = 5
    threshold_decay: float = 0.9

@dataclass
class PruningResult:
    """Results from model pruning operation"""
    original_params: int
    pruned_params: int
    compression_ratio: float
    accuracy_retention: float
    inference_speedup: float
    memory_reduction: float
    pruning_masks: Dict[str, torch.Tensor]
    performance_metrics: Dict[str, Any]

class ModelPruningOptimizer:
    """
    Advanced model pruning optimizer for real-time decision models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.original_model = copy.deepcopy(model)
        
        # Track pruning state
        self.pruning_masks = {}
        self.importance_scores = {}
        self.pruning_history = []
        
        # Performance tracking
        self.baseline_metrics = None
        self.current_metrics = None
        
        logger.info(f"Model pruning optimizer initialized on {device}")
        self._analyze_model_structure()
    
    def _analyze_model_structure(self):
        """Analyze model structure and compute baseline metrics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        layer_info = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                params = sum(p.numel() for p in module.parameters())
                layer_info[name] = {
                    'type': type(module).__name__,
                    'parameters': params,
                    'shape': list(module.weight.shape) if hasattr(module, 'weight') else None
                }
        
        self.model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': layer_info
        }
        
        logger.info(f"Model analysis: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    def compute_importance_scores(self, 
                                data_loader: Optional[torch.utils.data.DataLoader] = None,
                                criteria: ImportanceCriteria = ImportanceCriteria.L2_NORM) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for all model parameters
        """
        importance_scores = {}
        
        if criteria == ImportanceCriteria.L1_NORM:
            # L1 magnitude-based importance
            for name, param in self.model.named_parameters():
                if param.requires_grad and len(param.shape) > 1:  # Skip biases
                    importance_scores[name] = torch.abs(param.data)
                    
        elif criteria == ImportanceCriteria.L2_NORM:
            # L2 magnitude-based importance
            for name, param in self.model.named_parameters():
                if param.requires_grad and len(param.shape) > 1:
                    importance_scores[name] = param.data.pow(2)
                    
        elif criteria == ImportanceCriteria.GRADIENT_MAGNITUDE:
            # Gradient-based importance (requires data_loader)
            if data_loader is None:
                logger.warning("data_loader required for gradient-based importance, falling back to L2")
                return self.compute_importance_scores(data_loader, ImportanceCriteria.L2_NORM)
            
            # Accumulate gradients
            gradient_accumulator = {}
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 10:  # Limit to first 10 batches for efficiency
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                self.model.zero_grad()
                
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None and len(param.shape) > 1:
                        if name not in gradient_accumulator:
                            gradient_accumulator[name] = torch.zeros_like(param.data)
                        gradient_accumulator[name] += torch.abs(param.grad.data)
            
            # Normalize by number of batches
            for name in gradient_accumulator:
                importance_scores[name] = gradient_accumulator[name] / min(10, len(data_loader))
                
        elif criteria == ImportanceCriteria.TAYLOR_EXPANSION:
            # Taylor expansion based importance
            if data_loader is None:
                logger.warning("data_loader required for Taylor expansion, falling back to L2")
                return self.compute_importance_scores(data_loader, ImportanceCriteria.L2_NORM)
            
            self.model.train()
            taylor_scores = {}
            
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 5:  # Limit for efficiency
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                self.model.zero_grad()
                
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None and len(param.shape) > 1:
                        # Taylor expansion: importance = |weight * gradient|
                        taylor_score = torch.abs(param.data * param.grad.data)
                        
                        if name not in taylor_scores:
                            taylor_scores[name] = torch.zeros_like(param.data)
                        taylor_scores[name] += taylor_score
            
            # Normalize by number of batches
            for name in taylor_scores:
                importance_scores[name] = taylor_scores[name] / min(5, len(data_loader))
        
        self.importance_scores = importance_scores
        logger.info(f"Computed importance scores using {criteria.value} for {len(importance_scores)} parameter tensors")
        return importance_scores
    
    def apply_unstructured_pruning(self, 
                                 sparsity_ratio: float,
                                 importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> PruningResult:
        """
        Apply unstructured (weight-level) pruning to the model
        """
        if importance_scores is None:
            importance_scores = self.compute_importance_scores()
        
        # Collect all importance scores and determine global threshold
        all_scores = []
        for name, scores in importance_scores.items():
            all_scores.append(scores.flatten())
        
        all_scores = torch.cat(all_scores)
        threshold = torch.quantile(all_scores, sparsity_ratio)
        
        # Apply pruning masks
        pruning_masks = {}
        original_params = 0
        pruned_params = 0
        
        for name, param in self.model.named_parameters():
            if name in importance_scores:
                mask = importance_scores[name] > threshold
                pruning_masks[name] = mask
                
                # Apply mask to parameter
                param.data *= mask.float()
                
                original_params += param.numel()
                pruned_params += torch.sum(mask).item()
            else:
                # Keep non-pruned parameters
                if param.requires_grad:
                    original_params += param.numel()
                    pruned_params += param.numel()
        
        # Calculate metrics
        compression_ratio = (original_params - pruned_params) / original_params
        
        result = PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=compression_ratio,
            accuracy_retention=0.0,  # To be computed separately
            inference_speedup=0.0,   # To be measured
            memory_reduction=compression_ratio,
            pruning_masks=pruning_masks,
            performance_metrics={
                'pruning_type': 'unstructured',
                'sparsity_ratio': sparsity_ratio,
                'threshold': threshold.item()
            }
        )
        
        self.pruning_masks = pruning_masks
        self.pruning_history.append(result)
        
        logger.info(f"Applied unstructured pruning: {compression_ratio:.1%} compression, "
                   f"{pruned_params:,}/{original_params:,} parameters remaining")
        
        return result
    
    def apply_structured_pruning(self, 
                               sparsity_ratio: float,
                               granularity: str = "neuron",
                               importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> PruningResult:
        """
        Apply structured pruning (removing entire neurons/channels)
        """
        if importance_scores is None:
            importance_scores = self.compute_importance_scores()
        
        pruning_masks = {}
        original_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                weight_name = f"{name}.weight"
                if weight_name in importance_scores:
                    
                    if granularity == "neuron":
                        # Prune entire output neurons (rows)
                        neuron_importance = torch.sum(importance_scores[weight_name], dim=1)
                        num_neurons = neuron_importance.size(0)
                        num_to_prune = int(num_neurons * sparsity_ratio)
                        
                        if num_to_prune > 0:
                            # Get indices of least important neurons
                            _, prune_indices = torch.topk(neuron_importance, num_to_prune, largest=False)
                            
                            # Create mask
                            mask = torch.ones(num_neurons, dtype=torch.bool, device=self.device)
                            mask[prune_indices] = False
                            
                            # Apply structured pruning
                            module.weight.data = module.weight.data[mask, :]
                            if module.bias is not None:
                                module.bias.data = module.bias.data[mask]
                            
                            # Update layer dimensions
                            module.out_features = mask.sum().item()
                            
                            pruning_masks[weight_name] = mask
                            
                            logger.info(f"Pruned {num_to_prune} neurons from {name}: {mask.sum().item()}/{num_neurons} remaining")
                
                original_params += module.weight.numel()
                if module.bias is not None:
                    original_params += module.bias.numel()
                    
                pruned_params += module.weight.numel()
                if module.bias is not None:
                    pruned_params += module.bias.numel()
        
        compression_ratio = (original_params - pruned_params) / original_params if original_params > 0 else 0
        
        result = PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=compression_ratio,
            accuracy_retention=0.0,
            inference_speedup=0.0,
            memory_reduction=compression_ratio,
            pruning_masks=pruning_masks,
            performance_metrics={
                'pruning_type': 'structured',
                'granularity': granularity,
                'sparsity_ratio': sparsity_ratio
            }
        )
        
        self.pruning_masks.update(pruning_masks)
        self.pruning_history.append(result)
        
        logger.info(f"Applied structured pruning: {compression_ratio:.1%} compression")
        return result
    
    def apply_gradual_pruning(self, 
                            config: PruningConfig,
                            data_loader: Optional[torch.utils.data.DataLoader] = None) -> List[PruningResult]:
        """
        Apply gradual pruning over multiple steps
        """
        results = []
        current_sparsity = 0.0
        step_sparsity = config.sparsity_ratio / config.gradual_steps
        
        for step in range(config.gradual_steps):
            logger.info(f"Gradual pruning step {step + 1}/{config.gradual_steps}")
            
            # Compute fresh importance scores
            importance_scores = self.compute_importance_scores(data_loader, config.importance_criteria)
            
            # Apply pruning for this step
            current_sparsity += step_sparsity
            
            if config.strategy == PruningStrategy.STRUCTURED:
                result = self.apply_structured_pruning(
                    step_sparsity,
                    config.structured_granularity,
                    importance_scores
                )
            else:  # Unstructured
                result = self.apply_unstructured_pruning(
                    step_sparsity,
                    importance_scores
                )
            
            result.performance_metrics['gradual_step'] = step + 1
            result.performance_metrics['cumulative_sparsity'] = current_sparsity
            results.append(result)
            
            # Optional: Fine-tune for recovery_epochs here
            if config.recovery_epochs > 0:
                logger.info(f"Recovery training for {config.recovery_epochs} epochs would happen here")
        
        logger.info(f"Completed gradual pruning: {len(results)} steps, final sparsity {current_sparsity:.1%}")
        return results
    
    def find_lottery_ticket_subnetwork(self, 
                                     target_sparsity: float = 0.9,
                                     iterations: int = 5) -> PruningResult:
        """
        Implement Lottery Ticket Hypothesis to find winning subnetworks
        """
        logger.info(f"Finding lottery ticket subnetwork with {target_sparsity:.1%} sparsity")
        
        # Store initial weights (lottery ticket hypothesis requires rewinding)
        initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_weights[name] = param.data.clone()
        
        best_mask = None
        best_score = -float('inf')
        
        for iteration in range(iterations):
            logger.info(f"Lottery ticket iteration {iteration + 1}/{iterations}")
            
            # Reset to initial weights
            for name, param in self.model.named_parameters():
                if name in initial_weights:
                    param.data.copy_(initial_weights[name])
            
            # Compute importance scores (could be random for lottery ticket)
            importance_scores = self.compute_importance_scores()
            
            # Apply unstructured pruning
            result = self.apply_unstructured_pruning(target_sparsity, importance_scores)
            
            # Evaluate the subnetwork (simplified - would need actual evaluation)
            score = self._evaluate_subnetwork_quality()
            
            if score > best_score:
                best_score = score
                best_mask = result.pruning_masks.copy()
                logger.info(f"New best lottery ticket found with score {score:.4f}")
        
        # Apply best mask
        if best_mask:
            for name, param in self.model.named_parameters():
                if name in best_mask:
                    param.data *= best_mask[name].float()
        
        final_result = PruningResult(
            original_params=result.original_params,
            pruned_params=result.pruned_params,
            compression_ratio=result.compression_ratio,
            accuracy_retention=0.0,
            inference_speedup=0.0,
            memory_reduction=result.compression_ratio,
            pruning_masks=best_mask or {},
            performance_metrics={
                'pruning_type': 'lottery_ticket',
                'iterations': iterations,
                'best_score': best_score,
                'target_sparsity': target_sparsity
            }
        )
        
        return final_result
    
    def _evaluate_subnetwork_quality(self) -> float:
        """
        Evaluate the quality of current subnetwork
        (Simplified implementation - would use validation data in practice)
        """
        # Simple heuristic: count non-zero parameters and add some noise
        total_params = sum(p.numel() for p in self.model.parameters())
        non_zero_params = sum((p != 0).sum().item() for p in self.model.parameters())
        
        sparsity = 1.0 - (non_zero_params / total_params)
        # Add random component to simulate performance variation
        quality_score = sparsity + np.random.normal(0, 0.1)
        
        return quality_score
    
    def benchmark_inference_speed(self, 
                                input_size: Tuple[int, ...],
                                num_trials: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed of pruned vs original model
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn((1,) + input_size, device=self.device)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark pruned model
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start_time = time.perf_counter()
        
        for _ in range(num_trials):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if self.device == 'cuda' else None
        pruned_time = (time.perf_counter() - start_time) / num_trials
        
        # Benchmark original model
        self.original_model.eval()
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start_time = time.perf_counter()
        
        for _ in range(num_trials):
            with torch.no_grad():
                _ = self.original_model(dummy_input)
        
        torch.cuda.synchronize() if self.device == 'cuda' else None
        original_time = (time.perf_counter() - start_time) / num_trials
        
        speedup = original_time / pruned_time if pruned_time > 0 else 1.0
        
        return {
            'original_inference_time_ms': original_time * 1000,
            'pruned_inference_time_ms': pruned_time * 1000,
            'inference_speedup': speedup,
            'trials': num_trials
        }
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all pruning operations"""
        
        if not self.pruning_history:
            return {"message": "No pruning operations performed yet"}
        
        latest_result = self.pruning_history[-1]
        
        # Calculate cumulative compression
        total_compression = latest_result.compression_ratio
        
        # Count parameters
        current_params = sum(p.numel() for p in self.model.parameters())
        original_params = sum(p.numel() for p in self.original_model.parameters())
        
        return {
            'total_pruning_operations': len(self.pruning_history),
            'original_parameters': original_params,
            'current_parameters': current_params,
            'total_compression_ratio': (original_params - current_params) / original_params,
            'latest_operation': latest_result.performance_metrics,
            'model_info': self.model_info,
            'pruning_history': [r.performance_metrics for r in self.pruning_history]
        }
    
    def save_pruned_model(self, path: str):
        """Save the pruned model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pruning_masks': self.pruning_masks,
            'pruning_history': [r.__dict__ for r in self.pruning_history],
            'model_info': self.model_info
        }, path)
        
        logger.info(f"Saved pruned model to {path}")
    
    def load_pruned_model(self, path: str):
        """Load a previously pruned model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.pruning_masks = checkpoint['pruning_masks']
        
        # Reconstruct pruning history
        self.pruning_history = []
        for result_dict in checkpoint['pruning_history']:
            # Convert dict back to PruningResult
            result = PruningResult(**result_dict)
            self.pruning_history.append(result)
        
        if 'model_info' in checkpoint:
            self.model_info = checkpoint['model_info']
        
        logger.info(f"Loaded pruned model from {path}")


# Example usage functions
def create_test_model():
    """Create a simple test model for demonstration"""
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 5)
    )

def test_pruning_techniques():
    """Test different pruning techniques"""
    
    # Create test model
    model = create_test_model()
    optimizer = ModelPruningOptimizer(model)
    
    print("Original model:")
    print(optimizer.get_pruning_summary())
    
    # Test unstructured pruning
    print("\n=== Testing Unstructured Pruning ===")
    result = optimizer.apply_unstructured_pruning(sparsity_ratio=0.5)
    print(f"Compression ratio: {result.compression_ratio:.1%}")
    print(f"Parameters: {result.pruned_params:,}/{result.original_params:,}")
    
    # Test structured pruning on a fresh model
    model2 = create_test_model()
    optimizer2 = ModelPruningOptimizer(model2)
    
    print("\n=== Testing Structured Pruning ===")
    result2 = optimizer2.apply_structured_pruning(sparsity_ratio=0.3, granularity="neuron")
    print(f"Compression ratio: {result2.compression_ratio:.1%}")
    
    # Test lottery ticket
    model3 = create_test_model()
    optimizer3 = ModelPruningOptimizer(model3)
    
    print("\n=== Testing Lottery Ticket Hypothesis ===")
    result3 = optimizer3.find_lottery_ticket_subnetwork(target_sparsity=0.8)
    print(f"Lottery ticket compression: {result3.compression_ratio:.1%}")
    
    # Benchmark inference speed
    print("\n=== Benchmarking Inference Speed ===")
    speed_metrics = optimizer.benchmark_inference_speed(input_size=(128,))
    print(f"Speedup: {speed_metrics['inference_speedup']:.2f}x")
    print(f"Original: {speed_metrics['original_inference_time_ms']:.2f}ms")
    print(f"Pruned: {speed_metrics['pruned_inference_time_ms']:.2f}ms")

if __name__ == "__main__":
    test_pruning_techniques()
