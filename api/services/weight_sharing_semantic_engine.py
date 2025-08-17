"""
Weight Sharing Semantic Analysis Engine - Phase 2.2
Created by Tinne Smets - Expert in Weight Sharing & Knowledge Distillation

This system implements advanced weight sharing architectures for multi-task 
context understanding and semantic analysis, with efficient parameter sharing
across document, paragraph, sentence, and token-level analysis tasks.

Key Features:
- Multi-task weight sharing for semantic understanding
- Hierarchical context representation (document → paragraph → sentence → token)
- Knowledge distillation for model compression
- Cross-document semantic relationship modeling
- Context-aware entity linking and semantic role labeling
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from abc import ABC, abstractmethod
from enum import Enum

# Import existing components
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import spacy
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}")

logger = logging.getLogger(__name__)

class ContextLevel(Enum):
    """Hierarchical levels of context analysis."""
    TOKEN = "token"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    DOCUMENT = "document"
    CROSS_DOCUMENT = "cross_document"

@dataclass
class ContextRepresentation:
    """Unified context representation across all levels."""
    level: ContextLevel
    content: str
    embedding: np.ndarray
    position: Tuple[int, ...]  # Position in hierarchy (doc, para, sent, token)
    attention_weights: Optional[np.ndarray] = None
    semantic_roles: Dict[str, Any] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticTask:
    """Definition of a semantic analysis task for weight sharing."""
    task_id: str
    task_type: str  # 'entity_extraction', 'semantic_role_labeling', 'sentiment', etc.
    context_levels: List[ContextLevel]  # Which levels this task operates on
    shared_layers: List[str]  # Which layers are shared with other tasks
    task_specific_layers: List[str]  # Task-specific layer names
    loss_weight: float = 1.0
    priority: int = 1  # Higher priority tasks get more attention

class SharedEncoder(nn.Module):
    """
    Shared encoder architecture for multi-task semantic understanding.
    
    Uses weight sharing across different semantic analysis tasks while 
    maintaining task-specific capabilities through adaptive routing.
    """
    
    def __init__(
        self, 
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_shared_layers: int = 6,
        num_tasks: int = 5,
        dropout_rate: float = 0.1,
        use_layer_sharing: bool = True,
        sharing_strategy: str = "parameter_sharing"  # or "cross_stitch"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_shared_layers = num_shared_layers
        self.num_tasks = num_tasks
        self.use_layer_sharing = use_layer_sharing
        self.sharing_strategy = sharing_strategy
        
        # Shared transformation layers
        self.shared_layers = nn.ModuleList([
            self._create_shared_layer(hidden_dim, dropout_rate)
            for _ in range(num_shared_layers)
        ])
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Task routing mechanism for adaptive sharing
        self.task_router = nn.ModuleDict({
            f"task_{i}": nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_tasks)
        })
        
        # Cross-stitch units for parameter sharing (if enabled)
        if sharing_strategy == "cross_stitch":
            self.cross_stitch_units = nn.ModuleList([
                CrossStitchUnit(hidden_dim, num_tasks)
                for _ in range(num_shared_layers - 1)
            ])
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Weight sharing statistics
        self.parameter_usage = defaultdict(int)
        
        logger.info(f"SharedEncoder initialized with {sharing_strategy} sharing strategy")
    
    def _create_shared_layer(self, dim: int, dropout: float) -> nn.Module:
        """Create a shared layer for multi-task learning."""
        return nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        task_id: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task-aware weight sharing.
        
        Args:
            inputs: Input embeddings [batch_size, seq_len, input_dim]
            task_id: Which task this forward pass is for (0-based index)
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing shared and task-specific representations
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Project to hidden dimension
        hidden = self.input_projection(inputs)  # [batch, seq, hidden]
        hidden = self.layer_norm(hidden)
        
        # Store intermediate representations for analysis
        representations = {'input': hidden}
        
        # Apply shared layers with task-aware routing
        for layer_idx, shared_layer in enumerate(self.shared_layers):
            
            # Apply shared transformation
            shared_output = shared_layer(hidden)
            
            # Task-specific routing
            if f"task_{task_id}" in self.task_router:
                task_output = self.task_router[f"task_{task_id}"](hidden)
                
                # Combine shared and task-specific representations
                # Using adaptive weighting based on task importance
                alpha = torch.sigmoid(task_output.mean(dim=-1, keepdim=True))  # [batch, seq, 1]
                hidden = alpha * shared_output + (1 - alpha) * task_output
            else:
                hidden = shared_output
            
            # Apply cross-stitch units if using cross-stitch strategy
            if (self.sharing_strategy == "cross_stitch" and 
                layer_idx < len(self.cross_stitch_units)):
                hidden = self.cross_stitch_units[layer_idx](hidden, task_id)
            
            hidden = self.dropout(hidden)
            representations[f'layer_{layer_idx}'] = hidden
        
        # Update parameter usage statistics
        self.parameter_usage[task_id] += 1
        
        return {
            'hidden_states': hidden,
            'task_routing_weights': alpha if 'alpha' in locals() else None,
            'representations': representations,
            'sharing_efficiency': self._calculate_sharing_efficiency()
        }
    
    def _calculate_sharing_efficiency(self) -> float:
        """Calculate parameter sharing efficiency."""
        total_params = sum(p.numel() for p in self.parameters())
        unique_params = sum(p.numel() for p in self.shared_layers.parameters())
        task_specific_params = sum(p.numel() for p in self.task_router.parameters())
        
        # Efficiency = shared_params / total_params
        efficiency = unique_params / (unique_params + task_specific_params)
        return min(1.0, efficiency)

class CrossStitchUnit(nn.Module):
    """
    Cross-stitch unit for sharing information between tasks.
    
    Allows adaptive information sharing between different task representations
    while maintaining task-specific processing capabilities.
    """
    
    def __init__(self, hidden_dim: int, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        
        # Cross-stitch transformation matrix
        self.cross_stitch_matrix = nn.Parameter(
            torch.eye(num_tasks) + 0.1 * torch.randn(num_tasks, num_tasks)
        )
        
        # Task-specific transformations
        self.task_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_tasks)
        ])
    
    def forward(self, hidden: torch.Tensor, task_id: int) -> torch.Tensor:
        """Apply cross-stitch transformation."""
        batch_size, seq_len, hidden_dim = hidden.shape
        
        # Get task-specific transformation
        task_hidden = self.task_transforms[task_id](hidden)
        
        # Apply cross-stitch matrix (simplified for single task)
        # In full implementation, this would aggregate across all active tasks
        cross_stitch_weight = self.cross_stitch_matrix[task_id, task_id]
        output = cross_stitch_weight * task_hidden + (1 - cross_stitch_weight) * hidden
        
        return output

class WeightSharingSemanticEngine:
    """
    Main engine for weight sharing semantic analysis.
    
    Coordinates multi-task learning across different semantic analysis tasks
    while maintaining efficient parameter sharing and high-quality results.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any] = None,
        device: str = "cpu",
        enable_knowledge_distillation: bool = True
    ):
        self.device = device
        self.enable_knowledge_distillation = enable_knowledge_distillation
        self.model_config = model_config or self._default_config()
        
        # Initialize components
        self.shared_encoder = None
        self.task_definitions = {}
        
        # Statistics and monitoring
        self.performance_metrics = defaultdict(list)
        self.sharing_efficiency_history = []
        
        # Knowledge distillation components
        if enable_knowledge_distillation:
            self.teacher_models = {}
            self.distillation_losses = {}
        
        logger.info("WeightSharingSemanticEngine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the engine."""
        return {
            'input_dim': 768,
            'hidden_dim': 512,
            'num_shared_layers': 6,
            'max_sequence_length': 512,
            'dropout_rate': 0.1,
            'sharing_strategy': 'parameter_sharing',
            'enable_cross_attention': True
        }
    
    async def initialize(self):
        """Initialize all components."""
        try:
            # Create shared encoder
            self.shared_encoder = SharedEncoder(
                input_dim=self.model_config['input_dim'],
                hidden_dim=self.model_config['hidden_dim'],
                num_shared_layers=self.model_config['num_shared_layers'],
                dropout_rate=self.model_config['dropout_rate'],
                sharing_strategy=self.model_config['sharing_strategy']
            )
            
            # Define default semantic tasks
            await self._setup_default_tasks()
            
            # Initialize knowledge distillation if enabled
            if self.enable_knowledge_distillation:
                await self._initialize_knowledge_distillation()
            
            logger.info("WeightSharingSemanticEngine fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize WeightSharingSemanticEngine: {e}")
            raise
    
    async def _setup_default_tasks(self):
        """Setup default semantic analysis tasks."""
        default_tasks = [
            SemanticTask(
                task_id="entity_extraction",
                task_type="entity_extraction",
                context_levels=[ContextLevel.TOKEN, ContextLevel.SENTENCE],
                shared_layers=["shared_0", "shared_1", "shared_2"],
                task_specific_layers=["entity_head"],
                loss_weight=1.0,
                priority=2
            ),
            SemanticTask(
                task_id="semantic_similarity",
                task_type="similarity",
                context_levels=[ContextLevel.SENTENCE, ContextLevel.PARAGRAPH, ContextLevel.DOCUMENT],
                shared_layers=["shared_0", "shared_1", "shared_2", "shared_3"],
                task_specific_layers=["similarity_head"],
                loss_weight=1.2,
                priority=3
            ),
            SemanticTask(
                task_id="context_understanding",
                task_type="context_analysis",
                context_levels=[ContextLevel.PARAGRAPH, ContextLevel.DOCUMENT, ContextLevel.CROSS_DOCUMENT],
                shared_layers=["shared_1", "shared_2", "shared_3", "shared_4"],
                task_specific_layers=["context_head"],
                loss_weight=1.5,
                priority=3
            )
        ]
        
        for task in default_tasks:
            self.task_definitions[task.task_id] = task
        
        logger.info(f"Setup {len(default_tasks)} default semantic tasks")
    
    async def _initialize_knowledge_distillation(self):
        """Initialize knowledge distillation components."""
        try:
            # This would load larger teacher models for distillation
            self.distillation_losses = {
                'representation_loss': nn.MSELoss(),
                'attention_loss': nn.KLDivLoss(reduction='batchmean'),
                'prediction_loss': nn.CrossEntropyLoss()
            }
            
            logger.info("Knowledge distillation components initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge distillation: {e}")

# Factory function for easy instantiation
def create_weight_sharing_engine(
    config: Dict[str, Any] = None,
    device: str = "cpu"
) -> WeightSharingSemanticEngine:
    """Create and initialize a weight sharing semantic engine."""
    engine = WeightSharingSemanticEngine(
        model_config=config,
        device=device,
        enable_knowledge_distillation=True
    )
