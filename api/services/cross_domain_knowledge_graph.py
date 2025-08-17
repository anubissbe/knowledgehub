"""
Cross-Domain Knowledge Graph Extension - Phase 2.4
Created by Yves Vandenberghe - Expert in Low-Rank Factorization & Gradual Pruning

This service extends the Neo4j Knowledge Graph to support cross-domain knowledge synthesis
with specialized nodes and relationships for domain bridges and knowledge fusion.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)

class CrossDomainNodeType:
    """Extended node types for cross-domain knowledge."""
    DOMAIN = "Domain"
    DOMAIN_BRIDGE = "DomainBridge"
    SYNTHESIS_RESULT = "SynthesisResult"
    FACTORIZATION_MODEL = "FactorizationModel"
    PRUNING_RESULT = "PruningResult"

class CrossDomainRelationType:
    """Extended relationship types for cross-domain connections."""
    BRIDGES_TO = "BRIDGES_TO"
    SYNTHESIZES_FROM = "SYNTHESIZES_FROM"
    FACTORIZED_BY = "FACTORIZED_BY"
    PRUNED_FROM = "PRUNED_FROM"
    SEMANTICALLY_ALIGNED = "SEMANTICALLY_ALIGNED"

class CrossDomainKnowledgeGraph:
    """
    Extended knowledge graph service for cross-domain knowledge synthesis.
    """
    
    def __init__(self, base_graph_service=None):
        self.base_service = base_graph_service
        self.domain_registry = {}
        self.bridge_registry = {}
        self.synthesis_history = []
        logger.info("CrossDomainKnowledgeGraph initialized")
    
    async def create_domain_node(
        self,
        domain_id: str,
        domain_name: str,
        domain_type: str,
        entity_count: int,
        feature_dimensions: int,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a domain node in the knowledge graph."""
        
        # Register domain locally
        self.domain_registry[domain_id] = {
            'name': domain_name,
            'type': domain_type,
            'entity_count': entity_count,
            'feature_dimensions': feature_dimensions,
            'created_at': datetime.utcnow(),
            'metadata': metadata or {}
        }
        
        logger.info(f"Created domain node: {domain_name} ({domain_id})")
        return domain_id
    
    async def create_domain_bridge(
        self,
        source_domain: str,
        target_domain: str,
        bridge_strength: float,
        semantic_alignment: float,
        compression_ratio: float,
        bridge_metadata: Dict[str, Any] = None
    ) -> str:
        """Create a cross-domain bridge relationship."""
        
        bridge_id = str(uuid.uuid4())
        
        # Register bridge locally
        bridge_key = (source_domain, target_domain)
        self.bridge_registry[bridge_key] = {
            'bridge_id': bridge_id,
            'bridge_strength': bridge_strength,
            'semantic_alignment': semantic_alignment,
            'compression_ratio': compression_ratio,
            'created_at': datetime.utcnow(),
            'metadata': bridge_metadata or {}
        }
        
        logger.info(f"Created domain bridge: {source_domain} → {target_domain} "
                   f"(strength: {bridge_strength:.3f}, alignment: {semantic_alignment:.3f})")
        
        return bridge_id
    
    async def record_synthesis_result(
        self,
        synthesis_id: str,
        source_domains: List[str],
        synthesis_quality: float,
        results_count: int,
        processing_time: float,
        synthesis_metadata: Dict[str, Any] = None
    ) -> str:
        """Record a cross-domain knowledge synthesis result."""
        
        # Record locally
        self.synthesis_history.append({
            'synthesis_id': synthesis_id,
            'source_domains': source_domains,
            'synthesis_quality': synthesis_quality,
            'results_count': results_count,
            'processing_time': processing_time,
            'timestamp': datetime.utcnow(),
            'metadata': synthesis_metadata or {}
        })
        
        logger.info(f"Recorded synthesis result: {synthesis_id} "
                   f"(quality: {synthesis_quality:.3f}, domains: {len(source_domains)})")
        
        return synthesis_id
    
    async def get_cross_domain_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics on cross-domain knowledge graph."""
        
        # Domain statistics
        domain_stats = {
            'total_domains': len(self.domain_registry),
            'domain_types': {},
            'total_entities': 0,
            'avg_feature_dimensions': 0
        }
        
        if self.domain_registry:
            for domain_info in self.domain_registry.values():
                domain_type = domain_info['type']
                domain_stats['domain_types'][domain_type] = domain_stats['domain_types'].get(domain_type, 0) + 1
                domain_stats['total_entities'] += domain_info['entity_count']
            
            domain_stats['avg_feature_dimensions'] = sum(
                d['feature_dimensions'] for d in self.domain_registry.values()
            ) / len(self.domain_registry)
        
        # Bridge statistics
        bridge_stats = {
            'total_bridges': len(self.bridge_registry),
            'avg_bridge_strength': 0.0,
            'avg_semantic_alignment': 0.0,
            'avg_compression_ratio': 0.0
        }
        
        if self.bridge_registry:
            strengths = [b['bridge_strength'] for b in self.bridge_registry.values()]
            alignments = [b['semantic_alignment'] for b in self.bridge_registry.values()]
            compressions = [b['compression_ratio'] for b in self.bridge_registry.values()]
            
            bridge_stats['avg_bridge_strength'] = sum(strengths) / len(strengths)
            bridge_stats['avg_semantic_alignment'] = sum(alignments) / len(alignments)
            bridge_stats['avg_compression_ratio'] = sum(compressions) / len(compressions)
        
        # Synthesis statistics
        synthesis_stats = {
            'total_synthesis_operations': len(self.synthesis_history),
            'avg_synthesis_quality': 0.0,
            'total_results_generated': 0,
            'avg_processing_time': 0.0
        }
        
        if self.synthesis_history:
            qualities = [s['synthesis_quality'] for s in self.synthesis_history]
            processing_times = [s['processing_time'] for s in self.synthesis_history]
            
            synthesis_stats['avg_synthesis_quality'] = sum(qualities) / len(qualities)
            synthesis_stats['total_results_generated'] = sum(s['results_count'] for s in self.synthesis_history)
            synthesis_stats['avg_processing_time'] = sum(processing_times) / len(processing_times)
        
        return {
            'cross_domain_knowledge_graph_analytics': {
                'domain_statistics': domain_stats,
                'bridge_statistics': bridge_stats,
                'synthesis_statistics': synthesis_stats,
                'graph_connectivity': {
                    'domains_registered': len(self.domain_registry),
                    'bridges_created': len(self.bridge_registry),
                    'synthesis_operations': len(self.synthesis_history)
                }
            },
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'operational'
        }

def create_cross_domain_knowledge_graph(base_service=None):
    """Factory function for creating cross-domain knowledge graph."""
    return CrossDomainKnowledgeGraph(base_service=base_service)
