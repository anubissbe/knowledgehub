"""
Cross-Domain Knowledge Synthesis API Endpoints - Phase 2.4
Created by Yves Vandenberghe - Expert in Low-Rank Factorization & Gradual Pruning

This module provides REST API endpoints for cross-domain knowledge synthesis,
low-rank factorization, gradual pruning, and multi-domain knowledge fusion.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime
import json
import numpy as np

from ..services.cross_domain_knowledge_synthesis import (
    CrossDomainKnowledgeSynthesis,
    SynthesisConfig,
    create_cross_domain_synthesis_engine
)
from ..services.gradual_domain_integration import (
    GradualDomainIntegrator,
    PruningConfig,
    create_gradual_domain_integrator
)
from ..services.cross_domain_knowledge_graph import (
    CrossDomainKnowledgeGraph,
    create_cross_domain_knowledge_graph
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances (in production, these would be properly managed)
synthesis_engine = None
domain_integrator = None
knowledge_graph = None

# Request/Response Models
class DomainRegistrationRequest(BaseModel):
    domain_id: str = Field(..., description="Unique domain identifier")
    domain_name: str = Field(..., description="Human-readable domain name")
    domain_type: str = Field(..., description="Type of domain (nlp, cv, audio, etc.)")
    knowledge_vectors: List[List[float]] = Field(..., description="Knowledge representation vectors")
    entity_mappings: Dict[str, int] = Field(..., description="Entity to index mappings")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class CrossDomainBridgeRequest(BaseModel):
    source_domain: str = Field(..., description="Source domain identifier")
    target_domain: str = Field(..., description="Target domain identifier")
    semantic_threshold: Optional[float] = Field(0.7, description="Minimum semantic alignment threshold")

class KnowledgeSynthesisRequest(BaseModel):
    query_domains: List[str] = Field(..., description="Domains to synthesize knowledge from")
    query_context: str = Field(..., description="Context for knowledge synthesis")
    max_results: Optional[int] = Field(10, description="Maximum number of results")
    compression_target: Optional[float] = Field(0.5, description="Target compression ratio")

class GradualPruningRequest(BaseModel):
    domain_tensors: Dict[str, List[List[float]]] = Field(..., description="Domain tensors to prune")
    target_compression: Optional[float] = Field(0.5, description="Target compression ratio")
    pruning_method: Optional[str] = Field("magnitude", description="Pruning method to use")

class SynthesisConfigRequest(BaseModel):
    latent_dimensions: Optional[int] = Field(256, description="Latent space dimensions")
    compression_ratio: Optional[float] = Field(0.25, description="Target compression ratio")
    factorization_method: Optional[str] = Field("svd", description="Factorization method")
    pruning_rate: Optional[float] = Field(0.1, description="Pruning rate per iteration")

# Helper Functions
async def get_synthesis_engine():
    """Get or create synthesis engine instance."""
    global synthesis_engine
    if synthesis_engine is None:
        synthesis_engine = create_cross_domain_synthesis_engine()
    return synthesis_engine

async def get_domain_integrator():
    """Get or create domain integrator instance."""
    global domain_integrator
    if domain_integrator is None:
        domain_integrator = create_gradual_domain_integrator()
    return domain_integrator

async def get_knowledge_graph():
    """Get or create knowledge graph instance."""
    global knowledge_graph
    if knowledge_graph is None:
        knowledge_graph = create_cross_domain_knowledge_graph()
    return knowledge_graph

# API Endpoints
@router.post("/api/cross-domain/domains/register")
async def register_domain_knowledge(
    request: DomainRegistrationRequest
) -> JSONResponse:
    """
    Register knowledge from a specific domain for cross-domain synthesis.
    
    This endpoint allows registration of domain-specific knowledge vectors
    that will be used for low-rank factorization and gradual pruning.
    """
    try:
        engine = await get_synthesis_engine()
        kg = await get_knowledge_graph()
        
        # Convert to numpy array
        knowledge_vectors = np.array(request.knowledge_vectors, dtype=np.float32)
        
        # Register with synthesis engine
        success = await engine.register_domain_knowledge(
            domain_id=request.domain_id,
            domain_name=request.domain_name,
            knowledge_vectors=knowledge_vectors,
            entity_mappings=request.entity_mappings,
            metadata=request.metadata
        )
        
        if success:
            # Create domain node in knowledge graph
            await kg.create_domain_node(
                domain_id=request.domain_id,
                domain_name=request.domain_name,
                domain_type=request.domain_type,
                entity_count=len(request.knowledge_vectors),
                feature_dimensions=len(request.knowledge_vectors[0]) if request.knowledge_vectors else 0,
                metadata=request.metadata
            )
            
            return JSONResponse({
                "status": "success",
                "message": f"Domain knowledge registered: {request.domain_name}",
                "domain_id": request.domain_id,
                "entity_count": len(request.knowledge_vectors),
                "feature_dimensions": len(request.knowledge_vectors[0]) if request.knowledge_vectors else 0,
                "registration_time": datetime.utcnow().isoformat()
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to register domain knowledge")
            
    except Exception as e:
        logger.error(f"Domain registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cross-domain/bridges/create")
async def create_cross_domain_bridge(
    request: CrossDomainBridgeRequest
) -> JSONResponse:
    """
    Create a bridge between two knowledge domains.
    
    Uses low-rank factorization to identify shared latent spaces and
    creates connections based on semantic alignment.
    """
    try:
        engine = await get_synthesis_engine()
        kg = await get_knowledge_graph()
        
        # Create cross-domain bridge
        bridge = await engine.create_cross_domain_bridge(
            source_domain=request.source_domain,
            target_domain=request.target_domain,
            semantic_alignment_threshold=request.semantic_threshold
        )
        
        if bridge:
            # Record bridge in knowledge graph
            bridge_id = await kg.create_domain_bridge(
                source_domain=request.source_domain,
                target_domain=request.target_domain,
                bridge_strength=bridge.bridge_strength,
                semantic_alignment=bridge.semantic_alignment,
                compression_ratio=0.0,  # Will be calculated during synthesis
                bridge_metadata={
                    "created_via_api": True,
                    "threshold_used": request.semantic_threshold
                }
            )
            
            return JSONResponse({
                "status": "success",
                "message": "Cross-domain bridge created successfully",
                "bridge_details": {
                    "bridge_id": bridge_id,
                    "source_domain": bridge.source_domain,
                    "target_domain": bridge.target_domain,
                    "bridge_strength": bridge.bridge_strength,
                    "semantic_alignment": bridge.semantic_alignment,
                    "created_at": bridge.created_at.isoformat()
                }
            })
        else:
            return JSONResponse({
                "status": "failed",
                "message": "Could not create bridge - insufficient semantic alignment",
                "threshold_used": request.semantic_threshold
            }, status_code=422)
            
    except Exception as e:
        logger.error(f"Bridge creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cross-domain/synthesis/perform")
async def perform_knowledge_synthesis(
    request: KnowledgeSynthesisRequest
) -> JSONResponse:
    """
    Perform cross-domain knowledge synthesis.
    
    Applies low-rank factorization and gradual pruning to synthesize
    knowledge across multiple domains using established bridges.
    """
    try:
        engine = await get_synthesis_engine()
        kg = await get_knowledge_graph()
        
        # Perform cross-domain knowledge synthesis
        result = await engine.synthesize_cross_domain_knowledge(
            query_domains=request.query_domains,
            query_context=request.query_context,
            max_results=request.max_results
        )
        
        if "error" not in result:
            # Record synthesis result in knowledge graph
            synthesis_metadata = result.get("synthesis_metadata", {})
            synthesis_id = f"synthesis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            await kg.record_synthesis_result(
                synthesis_id=synthesis_id,
                source_domains=request.query_domains,
                synthesis_quality=synthesis_metadata.get("average_bridge_strength", 0.0),
                results_count=len(result.get("synthesized_knowledge", [])),
                processing_time=synthesis_metadata.get("processing_time_seconds", 0.0),
                synthesis_metadata={
                    "query_context": request.query_context,
                    "bridges_used": synthesis_metadata.get("bridges_used", 0),
                    "compression_achieved": synthesis_metadata.get("average_compression_ratio", 0.0)
                }
            )
            
            return JSONResponse({
                "status": "success",
                "synthesis_result": result,
                "synthesis_id": synthesis_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return JSONResponse({
                "status": "failed",
                "error": result["error"],
                "timestamp": datetime.utcnow().isoformat()
            }, status_code=422)
            
    except Exception as e:
        logger.error(f"Knowledge synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cross-domain/pruning/apply")
async def apply_gradual_pruning(
    request: GradualPruningRequest
) -> JSONResponse:
    """
    Apply gradual pruning to domain knowledge.
    
    Uses importance-based pruning to reduce knowledge while preserving
    the most valuable cross-domain connections.
    """
    try:
        integrator = await get_domain_integrator()
        
        # Convert tensors to proper format
        import torch
        domain_tensors = {}
        for domain_id, tensor_data in request.domain_tensors.items():
            domain_tensors[domain_id] = torch.tensor(tensor_data, dtype=torch.float32)
        
        # Apply gradual pruning
        pruning_result = await integrator.apply_gradual_pruning(
            domain_tensors=domain_tensors,
            target_compression=request.target_compression
        )
        
        if "error" not in pruning_result:
            # Convert tensors back to lists for JSON serialization
            serializable_result = pruning_result.copy()
            if "pruned_domains" in serializable_result:
                serializable_result["pruned_domains"] = {
                    domain_id: tensor.cpu().numpy().tolist()
                    for domain_id, tensor in pruning_result["pruned_domains"].items()
                }
            
            return JSONResponse({
                "status": "success",
                "pruning_result": serializable_result,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return JSONResponse({
                "status": "failed",
                "error": pruning_result["error"]
            }, status_code=500)
            
    except Exception as e:
        logger.error(f"Gradual pruning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/cross-domain/analytics")
async def get_cross_domain_analytics() -> JSONResponse:
    """
    Get comprehensive analytics on cross-domain knowledge synthesis.
    
    Provides metrics on synthesis performance, compression ratios,
    bridge strengths, and overall system efficiency.
    """
    try:
        engine = await get_synthesis_engine()
        integrator = await get_domain_integrator()
        kg = await get_knowledge_graph()
        
        # Gather analytics from all components
        synthesis_analytics = engine.get_synthesis_analytics()
        integration_analytics = integrator.get_integration_analytics()
        graph_analytics = await kg.get_cross_domain_analytics()
        
        return JSONResponse({
            "status": "success",
            "cross_domain_analytics": {
                "synthesis_engine": synthesis_analytics,
                "domain_integration": integration_analytics,
                "knowledge_graph": graph_analytics,
                "system_overview": {
                    "component_status": "All systems operational",
                    "capabilities": [
                        "Low-Rank Matrix Factorization (SVD, NMF, Tensor)",
                        "Gradual Pruning with Multi-Criteria Importance",
                        "Cross-Domain Bridge Creation and Management",
                        "Multi-Modal Knowledge Fusion",
                        "Tesla V100 GPU Optimization",
                        "Neo4j Knowledge Graph Integration"
                    ],
                    "performance_summary": {
                        "domains_registered": len(synthesis_analytics.get("domain_statistics", {})),
                        "bridges_created": graph_analytics.get("cross_domain_knowledge_graph_analytics", {}).get("bridge_statistics", {}).get("total_bridges", 0),
                        "synthesis_operations": graph_analytics.get("cross_domain_knowledge_graph_analytics", {}).get("synthesis_statistics", {}).get("total_synthesis_operations", 0)
                    }
                }
            },
            "timestamp": datetime.utcnow().isoformat(),
            "phase": "2.4 - Cross-Domain Knowledge Synthesis",
            "created_by": "Yves Vandenberghe - Low-Rank Factorization & Gradual Pruning Expert"
        })
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cross-domain/config")
async def update_synthesis_config(
    request: SynthesisConfigRequest
) -> JSONResponse:
    """
    Update configuration for cross-domain synthesis engine.
    
    Allows dynamic adjustment of factorization parameters,
    pruning settings, and optimization targets.
    """
    try:
        # Create new configuration
        new_config = SynthesisConfig(
            latent_dimensions=request.latent_dimensions,
            compression_ratio=request.compression_ratio,
            factorization_method=request.factorization_method,
            pruning_rate=request.pruning_rate
        )
        
        # Update global engine instance
        global synthesis_engine
        synthesis_engine = create_cross_domain_synthesis_engine(config=new_config)
        
        return JSONResponse({
            "status": "success",
            "message": "Synthesis configuration updated successfully",
            "new_config": {
                "latent_dimensions": new_config.latent_dimensions,
                "compression_ratio": new_config.compression_ratio,
                "factorization_method": new_config.factorization_method,
                "pruning_rate": new_config.pruning_rate,
                "semantic_similarity_threshold": new_config.semantic_similarity_threshold,
                "bridge_strength_threshold": new_config.bridge_strength_threshold
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/cross-domain/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for cross-domain synthesis system."""
    try:
        # Check all components
        engine = await get_synthesis_engine()
        integrator = await get_domain_integrator()
        kg = await get_knowledge_graph()
        
        return JSONResponse({
            "status": "healthy",
            "components": {
                "synthesis_engine": "operational",
                "domain_integrator": "operational",
                "knowledge_graph": "operational"
            },
            "capabilities": [
                "Cross-Domain Knowledge Synthesis",
                "Low-Rank Matrix Factorization",
                "Gradual Pruning Integration",
                "Knowledge Graph Management",
                "Tesla V100 GPU Optimization"
            ],
            "version": "Phase 2.4",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }, status_code=500)
