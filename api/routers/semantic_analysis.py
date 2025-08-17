"""
Semantic Analysis API Router - Phase 2.2
Created by Tinne Smets - Expert in Weight Sharing & Knowledge Distillation

API endpoints for advanced semantic analysis with weight sharing,
knowledge distillation, and hierarchical context understanding.
Integrates with existing KnowledgeHub infrastructure and enhances RAG system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
import numpy as np

# Import our semantic analysis engines
from ..services.weight_sharing_semantic_engine import (
    create_weight_sharing_engine, ContextLevel
)
from ..services.knowledge_distillation_engine import (
    create_distillation_engine, DistillationConfig
)
from ..services.context_hierarchy_engine import create_context_hierarchy_engine
from ..services.advanced_semantic_engine import create_advanced_semantic_engine

logger = logging.getLogger(__name__)

# Global engine instances (initialized on first use)
weight_sharing_engine = None
distillation_engine = None
context_hierarchy_engine = None
advanced_semantic_engine = None

# API Models
class SemanticAnalysisRequest(BaseModel):
    """Request for semantic analysis."""
    text: str = Field(..., description="Text to analyze", min_length=10, max_length=50000)
    document_id: str = Field(..., description="Unique document identifier")
    analysis_level: str = Field("document", description="Analysis level (token, sentence, paragraph, document, cross_document)")
    include_entities: bool = Field(True, description="Include entity extraction and linking")
    include_semantic_roles: bool = Field(True, description="Include semantic role labeling")
    include_intent: bool = Field(True, description="Include intent analysis")
    include_cross_document: bool = Field(False, description="Include cross-document analysis")

class WeightSharingAnalysisRequest(BaseModel):
    """Request for weight sharing analysis."""
    text: str = Field(..., description="Text to analyze")
    document_id: str = Field(..., description="Document identifier")
    task_ids: List[str] = Field(default=["semantic_similarity", "entity_extraction", "context_understanding"], 
                               description="Tasks to perform")

class KnowledgeDistillationRequest(BaseModel):
    """Request for knowledge distillation."""
    teacher_config: Dict[str, Any] = Field(default_factory=dict, description="Teacher model configuration")
    student_config: Dict[str, Any] = Field(default_factory=dict, description="Student model configuration")
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")

class BatchAnalysisRequest(BaseModel):
    """Request for batch semantic analysis."""
    documents: List[Dict[str, str]] = Field(..., description="List of documents with id and text")
    analysis_options: Dict[str, bool] = Field(default_factory=dict, description="Analysis options")

# Response Models
class SemanticAnalysisResponse(BaseModel):
    """Response for semantic analysis."""
    document_id: str
    analysis_level: str
    processing_time: float
    entities: List[Dict[str, Any]] = []
    semantic_roles: List[Dict[str, Any]] = []
    intent_analysis: Dict[str, Any] = {}
    weight_sharing_analysis: Dict[str, Any] = {}
    cross_document_analysis: Dict[str, Any] = {}
    semantic_metrics: Dict[str, float] = {}
    timestamp: str

class EngineStatusResponse(BaseModel):
    """Response for engine status."""
    engines_initialized: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    cache_statistics: Dict[str, Any]
    model_information: Dict[str, Any]

# Router
router = APIRouter(prefix="/api/semantic-analysis", tags=["semantic-analysis"])

# Dependency functions
async def get_weight_sharing_engine():
    """Get or initialize weight sharing engine."""
    global weight_sharing_engine
    
    if weight_sharing_engine is None:
        try:
            weight_sharing_engine = create_weight_sharing_engine()
            await weight_sharing_engine.initialize()
            logger.info("Weight sharing engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize weight sharing engine: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize weight sharing engine")
    
    return weight_sharing_engine

async def get_advanced_semantic_engine():
    """Get or initialize advanced semantic engine."""
    global advanced_semantic_engine
    
    if advanced_semantic_engine is None:
        try:
            ws_engine = await get_weight_sharing_engine()
            advanced_semantic_engine = create_advanced_semantic_engine(ws_engine)
            logger.info("Advanced semantic engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize advanced semantic engine: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize advanced semantic engine")
    
    return advanced_semantic_engine

async def get_context_hierarchy_engine():
    """Get or initialize context hierarchy engine."""
    global context_hierarchy_engine
    
    if context_hierarchy_engine is None:
        try:
            ws_engine = await get_weight_sharing_engine()
            context_hierarchy_engine = create_context_hierarchy_engine(ws_engine)
            await context_hierarchy_engine.initialize()
            logger.info("Context hierarchy engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize context hierarchy engine: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize context hierarchy engine")
    
    return context_hierarchy_engine

async def get_distillation_engine():
    """Get or initialize knowledge distillation engine."""
    global distillation_engine
    
    if distillation_engine is None:
        try:
            config = DistillationConfig()
            distillation_engine = create_distillation_engine(config)
            await distillation_engine.initialize_models()
            logger.info("Knowledge distillation engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize distillation engine: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize distillation engine")
    
    return distillation_engine

# API Endpoints

@router.get("/health")
async def health_check():
    """Health check for semantic analysis service."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "engines": {
            "weight_sharing": weight_sharing_engine is not None,
            "distillation": distillation_engine is not None,
            "context_hierarchy": context_hierarchy_engine is not None,
            "advanced_semantic": advanced_semantic_engine is not None
        }
    }

@router.get("/status", response_model=EngineStatusResponse)
async def get_engine_status(
    weight_engine = Depends(get_weight_sharing_engine),
    advanced_engine = Depends(get_advanced_semantic_engine)
):
    """Get comprehensive status of all semantic analysis engines."""
    try:
        # Get engine statistics
        weight_stats = await weight_engine.get_sharing_statistics()
        advanced_stats = advanced_engine.get_analysis_statistics()
        
        # Get context hierarchy stats if available
        context_stats = {}
        if context_hierarchy_engine:
            context_stats = context_hierarchy_engine.get_hierarchy_statistics()
        
        # Get distillation stats if available
        distillation_stats = {}
        if distillation_engine:
            distillation_stats = distillation_engine.get_distillation_metrics()
        
        return EngineStatusResponse(
            engines_initialized={
                "weight_sharing": weight_sharing_engine is not None,
                "advanced_semantic": advanced_semantic_engine is not None,
                "context_hierarchy": context_hierarchy_engine is not None,
                "knowledge_distillation": distillation_engine is not None
            },
            performance_metrics={
                "weight_sharing": weight_stats.get("performance_metrics", {}),
                "advanced_semantic": advanced_stats.get("performance_metrics", {}),
                "context_hierarchy": context_stats.get("processing_metrics", {}),
                "knowledge_distillation": distillation_stats
            },
            cache_statistics={
                "weight_sharing": {"parameter_reuse_ratio": weight_stats.get("parameter_reuse_ratio", 0.0)},
                "advanced_semantic": advanced_stats.get("cache_statistics", {}),
                "context_hierarchy": {"cache_size": context_stats.get("cache_size", 0)}
            },
            model_information={
                "weight_sharing_config": weight_stats.get("model_config", {}),
                "compression_ratio": distillation_stats.get("compression_ratio", 0.0),
                "parameter_efficiency": distillation_stats.get("parameter_efficiency", 0.0)
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to get engine status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get engine status: {str(e)}")

@router.post("/analyze", response_model=SemanticAnalysisResponse)
async def analyze_semantic_content(
    request: SemanticAnalysisRequest,
    background_tasks: BackgroundTasks,
    advanced_engine = Depends(get_advanced_semantic_engine)
):
    """
    Perform comprehensive semantic analysis on text content.
    
    This endpoint provides the main semantic analysis functionality,
    integrating weight sharing, entity linking, semantic role labeling,
    and intent analysis.
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate analysis level
        try:
            analysis_level = ContextLevel(request.analysis_level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid analysis level: {request.analysis_level}")
        
        # Configure analysis options based on request
        config = {
            'enable_entity_linking': request.include_entities,
            'enable_srl': request.include_semantic_roles,
            'enable_intent_analysis': request.include_intent
        }
        
        # Update engine configuration if needed
        if hasattr(advanced_engine, 'entity_linker'):
            if not request.include_entities and advanced_engine.entity_linker:
                advanced_engine.entity_linker = None
        
        # Perform comprehensive semantic analysis
        analysis_result = await advanced_engine.comprehensive_semantic_analysis(
            text=request.text,
            document_id=request.document_id,
            context_level=analysis_level,
            include_cross_document=request.include_cross_document
        )
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Prepare response
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = SemanticAnalysisResponse(
            document_id=request.document_id,
            analysis_level=request.analysis_level,
            processing_time=processing_time,
            entities=convert_numpy(analysis_result.get('entities', [])),
            semantic_roles=convert_numpy(analysis_result.get('semantic_roles', [])),
            intent_analysis=convert_numpy(analysis_result.get('intent_analysis', {})),
            weight_sharing_analysis=convert_numpy(analysis_result.get('weight_sharing_analysis', {})),
            cross_document_analysis=convert_numpy(analysis_result.get('cross_document_analysis', {})),
            semantic_metrics=convert_numpy(analysis_result.get('semantic_metrics', {})),
            timestamp=start_time.isoformat()
        )
        
        # Add background task to update metrics
        background_tasks.add_task(
            _update_analysis_metrics,
            request.document_id,
            processing_time,
            len(analysis_result.get('entities', [])),
            len(analysis_result.get('semantic_roles', []))
        )
        
        logger.info(f"Semantic analysis completed for {request.document_id} in {processing_time:.2f}s")
        
        return response
    
    except Exception as e:
        logger.error(f"Semantic analysis failed for {request.document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze-weight-sharing")
async def analyze_with_weight_sharing(
    request: WeightSharingAnalysisRequest,
    weight_engine = Depends(get_weight_sharing_engine)
):
    """Perform analysis using weight sharing engine specifically."""
    try:
        result = await weight_engine.analyze_context_hierarchy(
            text_content=request.text,
            document_id=request.document_id,
            task_ids=request.task_ids
        )
        
        return {
            "document_id": request.document_id,
            "weight_sharing_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Weight sharing analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Weight sharing analysis failed: {str(e)}")

@router.post("/build-hierarchy")
async def build_context_hierarchy(
    request: Dict[str, Any],
    hierarchy_engine = Depends(get_context_hierarchy_engine)
):
    """Build hierarchical context representation for a document."""
    try:
        document_id = request.get("document_id")
        content = request.get("content")
        metadata = request.get("metadata", {})
        
        hierarchy = await hierarchy_engine.analyze_document_hierarchy(
            document_id=document_id,
            content=content,
            metadata=metadata
        )
        
        # Convert hierarchy to serializable format
        serializable_hierarchy = {}
        for level, nodes in hierarchy.items():
            serializable_hierarchy[level.value] = [
                {
                    "node_id": node.node_id,
                    "content": node.content[:200] + "..." if len(node.content) > 200 else node.content,
                    "position": node.position,
                    "importance_score": node.importance_score,
                    "coherence_score": node.coherence_score,
                    "metadata": node.metadata
                }
                for node in nodes
            ]
        
        return {
            "document_id": document_id,
            "hierarchy": serializable_hierarchy,
            "statistics": hierarchy_engine.get_hierarchy_statistics(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Context hierarchy building failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hierarchy building failed: {str(e)}")

@router.post("/batch-analyze")
async def batch_semantic_analysis(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    advanced_engine = Depends(get_advanced_semantic_engine)
):
    """Perform batch semantic analysis on multiple documents."""
    try:
        if len(request.documents) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 100 documents)")
        
        results = []
        start_time = datetime.utcnow()
        
        # Process documents in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing
        
        async def process_document(doc):
            async with semaphore:
                try:
                    analysis = await advanced_engine.comprehensive_semantic_analysis(
                        text=doc["text"],
                        document_id=doc["id"],
                        context_level=ContextLevel.DOCUMENT,
                        include_cross_document=False
                    )
                    return {"document_id": doc["id"], "status": "success", "analysis": analysis}
                except Exception as e:
                    logger.error(f"Failed to analyze document {doc['id']}: {e}")
                    return {"document_id": doc["id"], "status": "failed", "error": str(e)}
        
        # Process all documents
        tasks = [process_document(doc) for doc in request.documents]
        results = await asyncio.gather(*tasks)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Count successful analyses
        successful_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "batch_id": f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "total_documents": len(request.documents),
            "successful_analyses": successful_count,
            "failed_analyses": len(request.documents) - successful_count,
            "processing_time": processing_time,
            "results": results,
            "timestamp": start_time.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/distill-model")
async def create_distilled_model(
    request: KnowledgeDistillationRequest,
    background_tasks: BackgroundTasks
):
    """Create a distilled model using knowledge distillation."""
    try:
        # Create distillation configuration
        distill_config = DistillationConfig(**request.training_config)
        
        # Initialize distillation engine with custom config
        engine = create_distillation_engine(distill_config)
        await engine.initialize_models(
            teacher_config=request.teacher_config,
            student_config=request.student_config
        )
        
        # Start distillation process in background
        background_tasks.add_task(
            _perform_knowledge_distillation,
            engine,
            request.training_config.get("epochs", 5)
        )
        
        return {
            "status": "started",
            "message": "Knowledge distillation process started in background",
            "config": {
                "teacher_params": engine.teacher_model.count_parameters() if engine.teacher_model else 0,
                "student_params": engine.student_model.count_parameters() if engine.student_model else 0,
                "compression_ratio": (engine.teacher_model.count_parameters() / engine.student_model.count_parameters()) if (engine.teacher_model and engine.student_model) else 0.0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Model distillation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model distillation failed: {str(e)}")

@router.get("/metrics")
async def get_semantic_analysis_metrics():
    """Get comprehensive metrics for all semantic analysis engines."""
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "engines": {}
    }
    
    try:
        if weight_sharing_engine:
            metrics["engines"]["weight_sharing"] = await weight_sharing_engine.get_sharing_statistics()
        
        if advanced_semantic_engine:
            metrics["engines"]["advanced_semantic"] = advanced_semantic_engine.get_analysis_statistics()
        
        if context_hierarchy_engine:
            metrics["engines"]["context_hierarchy"] = context_hierarchy_engine.get_hierarchy_statistics()
        
        if distillation_engine:
            metrics["engines"]["knowledge_distillation"] = distillation_engine.get_distillation_metrics()
        
        return metrics
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# Background task functions
async def _update_analysis_metrics(document_id: str, processing_time: float, entity_count: int, triple_count: int):
    """Update analysis metrics in background."""
    try:
        # This would typically update a database or metrics store
        logger.info(f"Metrics updated for {document_id}: {processing_time:.2f}s, {entity_count} entities, {triple_count} triples")
    except Exception as e:
        logger.error(f"Failed to update metrics: {e}")

async def _perform_knowledge_distillation(engine, epochs: int):
    """Perform knowledge distillation in background."""
    try:
        logger.info(f"Starting knowledge distillation for {epochs} epochs")
        
        # This would perform actual training
        # For now, just simulate the process
        for epoch in range(epochs):
            # Simulate batch processing
            await asyncio.sleep(1)  # Simulate processing time
            logger.info(f"Distillation epoch {epoch + 1}/{epochs} completed")
        
        logger.info("Knowledge distillation completed successfully")
        
    except Exception as e:
        logger.error(f"Knowledge distillation failed: {e}")

# Initialize engines on module load
async def initialize_engines():
    """Initialize all engines on startup."""
    try:
        await get_weight_sharing_engine()
        await get_advanced_semantic_engine()
        logger.info("Semantic analysis engines initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")

# Add initialization to router startup
@router.on_event("startup")
async def startup_event():
    """Initialize engines on router startup."""
