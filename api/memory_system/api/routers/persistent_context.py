"""
Persistent Context API Router

Provides endpoints for managing and querying the persistent context system.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from ...core.persistent_context import (
    get_persistent_context_manager,
    PersistentContextManager,
    ContextType,
    ContextScope,
    ContextVector
)
from ....models import get_db
from sqlalchemy.orm import Session

router = APIRouter()


# Pydantic models
class ContextAddRequest(BaseModel):
    """Request to add new context"""
    content: str = Field(..., description="Context content")
    context_type: ContextType = Field(..., description="Type of context")
    scope: ContextScope = Field(..., description="Scope of context")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")
    related_entities: List[str] = Field(default_factory=list, description="Related entities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContextQueryRequest(BaseModel):
    """Request to query context"""
    query: str = Field(..., description="Query text")
    context_type: Optional[ContextType] = Field(None, description="Filter by context type")
    scope: Optional[ContextScope] = Field(None, description="Filter by scope")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")


class ContextVectorResponse(BaseModel):
    """Response model for context vector"""
    id: str
    content: str
    context_type: str
    scope: str
    importance: float
    last_accessed: datetime
    access_count: int
    related_entities: List[str]
    metadata: Dict[str, Any]


class ContextSummaryResponse(BaseModel):
    """Response model for context summary"""
    total_vectors: int
    total_clusters: int
    context_types: Dict[str, int]
    scopes: Dict[str, int]
    top_importance: List[Dict[str, Any]]
    recent_access: List[Dict[str, Any]]


class ContextAnalyticsResponse(BaseModel):
    """Response model for context analytics"""
    total_vectors: int
    total_clusters: int
    avg_importance: float
    avg_access_count: float
    context_type_distribution: Dict[str, int]
    scope_distribution: Dict[str, int]
    cluster_health: Dict[str, Dict[str, Any]]
    access_patterns: Dict[str, Any]
    memory_usage: int


@router.post("/context", response_model=Dict[str, str])
async def add_context(
    request: ContextAddRequest,
    db: Session = Depends(get_db)
):
    """
    Add new context to the persistent system
    
    Creates a new context vector and integrates it into the
    persistent context graph for future retrieval.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        context_id = await manager.add_context(
            content=request.content,
            context_type=request.context_type,
            scope=request.scope,
            importance=request.importance,
            related_entities=request.related_entities,
            metadata=request.metadata
        )
        
        if context_id:
            return {
                "message": "Context added successfully",
                "context_id": str(context_id)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add context")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding context: {str(e)}")


@router.post("/context/query", response_model=List[ContextVectorResponse])
async def query_context(
    request: ContextQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Query persistent context for relevant information
    
    Searches the persistent context graph for vectors
    relevant to the given query.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        vectors = await manager.retrieve_context(
            query=request.query,
            context_type=request.context_type,
            scope=request.scope,
            limit=request.limit
        )
        
        return [
            ContextVectorResponse(
                id=str(vector.id),
                content=vector.content,
                context_type=vector.context_type.value,
                scope=vector.scope.value,
                importance=vector.importance,
                last_accessed=vector.last_accessed,
                access_count=vector.access_count,
                related_entities=vector.related_entities,
                metadata=vector.metadata
            )
            for vector in vectors
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying context: {str(e)}")


@router.get("/context/summary", response_model=ContextSummaryResponse)
async def get_context_summary(
    session_id: Optional[UUID] = Query(None, description="Filter by session ID"),
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    db: Session = Depends(get_db)
):
    """
    Get summary of persistent context
    
    Returns overview statistics and highlights from the
    persistent context system.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        summary = await manager.get_context_summary(
            session_id=session_id,
            project_id=project_id
        )
        
        return ContextSummaryResponse(**summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting context summary: {str(e)}")


@router.get("/context/analytics", response_model=ContextAnalyticsResponse)
async def get_context_analytics(
    db: Session = Depends(get_db)
):
    """
    Get detailed analytics about persistent context
    
    Returns comprehensive analytics about context usage,
    patterns, and system health.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        analytics = await manager.get_context_analytics()
        
        return ContextAnalyticsResponse(**analytics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting context analytics: {str(e)}")


@router.post("/context/decay")
async def apply_importance_decay(
    db: Session = Depends(get_db)
):
    """
    Apply importance decay to all context vectors
    
    Reduces importance scores over time to prioritize
    recently accessed and relevant context.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        await manager.decay_importance()
        
        return {
            "message": "Importance decay applied successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying decay: {str(e)}")


@router.post("/context/cleanup")
async def cleanup_old_context(
    max_age_days: int = Query(90, ge=1, le=365, description="Maximum age in days"),
    db: Session = Depends(get_db)
):
    """
    Clean up old, unused context vectors
    
    Removes context vectors that are old, have low importance,
    and haven't been accessed recently.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        await manager.cleanup_old_context(max_age_days=max_age_days)
        
        return {
            "message": f"Context cleanup completed for items older than {max_age_days} days",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up context: {str(e)}")


@router.get("/context/types", response_model=List[str])
async def get_context_types():
    """
    Get all available context types
    
    Returns a list of all supported context types
    for categorizing persistent context.
    """
    return [context_type.value for context_type in ContextType]


@router.get("/context/scopes", response_model=List[str])
async def get_context_scopes():
    """
    Get all available context scopes
    
    Returns a list of all supported context scopes
    for defining persistence boundaries.
    """
    return [scope.value for scope.value in ContextScope]


@router.get("/context/health")
async def get_context_health(
    db: Session = Depends(get_db)
):
    """
    Get health status of persistent context system
    
    Returns system health metrics and status information.
    No authentication required for health checks.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        analytics = await manager.get_context_analytics()
        
        # Determine health status
        health_status = "healthy"
        if analytics["total_vectors"] == 0:
            health_status = "empty"
        elif analytics["avg_importance"] < 0.3:
            health_status = "degraded"
        elif analytics["total_clusters"] == 0:
            health_status = "unclustered"
        
        return {
            "status": health_status,
            "persistent_context": "active",
            "total_vectors": analytics["total_vectors"],
            "total_clusters": analytics["total_clusters"],
            "avg_importance": analytics["avg_importance"],
            "context_types": len(analytics["context_type_distribution"]),
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/context/search")
async def search_context_simple(
    q: str = Query(..., description="Search query"),
    type: Optional[str] = Query(None, description="Context type filter"),
    scope: Optional[str] = Query(None, description="Context scope filter"),
    limit: int = Query(10, ge=1, le=50, description="Result limit"),
    db: Session = Depends(get_db)
):
    """
    Simple context search endpoint
    
    Provides a simple GET-based search interface for
    querying persistent context.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        # Convert string parameters to enums
        context_type = None
        if type:
            try:
                context_type = ContextType(type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid context type: {type}")
        
        scope_enum = None
        if scope:
            try:
                scope_enum = ContextScope(scope)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}")
        
        vectors = await manager.retrieve_context(
            query=q,
            context_type=context_type,
            scope=scope_enum,
            limit=limit
        )
        
        return {
            "query": q,
            "results": [
                {
                    "id": str(vector.id),
                    "content": vector.content[:200] + "..." if len(vector.content) > 200 else vector.content,
                    "context_type": vector.context_type.value,
                    "scope": vector.scope.value,
                    "importance": vector.importance,
                    "access_count": vector.access_count,
                    "last_accessed": vector.last_accessed.isoformat()
                }
                for vector in vectors
            ],
            "total_results": len(vectors),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching context: {str(e)}")


@router.get("/context/clusters")
async def get_context_clusters(
    db: Session = Depends(get_db)
):
    """
    Get information about context clusters
    
    Returns information about the current context clusters
    and their characteristics.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        clusters_info = []
        for cluster in manager.context_graph.clusters.values():
            clusters_info.append({
                "id": str(cluster.id),
                "name": cluster.name,
                "description": cluster.description,
                "vector_count": len(cluster.vectors),
                "coherence_score": cluster.coherence_score,
                "last_updated": cluster.last_updated.isoformat(),
                "sample_vectors": [
                    {
                        "id": str(v.id),
                        "content": v.content[:100] + "..." if len(v.content) > 100 else v.content,
                        "importance": v.importance
                    }
                    for v in cluster.vectors[:3]  # Show first 3 vectors
                ]
            })
        
        return {
            "clusters": clusters_info,
            "total_clusters": len(clusters_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting clusters: {str(e)}")


@router.get("/context/vectors/{vector_id}")
async def get_context_vector(
    vector_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific context vector
    
    Returns comprehensive information about a context vector
    including its connections and metadata.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        vector = manager.context_graph.nodes.get(vector_id)
        if not vector:
            raise HTTPException(status_code=404, detail="Context vector not found")
        
        # Get connections
        connections = manager.context_graph.edges.get(vector_id, [])
        connection_info = []
        for connected_id, similarity in connections[:5]:  # Top 5 connections
            connected_vector = manager.context_graph.nodes.get(connected_id)
            if connected_vector:
                connection_info.append({
                    "id": str(connected_id),
                    "content": connected_vector.content[:100] + "..." if len(connected_vector.content) > 100 else connected_vector.content,
                    "similarity": similarity,
                    "context_type": connected_vector.context_type.value
                })
        
        return {
            "vector": {
                "id": str(vector.id),
                "content": vector.content,
                "context_type": vector.context_type.value,
                "scope": vector.scope.value,
                "importance": vector.importance,
                "last_accessed": vector.last_accessed.isoformat(),
                "access_count": vector.access_count,
                "related_entities": vector.related_entities,
                "metadata": vector.metadata
            },
            "connections": connection_info,
            "connection_count": len(connections),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting vector: {str(e)}")


@router.delete("/context/vectors/{vector_id}")
async def delete_context_vector(
    vector_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a specific context vector
    
    Removes a context vector from the persistent context system
    and updates all related connections.
    """
    try:
        manager = get_persistent_context_manager(db)
        
        if vector_id not in manager.context_graph.nodes:
            raise HTTPException(status_code=404, detail="Context vector not found")
        
        # Remove from nodes
        del manager.context_graph.nodes[vector_id]
        
        # Remove from edges
        if vector_id in manager.context_graph.edges:
            del manager.context_graph.edges[vector_id]
        
        # Remove from other vectors' edges
        for edges in manager.context_graph.edges.values():
            edges[:] = [(node_id, weight) for node_id, weight in edges if node_id != vector_id]
        
        # Remove from clusters
        for cluster in manager.context_graph.clusters.values():
            cluster.vectors = [v for v in cluster.vectors if v.id != vector_id]
        
        # Save changes
        await manager._save_graph_to_cache()
        
        return {
            "message": "Context vector deleted successfully",
            "vector_id": str(vector_id),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting vector: {str(e)}")