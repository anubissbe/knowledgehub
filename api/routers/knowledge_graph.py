"""
Knowledge Graph API router.

Provides endpoints for graph-based knowledge management, relationship tracking,
and impact analysis.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import uuid

from ..services.knowledge_graph import (
    KnowledgeGraphService,
    NodeType,
    RelationType
)
from ..dependencies import get_current_user
from ...shared.logging import setup_logging

logger = setup_logging("api.knowledge_graph")

router = APIRouter(prefix="/api/knowledge-graph", tags=["knowledge-graph"])


class CreateNodeRequest(BaseModel):
    """Request to create a node"""
    node_type: str = Field(..., description="Type of node (Decision, Entity, etc)")
    properties: Dict[str, Any] = Field(..., description="Node properties")


class CreateRelationshipRequest(BaseModel):
    """Request to create a relationship"""
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Type of relationship")
    properties: Optional[Dict[str, Any]] = Field(None, description="Relationship properties")


class DependencyAnalysisResponse(BaseModel):
    """Response for dependency analysis"""
    nodes: List[Dict[str, Any]] = Field(..., description="Nodes in dependency tree")
    edges: List[Dict[str, Any]] = Field(..., description="Edges in dependency tree")


class ImpactAnalysisResponse(BaseModel):
    """Response for impact analysis"""
    total_impacted: int = Field(..., description="Total number of impacted nodes")
    impacted_nodes: List[Dict[str, Any]] = Field(..., description="List of impacted nodes")
    impact_paths: List[Dict[str, Any]] = Field(..., description="Paths of impact")
    critical_nodes: List[Dict[str, Any]] = Field(..., description="Critically impacted nodes")


class PatternSearchResponse(BaseModel):
    """Response for pattern search"""
    patterns: List[Dict[str, Any]] = Field(..., description="Found patterns")
    count: int = Field(..., description="Total patterns found")


class GraphSearchResponse(BaseModel):
    """Response for graph search"""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total results")


class GraphVisualizationResponse(BaseModel):
    """Response for graph visualization"""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    links: List[Dict[str, Any]] = Field(..., description="Graph links")
    statistics: Dict[str, Any] = Field(..., description="Graph statistics")


# Initialize service
knowledge_graph_service = None


async def get_knowledge_graph_service():
    """Get or create knowledge graph service instance"""
    global knowledge_graph_service
    if knowledge_graph_service is None:
        knowledge_graph_service = KnowledgeGraphService()
        await knowledge_graph_service.initialize()
    return knowledge_graph_service


@router.post("/nodes", response_model=Dict[str, str])
async def create_node(
    request: CreateNodeRequest,
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """
    Create a new node in the knowledge graph.
    
    Node types: Decision, Entity, Concept, Code, Pattern, Error, Solution, Project, User, Session, Memory, Document
    """
    try:
        # Validate node type
        try:
            node_type = NodeType(request.node_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid node type: {request.node_type}"
            )
        
        # Add user context
        request.properties['created_by'] = current_user['id']
        
        # Create node
        node_id = await service.create_node(node_type, request.properties)
        
        return {"node_id": node_id}
        
    except Exception as e:
        logger.error(f"Error creating node: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationships")
async def create_relationship(
    request: CreateRelationshipRequest,
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """
    Create a relationship between two nodes.
    
    Relationship types: DEPENDS_ON, IMPACTS, RELATES_TO, IMPLEMENTS, SOLVES, CAUSES, etc.
    """
    try:
        # Validate relationship type
        try:
            rel_type = RelationType(request.relationship_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid relationship type: {request.relationship_type}"
            )
        
        # Add metadata
        if request.properties is None:
            request.properties = {}
        request.properties['created_by'] = current_user['id']
        
        # Create relationship
        success = await service.create_relationship(
            request.from_id,
            request.to_id,
            rel_type,
            request.properties
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="One or both nodes not found"
            )
        
        return {"success": success}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating relationship: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}/dependencies", response_model=DependencyAnalysisResponse)
async def get_dependencies(
    node_id: str,
    depth: int = Query(3, ge=1, le=10, description="Maximum traversal depth"),
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """Get all dependencies of a node."""
    try:
        dependencies = await service.find_dependencies(node_id, depth)
        return DependencyAnalysisResponse(**dependencies)
        
    except Exception as e:
        logger.error(f"Error getting dependencies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}/impact", response_model=ImpactAnalysisResponse)
async def analyze_impact(
    node_id: str,
    max_depth: int = Query(5, ge=1, le=10, description="Maximum traversal depth"),
    impact_types: Optional[List[str]] = Query(None, description="Types of impact to analyze"),
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """Analyze the impact of changes to a node."""
    try:
        # Convert impact types
        rel_types = None
        if impact_types:
            rel_types = []
            for it in impact_types:
                try:
                    rel_types.append(RelationType(it))
                except ValueError:
                    logger.warning(f"Unknown impact type: {it}")
        
        impact = await service.impact_analysis(node_id, rel_types, max_depth)
        return ImpactAnalysisResponse(**impact)
        
    except Exception as e:
        logger.error(f"Error analyzing impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}/context")
async def get_node_context(
    node_id: str,
    context_depth: int = Query(2, ge=1, le=5, description="Context depth"),
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """Get the full context around a knowledge node."""
    try:
        context = await service.get_knowledge_context(node_id, context_depth)
        if not context:
            raise HTTPException(status_code=404, detail="Node not found")
        
        return context
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{pattern_type}", response_model=PatternSearchResponse)
async def find_patterns(
    pattern_type: str,
    min_occurrences: int = Query(2, ge=1, description="Minimum pattern occurrences"),
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """
    Find recurring patterns in the knowledge graph.
    
    Pattern types: decision_chain, error_solution, concept_cluster
    """
    try:
        patterns = await service.find_patterns(pattern_type, min_occurrences)
        
        return PatternSearchResponse(
            patterns=patterns,
            count=len(patterns)
        )
        
    except Exception as e:
        logger.error(f"Error finding patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=GraphSearchResponse)
async def search_graph(
    query: str = Query(..., description="Search query"),
    node_types: Optional[List[str]] = Query(None, description="Node types to search"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """Full-text search across the knowledge graph."""
    try:
        # Convert node types
        types = None
        if node_types:
            types = []
            for nt in node_types:
                try:
                    types.append(NodeType(nt))
                except ValueError:
                    logger.warning(f"Unknown node type: {nt}")
        
        results = await service.search_graph(query, types, limit)
        
        return GraphSearchResponse(
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error searching graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization", response_model=GraphVisualizationResponse)
async def get_visualization(
    center_id: Optional[str] = Query(None, description="Center node ID"),
    max_nodes: int = Query(100, ge=10, le=500, description="Maximum nodes"),
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """Get graph data formatted for visualization."""
    try:
        viz_data = await service.get_graph_visualization(center_id, max_nodes)
        return GraphVisualizationResponse(**viz_data)
        
    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolve")
async def track_evolution(
    old_node_id: str,
    new_node_id: str,
    evolution_type: str = "update",
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """Track knowledge evolution from one node to another."""
    try:
        success = await service.evolve_knowledge(
            old_node_id,
            new_node_id,
            evolution_type
        )
        
        return {"success": success}
        
    except Exception as e:
        logger.error(f"Error tracking evolution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_graph_statistics(
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """Get overall graph statistics."""
    try:
        # Query for statistics
        query = """
        MATCH (n)
        WITH labels(n) as node_labels
        UNWIND node_labels as label
        WITH label, count(*) as count
        RETURN collect({type: label, count: count}) as node_stats
        """
        
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
        """
        
        stats = {
            'nodes': {},
            'relationships': {},
            'total_nodes': 0,
            'total_relationships': 0
        }
        
        with service.driver.session() as session:
            # Node statistics
            result = session.run(query)
            record = result.single()
            if record and record['node_stats']:
                for stat in record['node_stats']:
                    stats['nodes'][stat['type']] = stat['count']
                    stats['total_nodes'] += stat['count']
            
            # Relationship statistics
            result = session.run(rel_query)
            for record in result:
                stats['relationships'][record['type']] = record['count']
                stats['total_relationships'] += record['count']
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", status_code=204)
async def clear_graph(
    confirm: bool = Query(False, description="Confirm graph deletion"),
    current_user: dict = Depends(get_current_user),
    service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    """Clear all data from the knowledge graph (DANGEROUS!)."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm=true to clear graph"
            )
        
        # Only allow admins
        if current_user.get('role') != 'admin':
            raise HTTPException(
                status_code=403,
                detail="Only admins can clear the graph"
            )
        
        with service.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        logger.warning(f"Graph cleared by user {current_user['id']}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))