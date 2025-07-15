"""
Knowledge Graph Service using Neo4j.

This service provides graph-based storage and querying for knowledge relationships,
decision dependencies, and impact analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import uuid
from enum import Enum

from neo4j import GraphDatabase, Result
from neo4j.exceptions import Neo4jError
import networkx as nx

from ...shared.config import Config
from ...shared.logging import setup_logging

logger = setup_logging("knowledge_graph")


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    DECISION = "Decision"
    ENTITY = "Entity"
    CONCEPT = "Concept"
    CODE = "Code"
    PATTERN = "Pattern"
    ERROR = "Error"
    SOLUTION = "Solution"
    PROJECT = "Project"
    USER = "User"
    SESSION = "Session"
    MEMORY = "Memory"
    DOCUMENT = "Document"


class RelationType(Enum):
    """Types of relationships in the knowledge graph"""
    DEPENDS_ON = "DEPENDS_ON"
    IMPACTS = "IMPACTS"
    RELATES_TO = "RELATES_TO"
    IMPLEMENTS = "IMPLEMENTS"
    SOLVES = "SOLVES"
    CAUSES = "CAUSES"
    MENTIONS = "MENTIONS"
    AUTHORED_BY = "AUTHORED_BY"
    BELONGS_TO = "BELONGS_TO"
    FOLLOWS = "FOLLOWS"
    DERIVED_FROM = "DERIVED_FROM"
    SIMILAR_TO = "SIMILAR_TO"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    EVOLVED_FROM = "EVOLVED_FROM"
    LEARNED_FROM = "LEARNED_FROM"


class KnowledgeGraphService:
    """
    Service for managing knowledge graph using Neo4j.
    
    Features:
    - Node and relationship management
    - Complex graph queries
    - Path finding and impact analysis
    - Pattern detection
    - Graph visualization data
    """
    
    def __init__(self, 
                 uri: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initialize the knowledge graph service.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.config = Config()
        
        # Neo4j connection settings
        self.uri = uri or self.config.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or self.config.get("NEO4J_USER", "neo4j")
        self.password = password or self.config.get("NEO4J_PASSWORD", "password")
        
        self.driver = None
        self._initialized = False
        
        logger.info(f"Initialized KnowledgeGraphService with URI: {self.uri}")
    
    async def initialize(self):
        """Initialize Neo4j connection and create constraints/indexes"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            
            # Verify connectivity
            self.driver.verify_connectivity()
            
            # Create constraints and indexes
            await self._create_constraints()
            
            self._initialized = True
            logger.info("Successfully connected to Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {str(e)}")
            raise
    
    async def _create_constraints(self):
        """Create database constraints and indexes"""
        constraints = [
            # Unique constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Decision) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Code) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Pattern) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Error) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Solution) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Project) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:User) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Session) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Memory) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            
            # Indexes for common queries
            "CREATE INDEX IF NOT EXISTS FOR (n:Decision) ON (n.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Pattern) ON (n.category)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Error) ON (n.severity)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Memory) ON (n.session_id)",
            "CREATE FULLTEXT INDEX decisionSearch IF NOT EXISTS FOR (n:Decision) ON EACH [n.description, n.reasoning]",
            "CREATE FULLTEXT INDEX entitySearch IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.description]"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint/index: {constraint[:50]}...")
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {str(e)}")
    
    async def create_node(self,
                         node_type: NodeType,
                         properties: Dict[str, Any]) -> str:
        """
        Create a node in the knowledge graph.
        
        Args:
            node_type: Type of node to create
            properties: Node properties
            
        Returns:
            Node ID
        """
        if not properties.get('id'):
            properties['id'] = str(uuid.uuid4())
        
        # Add timestamps
        properties['created_at'] = datetime.utcnow().isoformat()
        properties['updated_at'] = properties['created_at']
        
        query = f"""
        CREATE (n:{node_type.value} $properties)
        RETURN n.id as id
        """
        
        with self.driver.session() as session:
            result = session.run(query, properties=properties)
            record = result.single()
            
            if record:
                logger.info(f"Created {node_type.value} node: {record['id']}")
                return record['id']
            else:
                raise Exception("Failed to create node")
    
    async def create_relationship(self,
                                from_id: str,
                                to_id: str,
                                rel_type: RelationType,
                                properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            rel_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            Success status
        """
        props = properties or {}
        props['created_at'] = datetime.utcnow().isoformat()
        
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        CREATE (a)-[r:{rel_type.value} $properties]->(b)
        RETURN r
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                from_id=from_id,
                to_id=to_id,
                properties=props
            )
            
            if result.single():
                logger.info(f"Created relationship {rel_type.value}: {from_id} -> {to_id}")
                return True
            return False
    
    async def find_dependencies(self,
                              node_id: str,
                              depth: int = 3) -> Dict[str, Any]:
        """
        Find all dependencies of a node.
        
        Args:
            node_id: Node ID to analyze
            depth: Maximum depth to traverse
            
        Returns:
            Dependency tree
        """
        query = f"""
        MATCH path = (start {{id: $node_id}})-[:DEPENDS_ON*1..{depth}]->(dep)
        RETURN path
        """
        
        nodes = {}
        edges = []
        
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            
            for record in result:
                path = record['path']
                
                # Extract nodes
                for node in path.nodes:
                    node_dict = dict(node)
                    nodes[node_dict['id']] = {
                        'id': node_dict['id'],
                        'type': list(node.labels)[0],
                        'properties': node_dict
                    }
                
                # Extract relationships
                for rel in path.relationships:
                    edges.append({
                        'from': rel.start_node['id'],
                        'to': rel.end_node['id'],
                        'type': rel.type,
                        'properties': dict(rel)
                    })
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges
        }
    
    async def impact_analysis(self,
                            node_id: str,
                            impact_types: Optional[List[RelationType]] = None,
                            max_depth: int = 5) -> Dict[str, Any]:
        """
        Analyze the impact of changes to a node.
        
        Args:
            node_id: Node ID to analyze
            impact_types: Types of relationships to follow
            max_depth: Maximum depth to traverse
            
        Returns:
            Impact analysis results
        """
        if not impact_types:
            impact_types = [RelationType.IMPACTS, RelationType.DEPENDS_ON, RelationType.CAUSES]
        
        rel_pattern = '|'.join([rt.value for rt in impact_types])
        
        query = f"""
        MATCH (start {{id: $node_id}})
        CALL apoc.path.subgraphAll(start, {{
            relationshipFilter: "{rel_pattern}",
            maxLevel: {max_depth}
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        
        impacted_nodes = {}
        impact_paths = []
        
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            if record:
                # Process nodes
                for node in record['nodes']:
                    node_dict = dict(node)
                    node_id = node_dict['id']
                    impacted_nodes[node_id] = {
                        'id': node_id,
                        'type': list(node.labels)[0],
                        'properties': node_dict,
                        'impact_score': 0  # Will calculate based on distance
                    }
                
                # Process relationships to build paths
                for rel in record['relationships']:
                    impact_paths.append({
                        'from': rel.start_node['id'],
                        'to': rel.end_node['id'],
                        'type': rel.type,
                        'properties': dict(rel)
                    })
        
        # Calculate impact scores based on distance
        if impacted_nodes:
            await self._calculate_impact_scores(node_id, impacted_nodes, impact_paths)
        
        return {
            'total_impacted': len(impacted_nodes) - 1,  # Exclude the starting node
            'impacted_nodes': list(impacted_nodes.values()),
            'impact_paths': impact_paths,
            'critical_nodes': [n for n in impacted_nodes.values() if n['impact_score'] > 0.7]
        }
    
    async def _calculate_impact_scores(self,
                                     start_id: str,
                                     nodes: Dict[str, Any],
                                     paths: List[Dict[str, Any]]):
        """Calculate impact scores based on graph distance"""
        # Build adjacency list
        graph = nx.DiGraph()
        for path in paths:
            graph.add_edge(path['from'], path['to'])
        
        # Calculate shortest paths from start node
        if start_id in graph:
            distances = nx.single_source_shortest_path_length(graph, start_id)
            
            for node_id, node_data in nodes.items():
                if node_id in distances:
                    distance = distances[node_id]
                    # Impact decreases with distance
                    node_data['impact_score'] = 1.0 / (1.0 + distance)
    
    async def find_patterns(self,
                          pattern_type: str,
                          min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Find recurring patterns in the knowledge graph.
        
        Args:
            pattern_type: Type of pattern to search for
            min_occurrences: Minimum number of occurrences
            
        Returns:
            List of patterns found
        """
        patterns = []
        
        # Example: Find decision chains
        if pattern_type == "decision_chain":
            query = """
            MATCH path = (d1:Decision)-[:DEPENDS_ON]->(d2:Decision)-[:DEPENDS_ON]->(d3:Decision)
            WITH d1, d2, d3, count(path) as occurrences
            WHERE occurrences >= $min_occurrences
            RETURN d1, d2, d3, occurrences
            ORDER BY occurrences DESC
            """
            
            with self.driver.session() as session:
                result = session.run(query, min_occurrences=min_occurrences)
                
                for record in result:
                    patterns.append({
                        'type': 'decision_chain',
                        'nodes': [
                            dict(record['d1']),
                            dict(record['d2']),
                            dict(record['d3'])
                        ],
                        'occurrences': record['occurrences']
                    })
        
        # Example: Find error-solution patterns
        elif pattern_type == "error_solution":
            query = """
            MATCH (e:Error)-[:SOLVED_BY]->(s:Solution)
            WITH e.type as error_type, collect(s) as solutions, count(s) as solution_count
            WHERE solution_count >= $min_occurrences
            RETURN error_type, solutions, solution_count
            ORDER BY solution_count DESC
            """
            
            with self.driver.session() as session:
                result = session.run(query, min_occurrences=min_occurrences)
                
                for record in result:
                    patterns.append({
                        'type': 'error_solution',
                        'error_type': record['error_type'],
                        'solutions': [dict(s) for s in record['solutions']],
                        'occurrences': record['solution_count']
                    })
        
        return patterns
    
    async def get_knowledge_context(self,
                                  node_id: str,
                                  context_depth: int = 2) -> Dict[str, Any]:
        """
        Get the full context around a knowledge node.
        
        Args:
            node_id: Central node ID
            context_depth: How many hops to include
            
        Returns:
            Context information
        """
        query = f"""
        MATCH (center {{id: $node_id}})
        OPTIONAL MATCH (center)-[r*1..{context_depth}]-(connected)
        WITH center, collect(DISTINCT connected) as connected_nodes, 
             collect(DISTINCT r) as relationships
        RETURN center, connected_nodes, relationships
        """
        
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            if not record:
                return {}
            
            center = dict(record['center'])
            connected = [dict(n) for n in record['connected_nodes'] if n]
            
            # Build context summary
            context = {
                'center': {
                    'id': center['id'],
                    'type': list(record['center'].labels)[0],
                    'properties': center
                },
                'connected_nodes': connected,
                'statistics': {
                    'total_connections': len(connected),
                    'node_types': self._count_node_types(record['connected_nodes']),
                    'relationship_types': self._count_relationship_types(record['relationships'])
                }
            }
            
            return context
    
    def _count_node_types(self, nodes) -> Dict[str, int]:
        """Count occurrences of each node type"""
        counts = {}
        for node in nodes:
            if node:
                for label in node.labels:
                    counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _count_relationship_types(self, relationships) -> Dict[str, int]:
        """Count occurrences of each relationship type"""
        counts = {}
        for rel_list in relationships:
            if rel_list:
                for rel in rel_list:
                    counts[rel.type] = counts.get(rel.type, 0) + 1
        return counts
    
    async def evolve_knowledge(self,
                             old_node_id: str,
                             new_node_id: str,
                             evolution_type: str = "update") -> bool:
        """
        Track knowledge evolution over time.
        
        Args:
            old_node_id: Previous version node ID
            new_node_id: New version node ID
            evolution_type: Type of evolution
            
        Returns:
            Success status
        """
        return await self.create_relationship(
            old_node_id,
            new_node_id,
            RelationType.EVOLVED_FROM,
            {'type': evolution_type, 'timestamp': datetime.utcnow().isoformat()}
        )
    
    async def search_graph(self,
                         query_text: str,
                         node_types: Optional[List[NodeType]] = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search across the knowledge graph.
        
        Args:
            query_text: Search query
            node_types: Types of nodes to search
            limit: Maximum results
            
        Returns:
            Search results
        """
        if node_types:
            labels = ':'.join([nt.value for nt in node_types])
            query = f"""
            CALL db.index.fulltext.queryNodes('decisionSearch', $query_text) 
            YIELD node, score
            WHERE any(label in labels(node) WHERE label IN $labels)
            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
            """
            label_list = [nt.value for nt in node_types]
        else:
            query = """
            CALL db.index.fulltext.queryNodes('decisionSearch', $query_text) 
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
            """
            label_list = []
        
        results = []
        with self.driver.session() as session:
            if node_types:
                result = session.run(
                    query,
                    query_text=query_text,
                    labels=label_list,
                    limit=limit
                )
            else:
                result = session.run(
                    query,
                    query_text=query_text,
                    limit=limit
                )
            
            for record in result:
                node = record['node']
                results.append({
                    'id': node['id'],
                    'type': list(node.labels)[0],
                    'properties': dict(node),
                    'score': record['score']
                })
        
        return results
    
    async def get_graph_visualization(self,
                                    center_id: Optional[str] = None,
                                    max_nodes: int = 100) -> Dict[str, Any]:
        """
        Get graph data formatted for visualization.
        
        Args:
            center_id: Optional center node ID
            max_nodes: Maximum nodes to return
            
        Returns:
            Visualization data
        """
        if center_id:
            query = """
            MATCH (center {id: $center_id})
            OPTIONAL MATCH path = (center)-[r*1..3]-(connected)
            WITH center, connected, r
            LIMIT $max_nodes
            RETURN collect(DISTINCT center) + collect(DISTINCT connected) as nodes,
                   collect(DISTINCT r) as relationships
            """
            params = {'center_id': center_id, 'max_nodes': max_nodes}
        else:
            query = """
            MATCH (n)
            WITH n LIMIT $max_nodes
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE m IN nodes
            RETURN collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships
            """
            params = {'max_nodes': max_nodes}
        
        with self.driver.session() as session:
            result = session.run(query, **params)
            record = result.single()
            
            if not record:
                return {'nodes': [], 'links': []}
            
            # Format for D3.js or similar visualization
            nodes = []
            node_map = {}
            
            for node in record['nodes']:
                if node:
                    node_id = node['id']
                    node_data = {
                        'id': node_id,
                        'label': node.get('name', node.get('description', node_id)[:30]),
                        'type': list(node.labels)[0],
                        'properties': dict(node)
                    }
                    nodes.append(node_data)
                    node_map[node_id] = len(nodes) - 1
            
            links = []
            for rel_list in record['relationships']:
                if rel_list:
                    for rel in rel_list:
                        if rel.start_node['id'] in node_map and rel.end_node['id'] in node_map:
                            links.append({
                                'source': node_map[rel.start_node['id']],
                                'target': node_map[rel.end_node['id']],
                                'type': rel.type,
                                'properties': dict(rel)
                            })
            
            return {
                'nodes': nodes,
                'links': links,
                'statistics': {
                    'total_nodes': len(nodes),
                    'total_links': len(links),
                    'node_types': self._count_types([n['type'] for n in nodes])
                }
            }
    
    def _count_types(self, types: List[str]) -> Dict[str, int]:
        """Count occurrences of types"""
        counts = {}
        for t in types:
            counts[t] = counts.get(t, 0) + 1
        return counts
    
    async def cleanup(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")