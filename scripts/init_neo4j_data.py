#!/usr/bin/env python3
"""
Initialize Neo4j with sample data for KnowledgeHub
"""

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_initial_data(driver):
    """Create initial nodes and relationships in Neo4j"""
    
    with driver.session() as session:
        # Clear existing data
        logger.info("Clearing existing data...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create nodes
        logger.info("Creating initial nodes...")
        
        # Project node
        session.run("""
            CREATE (p:Project {
                id: 'knowledgehub-1',
                name: 'KnowledgeHub',
                description: 'AI-powered knowledge management system',
                created: datetime('2024-01-01T00:00:00Z')
            })
        """)
        
        # Memory System
        session.run("""
            CREATE (m:Memory {
                id: 'memory-system-1',
                name: 'Memory System',
                type: 'persistent',
                version: '2.0',
                features: ['session-continuity', 'mistake-learning', 'pattern-recognition']
            })
        """)
        
        # AI Intelligence
        session.run("""
            CREATE (ai:Pattern {
                id: 'ai-intelligence-1',
                name: 'AI Intelligence',
                version: '2.0',
                capabilities: ['predictive-analytics', 'workflow-automation', 'decision-reasoning']
            })
        """)
        
        # Session Management
        session.run("""
            CREATE (s:Session {
                id: 'session-mgmt-1',
                name: 'Session Management',
                active: true,
                features: ['context-preservation', 'session-linking', 'handoff']
            })
        """)
        
        # Error Learning
        session.run("""
            CREATE (e:Error {
                id: 'error-learning-1',
                name: 'Error Learning',
                count: 15,
                patterns_identified: 8,
                solutions_recorded: 12
            })
        """)
        
        # Decision Tracking
        session.run("""
            CREATE (d:Decision {
                id: 'decision-tracking-1',
                name: 'Decision Tracking',
                total: 42,
                categories: ['technical', 'architectural', 'performance']
            })
        """)
        
        # Code Evolution
        session.run("""
            CREATE (c:Code {
                id: 'code-evolution-1',
                name: 'Code Evolution',
                files: 125,
                refactorings: 34,
                patterns_detected: 18
            })
        """)
        
        # Performance Metrics
        session.run("""
            CREATE (w:Workflow {
                id: 'performance-metrics-1',
                name: 'Performance Metrics',
                optimized: true,
                metrics_tracked: 15,
                improvements: 23
            })
        """)
        
        # Create relationships
        logger.info("Creating relationships...")
        
        relationships = [
            ("MATCH (p:Project {id: 'knowledgehub-1'}), (m:Memory {id: 'memory-system-1'}) CREATE (p)-[:CONTAINS]->(m)"),
            ("MATCH (p:Project {id: 'knowledgehub-1'}), (ai:Pattern {id: 'ai-intelligence-1'}) CREATE (p)-[:IMPLEMENTS]->(ai)"),
            ("MATCH (m:Memory {id: 'memory-system-1'}), (s:Session {id: 'session-mgmt-1'}) CREATE (m)-[:MANAGES]->(s)"),
            ("MATCH (ai:Pattern {id: 'ai-intelligence-1'}), (e:Error {id: 'error-learning-1'}) CREATE (ai)-[:LEARNS_FROM]->(e)"),
            ("MATCH (ai:Pattern {id: 'ai-intelligence-1'}), (d:Decision {id: 'decision-tracking-1'}) CREATE (ai)-[:TRACKS]->(d)"),
            ("MATCH (s:Session {id: 'session-mgmt-1'}), (c:Code {id: 'code-evolution-1'}) CREATE (s)-[:MONITORS]->(c)"),
            ("MATCH (d:Decision {id: 'decision-tracking-1'}), (w:Workflow {id: 'performance-metrics-1'}) CREATE (d)-[:OPTIMIZES]->(w)"),
            ("MATCH (e:Error {id: 'error-learning-1'}), (w:Workflow {id: 'performance-metrics-1'}) CREATE (e)-[:IMPROVES]->(w)")
        ]
        
        for query in relationships:
            session.run(query)
        
        # Verify creation
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        node_count = result.single()["node_count"]
        
        result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
        rel_count = result.single()["rel_count"]
        
        logger.info(f"Created {node_count} nodes and {rel_count} relationships")


def main():
    """Main function"""
    # Connect to Neo4j
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "knowledgehub123"
    
    logger.info(f"Connecting to Neo4j at {uri}...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully")
        
        # Create initial data
        create_initial_data(driver)
        
        driver.close()
        logger.info("Neo4j initialization complete!")
        
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")
        raise


if __name__ == "__main__":
    main()